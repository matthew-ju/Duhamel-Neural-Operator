#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""abg_model.py -- a (alpha,beta,gamma)-conditioned neural split-step operator.

    x_{t+1}  =  NonlinearFNO_c ( SpectralSSM_c(theta) x_t )

Both halves are neural and both are conditioned on the SAME continuous vector
c = c(alpha,beta,gamma) from abg_laws.  This replaces the two hand-built pieces
the project shipped at fixed alpha -- a per-material least-squares propagator
bank and a specialist FNO -- with one operator that is a function of the
material.

WHY BOTH HALVES, AND WHY THIS SPLIT
-----------------------------------
Measured on this system (T400_SHOWDOWN_REPORT_v3): the linear propagator alone
is exact to corr 0.9993 through frame 55 and then diverges to rel L2 1.5e18 by
frame 400 -- it is a growth operator with no saturation mechanism.  The FNO
alone reaches corr 0.973 at frame 400 but 0.9984 at frame 55, because the
incubation phase is 1000x below saturation and a single residual map cannot
cover both.  Composed, they gave the production pipeline's 0.9891.  So the
split is not architectural taste; each half covers the regime the other fails
in, and this file makes the composition ONE trainable model instead of a
hand-off at a tuned frame.

------------------------------------------------------------------------------
1. THE SPECTRAL HALF: LEARN THE GENERATOR, NOT THE PROPAGATOR
------------------------------------------------------------------------------
The linearised FerroX dynamics is diagonal in the periodic (kx,ky) plane and
dense across the 10 non-periodic FE z-layers, because the Poisson/depolarisation
coupling is non-local in z.  incubation_propagator.py fits those 4096 blocks
directly by ridge least squares -- and says in its own header that a parameter
sweep needs a refit per material, which is the thing this campaign has to
remove.

Three constraints turn that table into a conditioned operator:

  (a) THE SYMBOL IS A FUNCTION, NOT A TABLE.  The blocks are emitted by an MLP
      of the DIMENSIONLESS wavenumber kappa = |k| mu(alpha) rather than stored
      per grid slot.  A table holds the right numbers in the wrong places as
      soon as a coefficient moves the physical wavenumber attached to each
      slot; a function of kappa moves with it.  The MLP is also fed the
      DISCRETE symbol (4 sin^2(k h /2) summed over x,y, in units of mu) so it
      can represent the grid's own anisotropy at high k, which the continuum
      kappa cannot.

  (b) GENERATOR, NOT PROPAGATOR.  The net emits L(kappa,c) and the model
      applies exp(theta L).  Then m frames is exp(m theta L) EXACTLY -- the
      semigroup is exact by construction, every fractional or multiple step is
      equally well conditioned, and strided training pairs are exact samples at
      m*theta rather than approximations.  Fitting the one-frame propagator
      instead makes A^(1/4) or A^4 numerically hopeless on a non-normal A: the
      previous campaign measured ||A|| = 1.91 against ||A^(1/4)|| = 15.9.

      Exactness survives the spectrum cap because the cap is on L, not on
      theta*L -- see the w_grow/w_decay note below.  An earlier version capped
      the EXPONENT, which broke exactness at precisely the strided pairs the
      architecture exists to make exact; abg_smoke.py caught it, and now
      measures the semigroup residual at the largest theta and the largest
      stride the campaign will actually hit rather than at a convenient one.

  (c) SYMMETRY OF THE GENERATOR.  TDGL is a gradient flow and the electrostatic
      coupling is self-adjoint, so L(k) is REAL SYMMETRIC.  We build it as
      L = Q diag(w) Q^T with Q orthogonal from a CAYLEY transform of a skew
      matrix.  Cayley and not eigh: eigh fails outright on the degenerate
      isotropic initialisation and is non-differentiable at eigenvalue
      crossings (measured in pi_net.py).  Symmetry is 55 numbers per mode
      instead of 200 and rules out the spurious rotating modes an unconstrained
      block learns from short windows.

------------------------------------------------------------------------------
2. THE NONLINEAR HALF: FNO IN REDUCED UNITS, GATED BY c
------------------------------------------------------------------------------
The FNO is the tuned production backbone (ampnorm + full-band residual).  What
is new is that it sees the field in the material's OWN units, v = P/P_s, and
that every conditioning parameter is gated by c so it vanishes at the reference
material.  The alpha campaign established both the necessity (an unpinned
adapter walks off the specialist optimum in epoch 1 and never returns) and the
mechanism for the failure mode this design is built to avoid: a conditioner fed
a coefficient that never varies in training receives ZERO gradient on that
input column, so at test time a nonzero value multiplies untrained weights and
the model is worse than if it had never been told.  Conditioning on the
INVARIANTS rather than the coefficients is what makes the gamma direction
reachable from beta-only training.

------------------------------------------------------------------------------
3. THE ARMS THIS FILE SUPPORTS
------------------------------------------------------------------------------
    cond_mode  = pi   | raw | none      what the conditioner is told
    units      = True | False           whether the state is reduced by P_s
    tau_scale  = True | False           whether the move is scaled by theta
so the ablation grid that separates "a change of units" from "a learned
invariant" is a matter of flags, not of separate models.
"""
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import abg_laws as al

Z_FE, H, W = 10, 64, 64


def _steps_t(steps, B, device):
    """Broadcast the stride to (B,) floats.

    A window may be strided by m frames, and m is drawn PER SAMPLE, so a batch
    genuinely mixes step sizes.  Everything downstream multiplies theta by it,
    which is the only place the stride is allowed to enter: a strided pair is
    the same operator evaluated at a larger dimensionless time, not a different
    operator."""
    if torch.is_tensor(steps):
        return steps.to(device=device, dtype=torch.float32).reshape(-1).expand(B)
    return torch.full((B,), float(steps), device=device, dtype=torch.float32)


# ===========================================================================
# Conditioning trunk
# ===========================================================================
class CondTrunk(nn.Module):
    """c (B,d_in) -> embedding (B,d_out), and EXACTLY zero when c is zero.

    Bias-free throughout and built from an activation with f(0)=0, so
    emb(0) == 0 in IEEE for any weights.  That is what pins the reference
    material: every head downstream is a linear map of this embedding.
    """

    def __init__(self, d_in, d_out=32, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden, bias=False), nn.Tanh(),
            nn.Linear(hidden, hidden, bias=False), nn.Tanh(),
            nn.Linear(hidden, d_out, bias=False))
        self.d_out = d_out

    def forward(self, c):
        return self.net(c)


# ===========================================================================
# Spectral half
# ===========================================================================
class SpectralGenerator(nn.Module):
    """Conditioned Fourier state-space operator on the FE z-stack.

    forward(x, theta, c, steps=1) applies exp(steps * theta * L(kappa,c)) mode
    by mode, where x is (B,1,Z,H,W) real and theta is (B,) dimensionless.
    """

    def __init__(self, cond_dim, z=Z_FE, hidden=128, w_grow=1.5, w_decay=80.0,
                 gmax=24.0, d_max=2.0, prior="abg_symbol_prior.npz"):
        super().__init__()
        self.z = z
        self.n_skew = z * (z - 1) // 2
        # symbol features: kappa, kappa^2, the discrete symbol, and the two
        # axis-resolved discrete symbols (grid anisotropy), all dimensionless
        self.n_feat = 5
        self.mlp = nn.Sequential(
            nn.Linear(self.n_feat + cond_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, self.n_skew + z))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        # THE SPECTRUM CAP IS ASYMMETRIC, and that is physics rather than a
        # numerical band-aid.  lam is DEFINED as the e-folding time of the
        # linear instability, so in these units the largest growth rate the
        # operator can physically have is about 1; +1.5 leaves 50% headroom.
        # Decay, by contrast, is genuinely large at high wavenumber and must
        # not be capped tightly, hence -4.
        #
        # THE CAP IS ON THE SPECTRUM, NOT ON THE EXPONENT.  Capping the
        # exponent would break the one property this architecture exists to
        # have: with a strided window the exponent is theta*m*w, which at the
        # deep end of the legacy cadence (theta = 0.646) and stride 4 already
        # reaches 3.9 -- so an exponent cap tight enough to prevent a runaway
        # would BIND exactly on the strided pairs whose exactness is the reason
        # for striding.  Capping w instead bounds per-frame growth to
        # exp(theta*1.5) <= 2.6 while leaving exp(m theta L) = exp(theta L)^m
        # true for every m.  gmax is then pure overflow protection (float32
        # exp overflows near 88) and never a modelling choice.
        #
        # Why any cap at all: without one, an untrained symbol can amplify a
        # frame by e^8 in a single step.  The amplitude guard would catch it,
        # but a clamped output passes NO GRADIENT, so the arm would sit dead at
        # the guard instead of recovering.
        self.w_grow = w_grow
        self.w_decay = w_decay
        self.gmax = gmax
        self.d_max = d_max
        iu = torch.triu_indices(z, z, offset=1)
        self.register_buffer("iu", iu, persistent=False)
        # reference spectrum prior, radially tabulated in dimensionless kappa
        import os as _os
        import numpy as _np
        _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), prior)
        if _os.path.exists(_p):
            _d = _np.load(_p)
            self.register_buffer("pk", torch.tensor(_d["kappa"],
                                                    dtype=torch.float32))
            self.register_buffer("pL", torch.tensor(_d["Ls"],
                                                    dtype=torch.float32))
        else:   # no prior file -> zero base, i.e. the un-primed behaviour
            self.register_buffer("pk", torch.linspace(0, 10, 2))
            self.register_buffer("pL", torch.zeros(2, z, z))

    # -- symbol features ----------------------------------------------------
    def _kappa(self, mu_over_dx, device, dtype):
        """(B, H, Wf, n_feat) dimensionless wavenumber features.

        mu_over_dx is mu/dx per sample: the number of grid cells in one domain
        wall.  Multiplying the grid wavenumber (in units of 1/cell) by it gives
        |k| mu, which is the only wavenumber the continuum operator knows about.
        """
        kx = torch.fft.fftfreq(H, d=1.0, device=device, dtype=dtype) * 2 * math.pi
        ky = torch.fft.rfftfreq(W, d=1.0, device=device, dtype=dtype) * 2 * math.pi
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")            # (H,Wf)
        m = mu_over_dx.view(-1, 1, 1).to(dtype)
        # continuum
        k2 = (KX ** 2 + KY ** 2).unsqueeze(0) * m ** 2
        kap = torch.sqrt(k2)
        # discrete 5-point symbol, per axis and summed
        dx2 = (4 * torch.sin(KX / 2) ** 2).unsqueeze(0) * m ** 2
        dy2 = (4 * torch.sin(KY / 2) ** 2).unsqueeze(0) * m ** 2
        return torch.stack([kap, k2, dx2 + dy2, dx2, dy2], dim=-1)

    def symbol(self, mu_over_dx, cond, device, dtype=torch.float32):
        """(B,H,Wf,Z,Z) real symmetric generator L = prior + learned."""
        feat = self._kappa(mu_over_dx, device, dtype)             # (B,H,Wf,F)
        B, Hh, Wf, _ = feat.shape
        c = cond.to(dtype).view(B, 1, 1, -1).expand(B, Hh, Wf, cond.shape[-1])
        out = self.mlp(torch.cat([feat, c], dim=-1))              # (B,H,Wf,S+Z)
        s, w = out[..., :self.n_skew], out[..., self.n_skew:]
        S = out.new_zeros(B, Hh, Wf, self.z, self.z)
        S[..., self.iu[0], self.iu[1]] = s
        S = S - S.transpose(-1, -2)
        # Cayley: Q = (I - S)(I + S)^-1 is orthogonal for any skew S, is the
        # identity at S = 0 (the zero-initialised net), and is differentiable
        # everywhere -- none of which is true of an eigendecomposition.
        I = torch.eye(self.z, device=device, dtype=dtype).expand_as(S)
        Q = torch.linalg.solve(I + S, I - S, left=False)
        # THE REFERENCE OPERATOR IS GIVEN, THE CORRECTION IS LEARNED.
        # From a zero start the network had to rediscover the linear operator
        # before it could work on the part that is actually unknown; the
        # spectral-only arm was still at relL2 1.8 after 12 epochs.
        #
        # The prior is the reference material's symmetrised generator,
        # tabulated radially in kappa -- the OPERATOR, not its eigenvalues.  A
        # first attempt tabulated eigenvalues alone and applied them in the
        # z-basis, scoring corr -0.002: the true eigenvectors are set by the
        # depolarisation coupling across z and look nothing like the z-basis.
        #
        # It is also not the textbook 1 - kappa^2, which is qualitatively wrong
        # here.  The measured spectrum is a BAND-PASS instability: every
        # eigenvalue at kappa = 0 is negative (depolarisation kills the uniform
        # mode, which is why a critical alpha exists), growth is confined to
        # kappa in [0.40, 1.25] peaking at 0.73, and the tail reaches -63.
        #
        # The prior is data from ONE material -- the same single reference the
        # closed-form baseline is handed -- and every arm gets it, so the
        # conditioning comparison is untouched.  What stays learned is the
        # correction AND the entire material dependence: the prior is read at
        # THIS material's kappa, but nothing in it knows how the z-coupling
        # changes when the film is 3.5 rather than 10 wall-widths thick.
        kap = feat[..., 0]                                        # (B,H,Wf)
        g = self.pk
        j = torch.clamp(torch.searchsorted(g.contiguous(),
                                           kap.reshape(-1).contiguous()),
                        1, g.numel() - 1)
        x0, x1 = g[j - 1], g[j]
        f = ((kap.reshape(-1) - x0) / (x1 - x0).clamp_min(1e-30)
             ).clamp(0, 1).view(-1, 1, 1)
        L0 = (self.pL[j - 1] * (1 - f) + self.pL[j] * f
              ).reshape(*kap.shape, self.z, self.z)
        # learned symmetric correction, spectrally bounded by d_max so the
        # total spectrum cannot run away: |eig(L)| <= |eig(L0)| + d_max
        d = self.d_max * torch.tanh(w)
        Ld = torch.matmul(Q * d.unsqueeze(-2), Q.transpose(-1, -2))
        return L0 + Ld

    def forward(self, x, theta, cond, steps=1.0, mu_over_dx=None, cache=None):
        B = x.shape[0]
        dtype = torch.float32
        # The symbol depends only on the MATERIAL, not on the state, so it is
        # constant for a whole rollout -- and it is not cheap (a Cayley solve on
        # B x 64 x 33 matrices of 10x10).  Recomputing it per frame made the
        # 400-step eval dominated by an operation whose answer never changes.
        # The cache lives in the caller's material context and is dropped by
        # set_material, so it can never outlive the material it was built for;
        # during training it is rebuilt once per batch and the k-step window
        # backprops through the same graph, which is what we want anyway.
        # keyed on the batch size: set_material may be given scalars, which
        # _ctx_for then broadcasts, so the same material context can legally be
        # used at two different batch sizes within one eval.
        # The key must include the WHOLE stride vector, not a scalar: the
        # stride is drawn per sample, so a batch legitimately mixes strides and
        # exp(theta*m*L) differs per sample.  An earlier version keyed on
        # _steps_t(steps, 1, ...), which cannot even be built when the batch
        # carries eight different strides -- every arm died on it, while the
        # smoke test passed because all four of its checks used a SCALAR
        # stride.  There is now a check that does not.
        stv = _steps_t(steps, B, x.device)
        key = ("sym", B, tuple(round(float(v), 6) for v in stv))
        if cache is not None and key in cache:
            E = cache[key]
        else:
            L = self.symbol(mu_over_dx, cond, x.device, dtype)
            st = stv.view(B, 1, 1, 1, 1)
            # exp(theta L) directly: L is a real symmetric 10x10 per mode, so
            # matrix_exp is exact and differentiable, and the semigroup still
            # holds because the argument is linear in theta -- exp(m theta L)
            # equals exp(theta L)^m for the same L.  No eigendecomposition, so
            # no non-differentiable eigenvalue crossings.
            E = torch.matrix_exp(
                L * (theta.view(B, 1, 1, 1, 1).to(dtype) * st))
            if cache is not None:
                cache[key] = E
        # rfft over the periodic (H,W) plane; z stays in real space
        Xf = torch.fft.rfft2(x[:, 0].to(dtype), dim=(-2, -1))     # (B,Z,H,Wf)
        Xf = Xf.permute(0, 2, 3, 1).unsqueeze(-1)                 # (B,H,Wf,Z,1)
        # L is real, so real and imaginary parts go through the same matrix
        Yf = torch.complex(torch.matmul(E, Xf.real),
                           torch.matmul(E, Xf.imag)).squeeze(-1)
        Yf = Yf.permute(0, 3, 1, 2)                               # (B,Z,H,Wf)
        y = torch.fft.irfft2(Yf, s=(H, W), dim=(-2, -1))
        return y.unsqueeze(1).to(x.dtype)


# ===========================================================================
# Nonlinear half
# ===========================================================================
class CondFNO(nn.Module):
    """Tuned next-step FNO (ampnorm + full residual) with vector conditioning.

    Conditioning enters three ways, all inert at the reference material:
      * an extra INPUT channel per conditioning component, zero-initialised in
        the lifting layer;
      * bias-free FiLM gains/biases on every FNO block, bounded by tanh;
      * the amplitude channel is measured against THIS material's saturation,
        so the network is told how far along its OWN approach to saturation the
        field is rather than an absolute rms.
    """

    def __init__(self, cond_emb, modes_z=4, modes_xy=16, width=48, n_layers=4,
                 zpad=3, film_gmax=0.25, film_bmax=0.25, use_film=True):
        super().__init__()
        from neuralop.models import FNO
        self.width, self.n_layers, self.zpad = width, n_layers, zpad
        self.film_gmax, self.film_bmax = film_gmax, film_bmax
        self.use_film = use_film
        in_ch = 1 + 1 + 1 + cond_emb        # field, z-coord, log-amp, cond
        self.cond_emb = cond_emb
        self.fno = FNO(n_modes=(modes_z, modes_xy, modes_xy),
                       in_channels=in_ch, out_channels=2,
                       hidden_channels=width, n_layers=n_layers,
                       positional_embedding=None)
        # zero the conditioning columns of the lifting layer so the model at
        # initialisation is exactly the unconditioned one
        with torch.no_grad():
            self.fno.lifting.fcs[0].weight[:, -cond_emb:] = 0.0
        if use_film:
            self.film = nn.Sequential(
                nn.Linear(cond_emb, 64, bias=False), nn.GELU(),
                nn.Linear(64, 2 * n_layers * width, bias=False))
            nn.init.zeros_(self.film[-1].weight)

    def _fno_forward(self, hp, emb):
        fno, B = self.fno, hp.shape[0]
        gb = (self.film(emb).view(B, self.n_layers, 2, self.width)
              if self.use_film else None)
        h = fno.lifting(hp)
        if fno.domain_padding is not None:
            h = fno.domain_padding.pad(h)
        for i in range(fno.n_layers):
            h = fno.fno_blocks(h, i)
            if gb is not None:
                g = self.film_gmax * torch.tanh(gb[:, i, 0]).view(B, -1, 1, 1, 1)
                b = self.film_bmax * torch.tanh(gb[:, i, 1]).view(B, -1, 1, 1, 1)
                h = h * (1.0 + g) + b
        if fno.domain_padding is not None:
            h = fno.domain_padding.unpad(h)
        return fno.projection(h)

    def forward(self, x, emb):
        """x is already in REDUCED units. Returns (delta_hat, log_gain, s)."""
        B, _, Z, Hh, Ww = x.shape
        with torch.amp.autocast("cuda", enabled=False):
            xf = x.float()
            s = xf.pow(2).mean(dim=(1, 2, 3, 4), keepdim=True).sqrt().clamp_min(1e-12)
            inp = (xf / s).to(x.dtype)
            # in reduced units saturation is O(1) for every material, so this
            # channel says the same thing at every config -- which is the point
            logs = torch.log10(s).expand(B, 1, Z, Hh, Ww).to(x.dtype)
        zc = torch.linspace(0.0, 1.0, Z, device=x.device, dtype=x.dtype)
        feats = [inp, zc.view(1, 1, Z, 1, 1).expand(B, 1, Z, Hh, Ww), logs,
                 emb.view(B, self.cond_emb, 1, 1, 1).expand(B, self.cond_emb, Z, Hh, Ww)]
        h = torch.cat(feats, dim=1)
        hp = F.pad(h, (0, 0, 0, 0, self.zpad, self.zpad), mode="replicate")
        out = self._fno_forward(hp, emb)[:, :, self.zpad:-self.zpad]
        g = out[:, 1:].mean(dim=(1, 2, 3, 4), keepdim=True).clamp(-1.0, 0.5)
        return out[:, :1], g, s


# ===========================================================================
# The operator
# ===========================================================================
class ABGOperator(nn.Module):
    """One conditioned next-step operator for the whole (alpha,beta,gamma) family."""

    def __init__(self, cond_mode="pi", cond_emb=16, units=True, tau_scale=True,
                 use_ssm=True, use_fno=True, width=48, n_layers=4,
                 modes_xy=16, modes_z=4, ssm_hidden=128, use_film=True,
                 guard=1.35):
        super().__init__()
        self.cond_mode = cond_mode
        self.units = bool(units)
        self.tau_scale = bool(tau_scale)
        self.guard = float(guard)
        self._guard_abs = self.guard * float(
            al.p_sat(al.ALPHA_REF, al.BETA_REF, al.GAMMA_REF)) / al.P_NORM
        d_in = al.cond_dim(cond_mode)
        self.trunk = CondTrunk(d_in, d_out=cond_emb)
        self.ssm = (SpectralGenerator(cond_emb, hidden=ssm_hidden)
                    if use_ssm else None)
        self.fno = (CondFNO(cond_emb, width=width, n_layers=n_layers,
                            modes_xy=modes_xy, modes_z=modes_z,
                            use_film=use_film) if use_fno else None)
        self._ctx = None
        self._cache = {}

    # -- material plumbing ---------------------------------------------------
    def set_material(self, alpha, beta, gamma, dt_frame, device=None):
        """Cache everything that depends on the material but not on the state."""
        dev = device or next(self.parameters()).device
        c = al.cond_vec(alpha, beta, gamma, dt_frame, self.cond_mode).to(dev)
        amp = al.amp_ratio(alpha, beta, gamma).reshape(-1).to(dev)
        th = al.theta(alpha, dt_frame).reshape(-1).to(dev).float()
        mud = (al.mu(alpha) / al.DX).reshape(-1).to(dev).float()
        self._ctx = dict(c=c, amp=amp, theta=th, mu_over_dx=mud)
        self._cache = {}          # symbol cache; belongs to THIS material only
        return self._ctx

    def _ctx_for(self, x):
        assert self._ctx is not None, "call set_material() first"
        ctx = dict(self._ctx)
        B = x.shape[0]
        for k, v in ctx.items():
            if v.shape[0] == 1 and B > 1:
                ctx[k] = v.expand(B, *v.shape[1:])
        return ctx

    def init_state(self, x0):
        return None

    def step(self, x, state=None, steps=1.0):
        ctx = self._ctx_for(x)
        emb = self.trunk(ctx["c"])
        amp = ctx["amp"].view(-1, 1, 1, 1, 1)
        # ---- into the material's own field units ---------------------------
        v = x / amp if self.units else x
        # ---- linear half: exact semigroup in dimensionless time ------------
        if self.ssm is not None:
            v = self.ssm(v, ctx["theta"], emb, steps=steps,
                         mu_over_dx=ctx["mu_over_dx"], cache=self._cache)
        # ---- nonlinear half: shape/scale correction ------------------------
        if self.fno is not None:
            delta, g, s = self.fno(v, emb)
            # TIME RETARGET.  One frame of this material is theta/theta_ref
            # reference frames; the correction is a rate, so it scales with it.
            # Exactly 1.0 at the reference, hence the identity there.
            # `steps` multiplies it too: a strided window is a bigger step in
            # the SAME dimensionless time, so the two enter identically and a
            # model trained on strided pairs is being trained on the cadence
            # axis, not on a second, unrelated axis.
            st = _steps_t(steps, x.shape[0], x.device)
            tau = ((ctx["theta"] * st
                    / al.theta(al.ALPHA_REF, al.DT_FRAME_LEGACY).item()
                    ).view(-1, 1, 1, 1, 1) if self.tau_scale else 1.0)
            with torch.amp.autocast("cuda", enabled=False):
                vf = v.float()
                sf = s.float()
                v = sf * torch.exp(tau * g.float()) * (vf / sf + tau * delta.float())
        # ---- amplitude guard, in MULTIPLES OF SATURATION --------------------
        # ONE constant works for the whole family only because the state is
        # reduced: in reduced units every material saturates at the same value
        # while physical P_sat spans 4x across this design.  Without the
        # reduction this guard has to be retuned per material, which the alpha
        # campaign had to do.
        #
        # `guard` is in units of the material's own saturation, so the numeric
        # clamp is guard * P_s(ref)/P_NORM -- the reduced field is scaled so the
        # REFERENCE maps to itself (which keeps learning rates transferable
        # between arms), and the reference saturates at P_s(ref)/0.25 = 0.654,
        # not at 1.  Writing the clamp as a bare 1.35 would have meant 2.1x
        # saturation and guarded nothing.
        #
        # THE SAME LINE RUNS IN EVERY ARM, WITH OR WITHOUT UNITS, and that is
        # deliberate.  Applying it only to the units arms would make "no units"
        # secretly mean "no guard", so the ablation would be measuring two
        # changes at once and the units arms would win partly for the wrong
        # reason.  Applied to v unconditionally, one constant means "1.35x THIS
        # material's saturation" when the state is reduced and "1.35x the
        # REFERENCE material's saturation" when it is not -- which is exactly
        # the choice a practitioner without the units is forced into, and
        # exactly what the units buy, expressed in one line.
        if self.guard > 0:
            v = torch.clamp(v, -self._guard_abs, self._guard_abs)
        return (v * amp if self.units else v).to(x.dtype), None

    def rollout(self, x0, n, keep=None):
        """n frames from x0. keep=list of frame indices to return (1-based)."""
        keep = set(range(1, n + 1) if keep is None else keep)
        x, out = x0, {}
        for t in range(1, n + 1):
            x, _ = self.step(x)
            if t in keep:
                out[t] = x
        return out


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="pi")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(0)
    m = ABGOperator(cond_mode=a.cond).to(a.device)
    print("params: total %d | trunk %d | ssm %d | fno %d" % (
        count_params(m), count_params(m.trunk),
        count_params(m.ssm) if m.ssm else 0,
        count_params(m.fno) if m.fno else 0))
    x = torch.randn(2, 1, Z_FE, H, W, device=a.device) * 1e-3
    m.set_material([-2.0e9, -2.3e9], [3.0e10, -4.5e9], [3.5e11, 1.3e11],
                   [2e-12, 2e-12])
    y, _ = m.step(x)
    print("step", tuple(y.shape), "finite:", bool(torch.isfinite(y).all()))
    # semigroup: two single steps vs one double step of the LINEAR half alone
    m2 = ABGOperator(cond_mode=a.cond, use_fno=False, units=False).to(a.device)
    m2.set_material(-2.0e9, 3.0e10, 3.5e11, 2e-12)
    for p in m2.ssm.mlp.parameters():
        if p.dim() > 1:
            nn.init.normal_(p, std=0.05)
    one, _ = m2.step(x)
    one, _ = m2.step(one)
    two, _ = m2.step(x, steps=2.0)
    err = (one - two).norm() / two.norm()
    print("semigroup exp(2 theta L) vs exp(theta L)^2: rel err %.2e" % err)
