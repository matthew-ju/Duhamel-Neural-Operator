#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""geo_model.py -- one conditioned FNO for BOTH device geometries (D1 5 nm film, D3 40 nm film).

Why a new conditioning. abg_model computes its "units" from abg_laws, which is calibrated to
the D1 stack only (critical alpha 1.367e9, dx = 0.5 nm, d_fe = 5 nm, eps_de = 10). D3 (40 nm
film, dx = 1 nm, eps_de = 1e4) has critical alpha ~3.5e7, so the same formulas are simply wrong
there. Here the conditioning vector is built from per-trajectory DIMENSIONLESS numbers that
mean the same thing in any geometry, all relative to the D1 reference material (so the
reference maps to exactly zero, as before):

  c0  log(theta / theta_ref)      theta = frame interval / linear growth time
  c1  log(P_s / P_s,ref)          saturation polarisation (also the amplitude unit)
  c2  r - r_ref                   6th-order share of the restoring force (well shape)
  c3  log(Pi_h / Pi_h,ref)        Pi_h = cell size / wall width   (grid resolution)
  c4  log(Pi_z / Pi_z,ref)        Pi_z = film thickness / wall width   (NEW: geometry)
  c5  log10(eps_de / 10)          dielectric screening   (NEW: electrostatics)

For every D1 trajectory c0..c3 are EXACTLY abg_laws.cond_vec(mode="pi"), and c4 = c3 (D1 has a
fixed 5 nm film, so Pi_z = 10 Pi_h) and c5 = 0. The trunk's first four input columns are copied
from a trained D1 model and the two new columns start at zero, so at initialisation this model
IS the trained D1 model on D1 data -- anything it learns about D3 is added, not relearned.

The FNO half is already geometry-agnostic in depth: it sees a normalised 0..1 z-coordinate and
truncated Fourier modes, so 10 (D1) or 40 (D3) ferroelectric layers go through the same weights.
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import abg_laws as al
from abg_model import ABGOperator, CondTrunk

N_COND = 6
THETA_REF = float(al.theta(al.ALPHA_REF, al.DT_FRAME_LEGACY))
PS_REF = float(al.p_sat(al.ALPHA_REF, al.BETA_REF, al.GAMMA_REF))
R_REF = float(al.r_share(al.ALPHA_REF, al.BETA_REF, al.GAMMA_REF))
PIH_REF = float(al.pi_h(al.ALPHA_REF))
PIZ_REF = 10.0 * PIH_REF                    # d_fe / dx = 5 nm / 0.5 nm = 10 in D1


def ctx_d1(alpha, beta, gamma, dt_frame):
    """(c[6], amp, theta) for a D1-geometry material, identical to abg_laws in c0..c3."""
    c4 = al.cond_vec(alpha, beta, gamma, dt_frame, "pi")[0].double()
    c = torch.cat([c4, c4[3:4], torch.zeros(1, dtype=torch.float64)])
    return (c.float(), float(al.amp_ratio(alpha, beta, gamma)), float(al.theta(alpha, dt_frame)))


def ctx_geom(theta, P_s, r, dx_over_mu, dfe_over_mu, eps_de):
    """(c[6], amp, theta) from the dimensionless numbers of ANY geometry (used for D3)."""
    c = torch.tensor([math.log(theta / THETA_REF), math.log(P_s / PS_REF), r - R_REF,
                      math.log(dx_over_mu / PIH_REF), math.log(dfe_over_mu / PIZ_REF),
                      math.log10(eps_de / 10.0)], dtype=torch.float32)
    return c, P_s / PS_REF, float(theta)


class GeoOperator(ABGOperator):
    """ABGOperator's FNO path (ampnorm, tau-retarget, guard) with an externally supplied context."""

    def __init__(self, width=48, n_layers=4, modes_xy=16, guard=1.35, modes_z=4):
        super().__init__(cond_mode="pi", use_ssm=False, use_fno=True, width=width,
                         n_layers=n_layers, modes_xy=modes_xy, modes_z=modes_z, guard=guard)
        self.trunk = CondTrunk(N_COND, d_out=16)

    def set_context(self, c, amp, theta, device=None):
        dev = device or next(self.parameters()).device
        c = torch.as_tensor(c, dtype=torch.float32).reshape(-1, N_COND).to(dev)
        B = c.shape[0]
        self._ctx = dict(c=c,
                         amp=torch.as_tensor(amp, dtype=torch.float32).reshape(-1).to(dev),
                         theta=torch.as_tensor(theta, dtype=torch.float32).reshape(-1).to(dev),
                         mu_over_dx=torch.ones(B, device=dev))
        self._cache = {}


def from_d1_checkpoint(path, template=os.path.join(HERE, "checkpoints", "abg_no_ssm_vol.pt")):
    """GeoOperator initialised from a trained D1 no_ssm model (q3 state dict or abg checkpoint):
    FNO copied, trunk's first 4 input columns copied, geometry columns zero."""
    m = GeoOperator()
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in st["model"].items() if k != "_metadata"}
    own = m.state_dict()
    for k, v in sd.items():
        if k not in own:
            continue
        if k == "trunk.net.0.weight":          # (hidden, 4) -> (hidden, 6)
            w = torch.zeros_like(own[k]); w[:, :v.shape[1]] = v; own[k] = w
        elif own[k].shape == v.shape:
            own[k] = v
    m.load_state_dict({k: v for k, v in own.items() if k != "_metadata"})
    return m


def widen_modes_z(src_state, modes_z):
    """GeoOperator with MORE depth Fourier modes, warm-started exactly from a 4-mode state dict.
    neuralop stores spectral weights fft-shifted (centred) along depth, so the old block goes in the
    middle and the new outer modes start at zero: the widened model computes exactly the same
    function at initialisation (D3's 40 layers need finer depth structure than D1's 10)."""
    m = GeoOperator(modes_z=modes_z)
    own = m.state_dict()
    for k, v in src_state.items():
        if k == "_metadata" or k not in own:
            continue
        if v.is_complex() and own[k].shape != v.shape:
            w = torch.zeros_like(own[k]); mz_old, mz_new = v.shape[2], own[k].shape[2]
            lo = (mz_new - mz_old) // 2
            w[:, :, lo:lo + mz_old] = v; own[k] = w
        elif own[k].shape == v.shape:
            own[k] = v
    m.load_state_dict({k: v for k, v in own.items() if k != "_metadata"})
    return m
