#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench/physmetrics.py -- physical-consistency metrics for every FerroX TDGL model of the benchmark (same checkpoints,
same test sets: 256 held-out D1 + 300 D2), computed on the fly during the 400-frame rollout.

Per trajectory:
  pattern     corr / rel. L2 at frames FR (and the mean rel. L2 curve over all 400 frames)
  amplitude   |<|P|>_pred / <|P|>_true - 1| at t=400 (saturation polarisation)
  timing      nucleation frame = first t with rms P(t) >= 0.5 rms P(400) (own threshold); |dt| in frames
  morphology  domain size L = 2 pi / <k>_S (in-plane structure factor, z-averaged) and domain-wall density (fraction of
              nearest-neighbour sign changes) at t = 100 and 400 -> relative errors; W1 distance of the P/P_s histogram
  electrostatics  potential phi and depolarisation field Ez from the predicted Pz with FerroX's Poisson problem
              (ferro_poisson.PoissonLayer, validated vs FerroX's stored Phi at corr 0.995) -> rel. L2 at t = 50, 400
  energy      Landau-Ginzburg-Devonshire free energy  F = <a/2 P^2 + b/4 P^4 + g/6 P^6 + G11/2 |grad P|^2 - 1/2 Ez Pz>
              per frame; error of F(t) relative to the true total drop F(0)-F(400), averaged over t; component errors
              at t = 400; fraction of frames where F increases (truth itself has some, see ferrox-emono-reviewer)
  spectrum    log10 ratio of the radially binned power spectrum pred/true at frames 1, 10, 50, 400 (model mean)
Out: bench/results_phys/<tag>.json and <tag>_examples.npz (6 frames x 4 trajectories, for figures).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import models as M
import tasks as TK
import run as RUN

DEV = TK.DEV
OUT = os.path.join(HERE, "results_phys"); os.makedirs(OUT, exist_ok=True)
FR = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400]
SPEC_T = [1, 10, 50, 400]
EX_T = [1, 5, 20, 50, 100, 400]
EX_IDX = {"D1_heldout": [0, 40, 80, 100, 120, 160, 200, 240], "D2_ood": [0, 50, 100, 150, 200, 250]}
N = 64
_kx = np.fft.fftfreq(N) * N
KB = torch.from_numpy(np.rint(np.sqrt(_kx[:, None] ** 2 + _kx[None, :] ** 2)).astype(np.int64))
NB = int(KB.max()) + 1


def build(a, task):
    """the same constructor logic as bench/run.py main()"""
    gen = a.model in M.GENERATIVE
    if a.model == "duhamel":
        model = M.DuhamelTDGL(task.scale)
    elif a.model.startswith("duhamel_uno"):
        lin = "bank" if a.model != "duhamel_uno_poly" else "poly"
        bank = os.environ.get("BENCH_BANK") or os.path.join(os.environ["SCRATCH"], "bench_cache", "alpha_bank_tdgl.npz")
        model = M.DuhamelUNO(task.nd, task.cin, task.dcu, task.guard, linear=lin, bank=bank if lin == "bank" else None,
                             persist_k0=a.model == "duhamel_uno_k0")
    elif a.model == "podlstm":
        model = M.PODLSTM(int(np.prod(task.Umm.shape[2:])), task.dcu, k=int(os.environ.get("POD_K", "256")))
    elif a.wrap:
        model = M.RecipeWrap(a.model, task.nd, task.cin, task.dcu, task.theta_ref, task.guard)
    else:
        model = M.build2(a.model, task.nd, task.cin, task.dcond, a.scale)
        if a.norm == "inst" and not gen:
            model = M.InstNorm(model)
    return model.to(DEV), gen


def spectrum(x):
    """x (B,Z,H,W) -> radially binned power (B, NB), z-averaged"""
    p = torch.fft.fft2(x).abs().pow(2).mean(1)                               # (B,H,W)
    out = torch.zeros(x.shape[0], NB, device=x.device, dtype=p.dtype)
    out.index_add_(1, KB.to(x.device).reshape(-1), p.reshape(x.shape[0], -1))
    return out


def domain_size(x):
    s = spectrum(x)[:, 1:]
    k = torch.arange(1, NB, device=x.device, dtype=s.dtype)
    k1 = (s * k).sum(1) / s.sum(1).clamp_min(1e-30)
    return N / k1                                                             # grid cells


def wall_density(x):
    s = torch.sign(x)
    return 0.5 * ((s != torch.roll(s, 1, -1)).float().mean((1, 2, 3)) + (s != torch.roll(s, 1, -2)).float().mean((1, 2, 3)))


def w1(a, b, n=20000):
    a = a.reshape(-1); b = b.reshape(-1)
    g = torch.Generator(device=a.device).manual_seed(0)
    ia = torch.randint(a.numel(), (n,), generator=g, device=a.device); ib = torch.randint(b.numel(), (n,), generator=g, device=a.device)
    return float((torch.sort(a[ia])[0] - torch.sort(b[ib])[0]).abs().mean())


class Physics:
    def __init__(self):
        import ferro_poisson as FP
        import abg_laws as al
        self.FP, self.al = FP, al
        self.pois = FP.PoissonLayer(learn_scale=False, normalise=False).to(DEV)

    def fields(self, P):
        """P (B,Z,H,W) physical [C/m^2] -> phi, Ez (B,Z,H,W) in the FE layers"""
        phi, ez = self.pois(P[:, None])
        return phi[:, 0], ez[:, 0]

    def energy(self, P, mat):
        """P (B,Z,H,W) [C/m^2], mat list of (alpha, beta, gamma, dt) -> dict of (B,) energy densities [J/m^3]"""
        a = torch.tensor([m[0] for m in mat], device=P.device).view(-1, 1, 1, 1)
        b = torch.tensor([m[1] for m in mat], device=P.device).view(-1, 1, 1, 1)
        g = torch.tensor([m[2] for m in mat], device=P.device).view(-1, 1, 1, 1)
        P2 = P * P
        lan = (a / 2 * P2 + b / 4 * P2 ** 2 + g / 6 * P2 ** 3).mean((1, 2, 3))
        dx = self.al.DX
        gx = (torch.roll(P, -1, -1) - P) / dx; gy = (torch.roll(P, -1, -2) - P) / dx; gz = (P[:, 1:] - P[:, :-1]) / dx
        grad = self.al.G11 / 2 * ((gx ** 2 + gy ** 2).mean((1, 2, 3)) + (gz ** 2).sum(1).mean((1, 2)) / P.shape[1])
        _, ez = self.fields(P)
        ele = (-0.5 * ez * P).mean((1, 2, 3))
        return dict(landau=lan, grad=grad, elec=ele, total=lan + grad + ele)


def rel(a, b):
    red = tuple(range(1, a.dim()))
    return ((a - b).pow(2).sum(red).sqrt() / b.pow(2).sum(red).sqrt().clamp_min(1e-30))


def corr(a, b):
    red = tuple(range(1, a.dim()))
    ac, bc = a - a.mean(red, keepdim=True), b - b.mean(red, keepdim=True)
    return (ac * bc).sum(red) / (ac.pow(2).sum(red).sqrt() * bc.pow(2).sum(red).sqrt() + 1e-30)


@torch.no_grad()
def batch_metrics(Ph, Gt, mat, Ps, phys):
    """Ph, Gt (B,401,Z,H,W) physical incl. frame 0 (Ph[:,0] = Gt[:,0]). Returns list of per-trajectory dicts + model-level
    accumulators (rel curve, spectra)."""
    B = Ph.shape[0]; rows = [dict() for _ in range(B)]
    rl_curve = rel(Ph[:, 1:].flatten(0, 1), Gt[:, 1:].flatten(0, 1)).view(B, -1)          # (B,400)
    for t in FR:
        r = rel(Ph[:, t], Gt[:, t]); c = corr(Ph[:, t], Gt[:, t])
        for j in range(B):
            rows[j]["rel_t%d" % t] = float(r[j]); rows[j]["corr_t%d" % t] = float(c[j])
    # amplitude / saturation
    ap, at = Ph[:, 400].abs().mean((1, 2, 3)), Gt[:, 400].abs().mean((1, 2, 3))
    # timing
    rp = Ph.pow(2).mean((2, 3, 4)).sqrt(); rt = Gt.pow(2).mean((2, 3, 4)).sqrt()          # (B,401)
    def tnuc(r):
        thr = 0.5 * r[:, -1:]
        above = (r >= thr).float(); idx = torch.argmax(above, 1)
        return idx.float()
    tp, tt = tnuc(rp), tnuc(rt)
    # morphology
    morph = {}
    for t in (100, 400):
        morph[t] = (domain_size(Ph[:, t]), domain_size(Gt[:, t]), wall_density(Ph[:, t]), wall_density(Gt[:, t]))
    # electrostatics
    fld = {}
    for t in (50, 400):
        pp, ep = phys.fields(Ph[:, t]); pt, et = phys.fields(Gt[:, t])
        fld[t] = (rel(pp, pt), rel(ep, et), corr(pp, pt))
    # energy over all frames (stride 1)
    Ep = {k: [] for k in ("landau", "grad", "elec", "total")}; Et = {k: [] for k in Ep}
    for t0 in range(0, 401, 50):
        sl = slice(t0, min(t0 + 50, 401)); nt = sl.stop - sl.start
        mrep = [m for m in mat for _ in range(nt)]
        ep = phys.energy(Ph[:, sl].flatten(0, 1), mrep); et = phys.energy(Gt[:, sl].flatten(0, 1), mrep)
        for k in Ep:
            Ep[k].append(ep[k].view(B, nt)); Et[k].append(et[k].view(B, nt))
    Ep = {k: torch.cat(v, 1) for k, v in Ep.items()}; Et = {k: torch.cat(v, 1) for k, v in Et.items()}   # (B,401)
    drop = (Et["total"][:, 0] - Et["total"][:, -1]).abs().clamp_min(1e-30)
    e_traj = (Ep["total"] - Et["total"]).abs().mean(1) / drop
    inc_p = (Ep["total"][:, 1:] > Ep["total"][:, :-1] + 1e-9 * Et["total"][:, :1].abs()).float().mean(1)
    inc_t = (Et["total"][:, 1:] > Et["total"][:, :-1] + 1e-9 * Et["total"][:, :1].abs()).float().mean(1)
    for j in range(B):
        r = rows[j]
        r.update(rel_final=float(rl_curve[j, -1]), corr_final=r["corr_t400"], rel_mean=float(rl_curve[j].mean()),
                 psat_err=float((ap[j] / at[j] - 1).abs()), psat_true_over_Ps=float(at[j] / Ps[j]),
                 tnuc_pred=float(tp[j]), tnuc_true=float(tt[j]), tnuc_err=float((tp[j] - tt[j]).abs()),
                 hist_w1=w1(Ph[j, 400] / Ps[j], Gt[j, 400] / Ps[j]),
                 energy_traj_err=float(e_traj[j]), energy_inc_frac_pred=float(inc_p[j]), energy_inc_frac_true=float(inc_t[j]))
        for t in (100, 400):
            Lp, Lt, wp, wt = morph[t]
            r["dsize_err_t%d" % t] = float((Lp[j] / Lt[j] - 1).abs()); r["wall_err_t%d" % t] = float((wp[j] / wt[j].clamp_min(1e-6) - 1).abs())
            r["dsize_true_t%d" % t] = float(Lt[j]); r["wall_true_t%d" % t] = float(wt[j])
        for t in (50, 400):
            r["phi_rel_t%d" % t] = float(fld[t][0][j]); r["ez_rel_t%d" % t] = float(fld[t][1][j]); r["phi_corr_t%d" % t] = float(fld[t][2][j])
        for k in ("landau", "grad", "elec", "total"):
            r["E_%s_err_t400" % k] = float(((Ep[k][j, -1] - Et[k][j, -1]) / Et["total"][j, -1].abs().clamp_min(1e-30)).abs())
    spec = {t: torch.log10(spectrum(Ph[:, t]).clamp_min(1e-30) / spectrum(Gt[:, t]).clamp_min(1e-30)) for t in SPEC_T}
    # error spectrum: share of the squared relative error carried by each |k| bin (sums to rel_L2^2 over bins)
    for t in SPEC_T:
        e = spectrum(Ph[:, t] - Gt[:, t]) / spectrum(Gt[:, t]).sum(1, keepdim=True).clamp_min(1e-30)
        spec["err%d" % t] = e
        spec["truepow%d" % t] = spectrum(Gt[:, t]) / spectrum(Gt[:, t]).sum(1, keepdim=True).clamp_min(1e-30)
    return rows, rl_curve, spec, (Ep["total"], Et["total"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", default="mse")
    ap.add_argument("--wrap", action="store_true")
    ap.add_argument("--scale", default="global")
    ap.add_argument("--norm", default="global")
    ap.add_argument("--n", type=int, default=0, help="limit trajectories per set (testing)")
    a = ap.parse_args()
    torch.manual_seed(1234)
    task = TK.make("tdgl")
    phys = Physics()
    if a.model.startswith("ours"):
        tag = "tdgl_%s" % a.model
        f_ours = RUN.ours_tdgl(task, a.model); step = None
    else:
        if a.model in ("duhamel", "duhamel_ou2", "podlstm") or a.model.startswith("duhamel_uno"):
            a.wrap = True
        gen = a.model in M.GENERATIVE
        if a.wrap:
            a.loss = "rel"
        cfg = "lr%g_%s%s" % (a.lr, ("scale-" + a.scale) if gen else ("recipe" if a.wrap else a.loss), "_inst" if (a.norm == "inst" and not gen and not a.wrap) else "")
        tag = "tdgl_%s_%s_s%d%s" % (a.model, cfg, a.seed, os.environ.get("BENCH_TAG", ""))
        model, gen = build(a, task)
        ck = torch.load(os.path.join(RUN.CK, tag + ".pt"), map_location=DEV, weights_only=False)
        sd = {k: v for k, v in ck["model"].items() if not k.endswith("_metadata")}
        try:
            model.load_state_dict(sd)
        except RuntimeError as e:                                       # pre-fix checkpoint without coordinate channels
            if "size mismatch" not in str(e):
                raise
            os.environ["BENCH_NOPOS"] = "1"; model, gen = build(a, task); model.load_state_dict(sd)
            res_nopos = True
            print("loaded without coordinate channels (pre-fix checkpoint)", flush=True)
        model.eval()
        if gen:
            ds = []; rng = np.random.default_rng(a.seed)
            # generative models store their residual scale as a buffer/attr; re-derive as in run.py if missing
            if getattr(model, "dscale", None) is None:
                for _ in range(32):
                    w, _ = task.sample(16, 1, rng); ds.append((w[:, 1] - w[:, 0]).float().pow(2).mean().item())
                model.set_dscale(float(np.sqrt(np.mean(ds))) + 1e-8)

        mem = {}

        def step(x, c, aux=None):
            if a.model == "podlstm":
                last = mem.get("y")
                st = mem.get("st") if (last is not None and last.shape == x.shape and torch.equal(last, x)) else None
                y, st = model.step(x, aux["cu"], aux["U"], aux["th"], st)
                y = y.float().clamp(-task.bound, task.bound); mem["st"] = st; mem["y"] = torch.nan_to_num(y).clamp(-50, 50)
                return y
            if a.model in ("duhamel",) or a.model.startswith("duhamel_uno"):
                return model(x, aux).float().clamp(-task.bound, task.bound)
            if a.wrap:
                return model(x, aux["cu"], aux["U"], aux["th"]).float().clamp(-task.bound, task.bound)
            return model(x, c).float().clamp(-task.bound, task.bound)
    print("physmetrics", tag, flush=True)
    sc = task.scale * phys.al.P_NORM                                   # normalised -> C/m^2
    res = dict(tag=tag, model=a.model, seed=a.seed, sets={})
    ex = {}
    t0 = time.time()
    for sp, S in task.eval_sets().items():
        trajs = S["trajs"][: a.n] if a.n else S["trajs"]
        rows, curves, specs = [], [], {t: [] for t in SPEC_T}; Es = []
        specs.update({"err%d" % t: [] for t in SPEC_T}); specs.update({"truepow%d" % t: [] for t in SPEC_T})
        for i in range(0, len(trajs), 16):
            b = trajs[i:i + 16]
            u, c = task.load_batch(b)
            u = u.to(DEV).float()
            if step is None:
                P = f_ours(b).float()
            else:
                A = RUN._aux_to(task.batch_aux); A["cu"] = A["cu"].unsqueeze(1).expand(-1, u.shape[1] - 1, -1)
                cs = c.to(DEV).float().unsqueeze(1).expand(-1, u.shape[1] - 1, -1)
                P = RUN.rollout(step, u[:, 0], cs, A)
            Ph = torch.cat([u[:, :1], P], 1) * sc; Gt = u * sc
            mat = [tuple(float(v) for v in t.mat) for t in b]; Ps = [float(t.P_s) for t in b]
            r, cv, spc, E = batch_metrics(Ph, Gt, mat, Ps, phys)
            for rr, t in zip(r, b):
                rr.update(tag=t.tag, subset=(t.box if getattr(t, "box", None) else t.split))
            rows += r; curves.append(cv.cpu()); Es.append(torch.stack(E, 1).cpu())
            for t in list(specs):
                specs[t].append(spc[t].cpu())
            for j in EX_IDX.get(sp, []):
                if i <= j < i + len(b):
                    ex["%s_%d_pred" % (sp, j)] = (Ph[j - i, EX_T] / phys.al.P_NORM).cpu().numpy().astype(np.float16)
                    ex["%s_%d_true" % (sp, j)] = (Gt[j - i, EX_T] / phys.al.P_NORM).cpu().numpy().astype(np.float16)
            print("  %s %d/%d  %.1f min" % (sp, i + len(b), len(trajs), (time.time() - t0) / 60), flush=True)
        keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
        out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        out.update({k + "_median": float(np.median([r[k] for r in rows])) for k in keys})
        out["n"] = len(rows)
        out["rel_curve_mean"] = torch.cat(curves).mean(0).numpy().round(5).tolist()
        out["rel_curve_median"] = torch.cat(curves).median(0).values.numpy().round(5).tolist()
        for t in SPEC_T:
            s = torch.cat(specs[t])
            out["spec_logratio_t%d" % t] = s.mean(0).numpy().round(4).tolist()
            out["spec_abslogratio_t%d" % t] = s.abs().mean(0).numpy().round(4).tolist()
            out["spec_err_t%d" % t] = torch.cat(specs["err%d" % t]).mean(0).numpy().tolist()
            out["spec_truepow_t%d" % t] = torch.cat(specs["truepow%d" % t]).mean(0).numpy().tolist()
        Ecat = torch.cat(Es)                                            # (n, 2, 401)
        out["energy_pred_mean"] = Ecat[:, 0].mean(0).numpy().tolist(); out["energy_true_mean"] = Ecat[:, 1].mean(0).numpy().tolist()
        groups = {}
        for r in rows:
            groups.setdefault(r["subset"], []).append(r)
        out["subsets"] = {g: dict({k: float(np.mean([r[k] for r in rs])) for k in keys}, n=len(rs)) for g, rs in groups.items()}
        out["rows"] = rows
        res["sets"][sp] = out
        print("%s corr400 %.4f rel400 %.4f psat %.4f tnuc %.2f dsize %.4f wall %.4f phi %.4f E %.4f" % (
            sp, out["corr_t400"], out["rel_t400"], out["psat_err"], out["tnuc_err"], out["dsize_err_t400"], out["wall_err_t400"],
            out["phi_rel_t400"], out["energy_traj_err"]), flush=True)
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(os.path.join(OUT, tag + ".json"), "w"))
    np.savez_compressed(os.path.join(OUT, tag + "_examples.npz"), t=np.array(EX_T), **ex)
    print("DONE", tag, flush=True)


if __name__ == "__main__":
    main()
