#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task adapters for the comparative benchmark. A task provides
  .nd, .cin, .dcond, .scale             spatial dims, channels, conditioning width, global field scale
  .sample(bs, k, rng) -> win (B,k+1,C,*S), cond (B,k,D)      normalised training windows (on GPU)
  .eval_sets() -> {split: [item, ...]}  item = dict(u (T,C,*S) normalised tensor, cond (T-1,D), meta)
  .extra_metrics(split, pred, item)     task-specific metrics (hysteresis loop metrics)
Conditioning for every baseline = the RAW physical parameters (normalised by the training mean/std), broadcast
as channels -- the standard parametric-PDE setup. Fields are divided by one global training std.
"""
import json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # repository root
for p in (HERE, os.path.join(HERE, "ml_units"), os.path.join(HERE, "geo"), os.path.join(HERE, "q_probe"),
          os.path.join(HERE, "handoff"), os.path.join(HERE, "q_probe", "volt_sweep")):
    if p not in sys.path:
        sys.path.insert(0, p)
DEV = torch.device("cuda")


class MLUnitsTask:
    """ac / ch / nsd / nsf / rd / burgers from ml_units (train 1000-2200, test splits id / ood_units / ood_inv)."""
    SPLITS = ("id", "ood_units", "ood_inv")

    def __init__(self, system):
        import train as T
        self.T, self.system = T, system
        tr = T.load_split(system, "train")
        nv = max(16, len(tr["u"]) // 10)                          # validation = last 10% of training trajectories
        self.val_raw = {k: v[-nv:] for k, v in tr.items()}
        tr = {k: v[:-nv] for k, v in tr.items()}
        self.nd = tr["u"].ndim - 2; self.cin = 1
        self.cm, self.cs = tr["raw"].mean(0), tr["raw"].std(0) + 1e-6
        self.scale = float(tr["u"].std())
        self.U = torch.from_numpy(tr["u"] / self.scale).to(DEV)
        self.bound = 3.0 * float(self.U.abs().max())
        self.C = torch.from_numpy((tr["raw"] - self.cm) / self.cs).float().to(DEV)
        self.dcond = self.C.shape[1]
        N, Tn = self.U.shape[:2]
        self.Tn = Tn
        # units info for the recipe wrapper (bench/models.RecipeWrap)
        self.cref = np.median(tr["dml"], 0).astype(np.float32); self.theta_ref = float(np.median(tr["theta"]))
        self.AUX = dict(U=torch.from_numpy(tr["U"] / self.scale).float().to(DEV), th=torch.from_numpy(tr["theta"]).float().to(DEV),
                        cu=torch.from_numpy(tr["dml"] - self.cref).float().to(DEV))
        self.dcu = self.AUX["cu"].shape[1]
        self.guard = {"ac": 1.35, "ch": 1.35, "rd": 1.35}.get(system.rstrip("0123456789"), 0.0) / 1.0

    def sample(self, bs, k, rng):
        N, Tn = self.U.shape[:2]
        i = rng.integers(N, size=bs)
        # growth phase (first 40%) oversampled x3, as in the recipe
        t = np.where(rng.random(bs) < 0.6, rng.integers(0, int(0.4 * (Tn - k)) + 1, size=bs), rng.integers(0, Tn - k, size=bs))
        win = torch.stack([self.U[a, b:b + k + 1] for a, b in zip(i, t)]).unsqueeze(2)
        ii = torch.from_numpy(i).to(DEV)
        cond = self.C[ii].unsqueeze(1).expand(-1, k, -1)
        self.last_aux = dict(U=self.AUX["U"][ii], th=self.AUX["th"][ii], cu=self.AUX["cu"][ii].unsqueeze(1).expand(-1, k, -1))
        return win, cond

    def val_set(self):
        S = self.val_raw
        return {"val": dict(u=torch.from_numpy(S["u"] / self.scale).unsqueeze(2), cond=torch.from_numpy((S["raw"] - self.cm) / self.cs).float(),
                            batched=True, aux=dict(U=torch.from_numpy(S["U"] / self.scale).float(), th=torch.from_numpy(S["theta"]).float(),
                                                   cu=torch.from_numpy(S["dml"] - self.cref).float()))}

    def eval_sets(self):
        out = {}
        for sp in self.SPLITS:
            if not os.path.exists(os.path.join(self.T.D, "%s_%s.npz" % (self.system, sp))):
                continue
            S = self.T.load_split(self.system, sp)
            u = torch.from_numpy(S["u"] / self.scale).unsqueeze(2)
            c = torch.from_numpy((S["raw"] - self.cm) / self.cs).float()
            out[sp] = dict(u=u, cond=c, batched=True, aux=dict(U=torch.from_numpy(S["U"] / self.scale).float(),
                           th=torch.from_numpy(S["theta"]).float(), cu=torch.from_numpy(S["dml"] - self.cref).float()))
        return out


SYM = {"ac": "d4sign", "ch": "d4sign", "tdgl": "d4sign", "hyst": "d4signV", "nsd": "rot", "nsf": "none", "rd": "flip", "burgers": "flipneg"}


def augment(win, cond, aux, kind, nd):
    """exact symmetries of each system, one random element per batch: win (B,K,C,*S); hyst sign flip also flips V."""
    if kind == "none":
        return win, cond, aux
    if nd == 2:
        k = int(torch.randint(4, (1,)))
        if k and kind != "none":
            win = torch.rot90(win, k, dims=(-2, -1))
        if kind in ("d4sign", "d4signV") and torch.rand(1) < 0.5:
            win = torch.flip(win, dims=(-1,))
    else:
        if kind in ("flip", "flipneg") and torch.rand(1) < 0.5:
            win = torch.flip(win, dims=(-1,))
            if kind == "flipneg":
                win = -win
    if kind in ("d4sign", "d4signV") and torch.rand(1) < 0.5:
        win = -win
        if kind == "d4signV":                          # P -> -P together with V -> -V
            cond = cond.clone(); cond[..., 3:6] = -cond[..., 3:6]
            if aux is not None:
                aux = dict(aux); cu = aux["cu"].clone(); cu[..., 4:7] = -cu[..., 4:7]; aux["cu"] = cu
    return win, cond, aux


class TDGLTask:
    """FerroX Dataset 1, multi-material (alpha, beta, gamma), unique-seed trajectories.
    train = 745 D1 train configs x unique seed (2000+i); tests = 256 held-out D1 configs (new materials, new noise)
    and 300 Dataset 2 configs (out of range, zero-shot, new noise). z layers (10) as channels."""

    def __init__(self, max_train=None):
        import geo_data as gd
        self.gd = gd
        trs = [t for t in gd.index_d1(("useed",)) if t.split == "train"]
        if max_train:
            trs = trs[:max_train]
        self.train = trs
        self.raw = lambda t: np.array([np.log(abs(t.mat[0])), t.mat[1] / 1e10, t.mat[2] / 1e11, np.log(t.mat[3])], np.float32)
        R = np.stack([self.raw(t) for t in trs])
        self.cm, self.cs = R.mean(0), R.std(0) + 1e-6
        print("TDGL: loading %d training trajectories" % len(trs), flush=True)
        cache = os.path.join(os.environ["SCRATCH"], "bench_cache", "tdgl_train_fp16.npy")
        self.Umm = np.load(cache, mmap_mode="r")                    # (745,401,10,64,64) fp16, shared page cache
        assert self.Umm.shape[0] == len(trs)
        smp = np.stack([np.asarray(self.Umm[i, 200:401:20], np.float32) for i in range(0, len(trs), 12)])
        self.scale = float(smp.std())
        self.bound = 3.0 * float(np.abs(smp).max()) / self.scale    # generic output bound: 3x the training max
        print("TDGL: memmap", self.Umm.shape, "scale %.4f bound %.2f" % (self.scale, self.bound), flush=True)
        self.C = torch.from_numpy((R - self.cm) / self.cs).to(DEV)
        self.nd, self.cin, self.dcond = 2, self.Umm.shape[2], self.C.shape[1]
        self.val_idx = np.arange(0, len(trs), 12); self.tr_idx = np.setdiff1d(np.arange(len(trs)), self.val_idx)
        nsub = int(os.environ.get("TDGL_SUBSET", "0"))                  # data-efficiency runs: K random training materials
        if nsub:
            self.tr_idx = np.sort(np.random.default_rng(0).choice(self.tr_idx, nsub, replace=False))
            print("TDGL: training subset of %d materials" % nsub, flush=True)
        import abg_laws as al
        self.theta_ref = float(al.theta(al.ALPHA_REF, al.DT_FRAME_LEGACY)); self.guard = 1.35 * float(al.p_sat(al.ALPHA_REF, al.BETA_REF, al.GAMMA_REF)) / al.P_NORM
        self.AUX = dict(U=torch.tensor([float(t.amp) / self.scale for t in trs]).to(DEV), th=torch.tensor([float(t.theta) for t in trs]).to(DEV),
                        cu=torch.stack([torch.as_tensor(t.c).float().reshape(-1) for t in trs]).to(DEV))
        self.dcu = self.AUX["cu"].shape[1]
        self.nounits = os.environ.get("BENCH_NOUNITS") == "1"        # ABLATION: no change of units -- one amplitude unit and one time unit for every material
        if self.nounits:
            self.U0 = float(self.AUX["U"].median()); self.TH0 = float(self.AUX["th"].median())
            self.AUX["U"] = torch.full_like(self.AUX["U"], self.U0); self.AUX["th"] = torch.full_like(self.AUX["th"], self.TH0)
            print("TDGL: NO-UNITS ablation, U0 %.4f theta0 %.4f" % (self.U0, self.TH0), flush=True)

    def sample(self, bs, k, rng):
        N, Tn = self.Umm.shape[:2]
        i = self.tr_idx[rng.integers(len(self.tr_idx), size=bs)]
        t = np.where(rng.random(bs) < 0.6, rng.integers(0, 121, size=bs), rng.integers(0, Tn - k, size=bs))
        t = np.minimum(t, Tn - k - 1)
        win = np.stack([np.asarray(self.Umm[a, b:b + k + 1], np.float32) for a, b in zip(i, t)]) / self.scale
        ii = torch.from_numpy(i).to(DEV)
        cond = self.C[ii].unsqueeze(1).expand(-1, k, -1)
        self.last_aux = dict(U=self.AUX["U"][ii], th=self.AUX["th"][ii], cu=self.AUX["cu"][ii].unsqueeze(1).expand(-1, k, -1),
                             mat=[tuple(float(v) for v in self.train[j].mat) for j in i])
        return torch.from_numpy(win).to(DEV), cond

    def val_set(self):
        return {"val": dict(trajs=[self.train[i] for i in self.val_idx], batched=True, lazy=True)}

    def eval_sets(self, n_d1=256, n_d2=300):
        gd = self.gd
        d1 = [t for t in gd.index_d1(("useed",)) if t.split in ("interp", "extrap")][:n_d1]
        d2 = gd.index_d2u()[:n_d2]
        return {"D1_heldout": dict(trajs=d1, batched=True, lazy=True), "D2_ood": dict(trajs=d2, batched=True, lazy=True)}

    def load_batch(self, trajs):
        u = torch.stack([torch.from_numpy(self.gd.frames(t, 0, 401)[:, 0]) for t in trajs]) / self.scale
        c = torch.from_numpy((np.stack([self.raw(t) for t in trajs]) - self.cm) / self.cs)
        self.batch_aux = dict(U=torch.tensor([float(t.amp) / self.scale for t in trajs]), th=torch.tensor([float(t.theta) for t in trajs]),
                              cu=torch.stack([torch.as_tensor(t.c).float().reshape(-1) for t in trajs]),
                              mat=[tuple(float(v) for v in t.mat) for t in trajs])
        if self.nounits:
            n = len(trajs); self.batch_aux["U"] = torch.full((n,), self.U0); self.batch_aux["th"] = torch.full((n,), self.TH0)
        return u, c


class HystTask:
    """FerroX voltage-sweep hysteresis loops (eps_de = 100). train = 100 loops (50 materials x 2 V_max), tests =
    new noise (100), held-out materials (30), closed loops (10, extra leg). cond per step = material + V(t), V(t+1),
    dV, log eps (same extras as train_hyst.extra_vec)."""

    def __init__(self):
        import glob
        os.environ.setdefault("HYST_DIR", "ferrox_hyst2")
        import train_hyst as th
        self.th = th
        paths = sorted(glob.glob(os.path.join(os.environ["SCRATCH"], os.environ["HYST_DIR"], "h_*.npz")))
        sp = json.load(open(th.SPLIT2))
        groups = {}
        for p in paths:                                    # group by name only; test loops are loaded lazily
            info = th.parse(os.path.basename(p)[:-4])
            groups.setdefault((info["eps"], info["tag"], info["vmax"]), []).append((info["seed"], p))
        tr_p, self.t_noise, self.t_mat = [], [], []
        for (eps, tag, vm), Ls in groups.items():
            Ls = [p for _, p in sorted(Ls)]
            if tag in sp["heldout"]:
                self.t_mat += Ls
            else:
                tr_p.append(Ls[0]); self.t_noise += Ls[1:]
        self.t_closed = sorted(glob.glob(os.path.join(os.environ["SCRATCH"], "ferrox_hyst_closed", "as_h", "h_*.npz")))
        tags = sorted({th.parse(os.path.basename(p)[:-4])["tag"] for p in tr_p})
        vtags = set(tags[::10])
        self.t_val = [p for p in tr_p if th.parse(os.path.basename(p)[:-4])["tag"] in vtags]
        tr_p = [p for p in tr_p if p not in self.t_val]
        self.train = []
        for p in tr_p:
            L = th.load_loop(p); L["P"] = L["P"].astype(np.float16); self.train.append(L)
        R = np.stack([self._mraw(L) for L in self.train])
        self.cm, self.cs = R.mean(0), R.std(0) + 1e-6
        self.scale = float(np.mean([L["P"][len(L["P"]) // 2:].astype(np.float32).std() for L in self.train]))
        self.P = [(torch.from_numpy(L["P"][:, 0]).to(DEV).float() / self.scale).half() for L in self.train]
        for L in self.train:
            L["P"] = None
        self.Cs = [self.cond_seq(L).to(DEV) for L in self.train]
        self.bound = 3.0 * max(float(p.float().abs().max()) for p in self.P)
        import abg_laws as al
        self.al = al
        self.theta_ref = float(al.theta(al.ALPHA_REF, al.DT_FRAME_LEGACY)); self.guard = 2.5 * float(al.p_sat(al.ALPHA_REF, al.BETA_REF, al.GAMMA_REF)) / al.P_NORM
        self.AUXL = [self.loop_aux(L) for L in self.train]
        self.dcu = self.AUXL[0]["cu"].shape[1]
        self.items = [(i, t) for i, L in enumerate(self.train) for t in range(len(L["V"]) - 13)]
        self.nd, self.cin = 2, self.P[0].shape[1]
        self.dcond = self.Cs[0].shape[1]
        print("hyst: train %d loops (%d windows), noise %d, material %d, closed %d" % (
            len(self.train), len(self.items), len(self.t_noise), len(self.t_mat), len(self.t_closed)), flush=True)

    def _mraw(self, L):
        a, b, g, dtf = L["mat"]
        return np.array([np.log(abs(a)), b / 1e10, g / 1e11], np.float32)

    def loop_aux(self, L):
        a, b, g, dtf = L["mat"]; al = self.al
        cm = al.cond_vec([a], [b], [g], [dtf], "pi").float().reshape(1, -1)
        V = torch.from_numpy(L["V"]); eps = torch.full((len(V) - 1,), float(L["eps"]))
        ex = self.th.extra_vec(V[:-1], V[1:], eps).float()
        return dict(U=torch.tensor(float(al.amp_ratio(a, b, g)) / self.scale), th=torch.tensor(float(al.theta(a, dtf))),
                    cu=torch.cat([cm.expand(len(V) - 1, -1), ex], 1).to(DEV))

    def cond_seq(self, L):
        m = torch.from_numpy((self._mraw(L) - self.cm) / self.cs).float()
        V = torch.from_numpy(L["V"]); eps = torch.full((len(V) - 1,), float(L["eps"]))
        ex = self.th.extra_vec(V[:-1], V[1:], eps).float()
        return torch.cat([m.expand(len(V) - 1, -1), ex], 1)

    def sample(self, bs, k, rng):
        sel = rng.integers(len(self.items), size=bs)
        win, cond, U, th, cu = [], [], [], [], []
        for j in sel:
            i, t = self.items[j]
            t = min(t, len(self.train[i]["V"]) - k - 1)
            win.append(self.P[i][t:t + k + 1]); cond.append(self.Cs[i][t:t + k])
            A = self.AUXL[i]; U.append(A["U"]); th.append(A["th"]); cu.append(A["cu"][t:t + k])
        self.last_aux = dict(U=torch.stack(U).to(DEV), th=torch.stack(th).to(DEV), cu=torch.stack(cu),
                             mat=[tuple(float(v) for v in self.train[self.items[j][0]]["mat"]) for j in sel])
        return torch.stack(win).float(), torch.stack(cond)

    def val_set(self):
        return {"val": dict(loops=self.t_val, batched=False)}

    def eval_sets(self):
        return {"test_material": dict(loops=self.t_mat, batched=False), "test_noise": dict(loops=self.t_noise[::3][:34], batched=False),
                "closed": dict(loops=self.t_closed, batched=False)}

    def load(self, path):
        L = self.th.load_loop(path); L["name"] = os.path.basename(path)[:-4]; return L

    def loop_metrics(self, pred_norm, L):
        """pred_norm (T, C, H, W) in normalised units -> train_hyst.loop_metrics."""
        pred = (pred_norm * self.scale).unsqueeze(1).cpu().numpy()        # (T,1,Z,H,W)
        r, mp, mt = self.th.loop_metrics(pred, L)
        return dict(loop_err=r["loop_err"], field_corr=float(np.mean(r["field_corr"])), relL2_mean=r["relL2_mean"],
                    coercive_exact=float(r["coercive_pred"] == r["coercive_true"]))


def make(name):
    if name == "tdgl":
        return TDGLTask(max_train=int(os.environ.get("TDGL_MAX", "0")) or None)
    if name == "hyst":
        return HystTask()
    return MLUnitsTask(name)
