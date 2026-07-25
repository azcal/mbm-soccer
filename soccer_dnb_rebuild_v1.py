"""
MBM Big-5 soccer DNB rebuild v1 -- standalone, no market inputs.
Target: home win conditional on no draw (draws ~25% grade as pushes).
Data: football-data.co.uk results + shots, seasons 2019-20 .. 2025-26.
Protocol: tune on 2024-25 only; early-season shrink fitted on pre-2025-26 walk-forward
out-of-sample predictions; 2025-26 is the frozen holdout, touched once.
"""
import numpy as np, pandas as pd, json
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526"]
LEAGUES = ["EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1"]

class Decay:
    def __init__(self, hl): self.d = 0.5 ** (1.0/hl); self.sw = 0.0; self.sx = 0.0
    def add(self, x): self.sw = self.sw*self.d + 1; self.sx = self.sx*self.d + x
    def brk(self, keep): self.sw *= keep; self.sx *= keep
    def shrunk(self, tau=4.0): return self.sx/(self.sw + tau)

def build(df, cfg):
    K, HA, CARRY, PROM = cfg["K"], cfg["HA"], cfg["carry"], cfg["prom_init"]
    HLF, HLS = cfg["hl_fast"], cfg["hl_slow"]
    elo = {}; st = {}; seen = defaultdict(set); season_of = {}; prev_date = {}; gp = defaultdict(int)
    prom_flag = defaultdict(int)
    lg_g = Decay(800); lg_sot = Decay(800); lg_sh = Decay(800)
    feats = []
    for r in df.itertuples():
        sidx = SEASONS.index(r.season)
        for tm in (r.HomeTeam, r.AwayTeam):
            key = (r.league, tm)
            if season_of.get(key) != r.season:
                promoted = sidx > 0 and tm not in seen[(r.league, SEASONS[sidx-1])]
                prom_flag[key] = 1 if (promoted or key not in elo) else 0
                if key not in elo or promoted:
                    elo[key] = 1500 + PROM
                    st[key] = {k: Decay(HLS) for k in ("gs", "ga", "sots", "sota", "shs", "sha")}
                    st[key]["gf_fast"] = Decay(HLF); st[key]["ga_fast"] = Decay(HLF)
                else:
                    elo[key] = 1500 + CARRY*(elo[key]-1500)
                    for d in st[key].values(): d.brk(CARRY)
                gp[key] = 0; prev_date.pop(key, None)
                season_of[key] = r.season
            seen[(r.league, r.season)].add(tm)
        hk, ak = (r.league, r.HomeTeam), (r.league, r.AwayTeam)
        row = {"date": r.date, "season": r.season, "league": r.league,
               "home": r.HomeTeam, "away": r.AwayTeam, "hg": r.FTHG, "ag": r.FTAG}
        for side, k in (("h", hk), ("a", ak)):
            pv = prev_date.get(k)
            rest = min((r.date - pv).days, 14) if pv is not None else 7
            row.update({f"{side}_elo": elo[k], f"{side}_rest": rest, f"{side}_gp": min(gp[k], 10),
                        f"{side}_prom": 1 if (prom_flag[k] and gp[k] < 10) else 0,
                        f"{side}_gnet_f": st[k]["gf_fast"].shrunk() - st[k]["ga_fast"].shrunk(),
                        f"{side}_gnet_s": st[k]["gs"].shrunk() - st[k]["ga"].shrunk(),
                        f"{side}_sotnet": st[k]["sots"].shrunk() - st[k]["sota"].shrunk(),
                        f"{side}_shnet": st[k]["shs"].shrunk() - st[k]["sha"].shrunk()})
        feats.append(row)
        # ---- post-match updates ----
        lgm_g, lgm_sot, lgm_sh = lg_g.sx/max(lg_g.sw,1), lg_sot.sx/max(lg_sot.sw,1), lg_sh.sx/max(lg_sh.sw,1)
        for k, gf, ga, sf, sa, stf, sta, ok in ((hk, r.FTHG, r.FTAG, r.HS, r.AS, r.HST, r.AST, ak),
                                                 (ak, r.FTAG, r.FTHG, r.AS, r.HS, r.AST, r.HST, hk)):
            opp = st[ok]
            adj_gf = gf - (opp["ga"].shrunk())            # centered-ish: opponent leak strength
            adj_ga = ga - (opp["gs"].shrunk())
            st[k]["gf_fast"].add(adj_gf); st[k]["ga_fast"].add(adj_ga)
            st[k]["gs"].add(adj_gf); st[k]["ga"].add(adj_ga)
            if sf == sf: st[k]["shs"].add(sf - lgm_sh)
            if sa == sa: st[k]["sha"].add(sa - lgm_sh)
            if stf == stf: st[k]["sots"].add(stf - lgm_sot)
            if sta == sta: st[k]["sota"].add(sta - lgm_sot)
            gp[k] += 1; prev_date[k] = r.date
        lg_g.add(r.FTHG); lg_g.add(r.FTAG)
        if r.HS == r.HS: lg_sh.add(r.HS); lg_sh.add(r.AS)
        if r.HST == r.HST: lg_sot.add(r.HST); lg_sot.add(r.AST)
        res = 1.0 if r.FTHG > r.FTAG else 0.0 if r.FTHG < r.FTAG else 0.5
        e = 1/(1+10**(-((elo[hk]+HA)-elo[ak])/400))
        elo[hk] += K*(res-e); elo[ak] -= K*(res-e)
    return pd.DataFrame(feats)

FEATS = ["elo_d", "gnetf_d", "gnets_d", "sot_d", "sh_d", "rest_d", "h_gp", "a_gp",
         "h_prom", "a_prom", "cong_h", "cong_a", "lg_LaLiga", "lg_SerieA", "lg_Bundesliga", "lg_Ligue1"]

def add_diffs(f):
    f = f.copy()
    f["elo_d"] = f.h_elo - f.a_elo
    f["gnetf_d"] = f.h_gnet_f - f.a_gnet_f; f["gnets_d"] = f.h_gnet_s - f.a_gnet_s
    f["sot_d"] = f.h_sotnet - f.a_sotnet; f["sh_d"] = f.h_shnet - f.a_shnet
    f["rest_d"] = f.h_rest - f.a_rest
    f["cong_h"] = (f.h_rest <= 3).astype(int); f["cong_a"] = (f.a_rest <= 3).astype(int)
    for lg in ("LaLiga", "SerieA", "Bundesliga", "Ligue1"):
        f[f"lg_{lg}"] = (f.league == lg).astype(int)
    f["draw"] = (f.hg == f.ag); f["y"] = (f.hg > f.ag).astype(int)
    f["min_gp"] = f[["h_gp", "a_gp"]].min(axis=1)
    return f

def fit_eval(f, tr_seasons, te_season, C):
    tr = f.season.isin(tr_seasons) & (~f.draw); te = (f.season == te_season) & (~f.draw)
    sc = StandardScaler().fit(f.loc[tr, FEATS].values)
    m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(f.loc[tr, FEATS].values), f.loc[tr, "y"].values)
    p = m.predict_proba(sc.transform(f.loc[te, FEATS].values))[:, 1]
    return p, f.loc[te, "y"].values, f.loc[te], (m, sc)

def shrink(p, min_gp, a0):
    alpha = a0 + (1-a0)*np.minimum(min_gp, 6)/6.0
    lo = np.log(np.clip(p, 1e-6, 1-1e-6)/(1-np.clip(p, 1e-6, 1-1e-6)))
    return 1/(1+np.exp(-alpha*lo))

def M(y, p, lbl):
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6)
    return dict(model=lbl, n=len(y), log_loss=round(float(log_loss(y, p)), 4),
                brier=round(float(brier_score_loss(y, p)), 4), hit=round(float(np.mean((p > .5) == (y == 1))), 4))

def run():
    df = pd.read_csv("soccer/matches.csv", parse_dates=["date"])
    df["season"] = df.season.astype(str).str.zfill(4)
    best = None
    for K in (16, 24):
        for carry in (0.7, 0.85):
            for HA in (50, 70):
                for prom in (-100, -200):
                    cfg = dict(K=K, HA=HA, carry=carry, prom_init=prom, hl_fast=6, hl_slow=19)
                    f = add_diffs(build(df, cfg))
                    for C in (0.05, 0.2, 1.0):
                        p, y, _, _ = fit_eval(f, SEASONS[:5], "2425", C)
                        ll = log_loss(y, np.clip(p, 1e-6, 1-1e-6))
                        if best is None or ll < best[0]:
                            best = (ll, cfg, C, f)
    ll25, cfg, C, f = best
    print(f"tuned on 2024-25: LL={ll25:.4f}  K={cfg['K']} HA={cfg['HA']} carry={cfg['carry']} prom={cfg['prom_init']} C={C}")
    # shrink fit on walk-forward OOS (seasons 2122..2425), early games only
    oos = []
    for i, s in enumerate(SEASONS[2:6], start=2):
        p, y, rows, _ = fit_eval(f, SEASONS[:i], s, C)
        oos.append(pd.DataFrame({"p": p, "y": y, "min_gp": rows.min_gp.values}))
    oos = pd.concat(oos); early = oos[oos.min_gp < 6]
    grid = np.round(np.arange(0.2, 1.01, 0.05), 2)
    a0 = min(grid, key=lambda a: log_loss(early.y, np.clip(shrink(early.p.values, early.min_gp.values, a), 1e-6, 1-1e-6)))
    print(f"shrink a0={a0} on {len(early)} OOS early games: "
          f"{log_loss(early.y, np.clip(early.p,1e-6,1-1e-6)):.4f} -> "
          f"{log_loss(early.y, np.clip(shrink(early.p.values, early.min_gp.values, a0),1e-6,1-1e-6)):.4f}")
    json.dump({"cfg": cfg, "C": C, "a0": float(a0)}, open("soccer/spec.json", "w"))
    # frozen eval 2025-26
    p, y, rows, fitted = fit_eval(f, SEASONS[:6], "2526", C)
    p11 = shrink(p, rows.min_gp.values, a0)
    e = 1/(1+10**(-((rows.h_elo + cfg["HA"]) - rows.a_elo)/400)).values
    base = f.loc[f.season.isin(SEASONS[:6]) & (~f.draw), "y"].mean()
    res = [M(y, np.full_like(y, 0.5, float), "coin flip"),
           M(y, np.full_like(y, base, float), f"base rate ({base:.3f} home)"),
           M(y, e, "raw Elo (floor)"),
           {"model": "always Elo favorite", "n": len(y), "log_loss": None, "brier": None,
            "hit": round(float(np.mean((e > 0.5).astype(int) == y)), 4)},
           M(y, p, "REBUILD v1"), M(y, p11, "REBUILD v1.1 (shrink)")]
    R = pd.DataFrame(res)
    rows.assign(p_home=p11).to_csv("soccer/eval_2526.csv", index=False)
    m, sc = fitted
    co = pd.Series(m.coef_[0], index=FEATS).sort_values(key=abs, ascending=False)
    print("\ntop coefficients:"); print(co.head(8).round(3).to_string())
    print(f"\n2025-26 frozen holdout, DNB (decided games, draws {rows.shape[0]} of "
          f"{int((f.season=='2526').sum())} total excluded... n below):")
    pd.set_option("display.width", 160)
    print(R.to_string(index=False))

if __name__ == "__main__":
    run()
