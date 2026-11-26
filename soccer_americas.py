"""
The Americas boards, run alongside Big 5 inside the soccer repo.

Big 5 is not touched by any of this. It keeps its own spec, its own feature set including
shots, and its own frozen holdout. Pooling was rejected on measurement: these leagues have
no shot data, so a shared head would have to drop `sot_d` and `sh_d` for everyone, and that
costs Big 5 +0.0071 log loss on its 2025-26 holdout at p=0.012. Two heads, one repo.

Data. Results come from football-data's `new/` files, which carry date, teams, goals and
result but no shots. Fixtures do NOT exist there: `new_fixtures.csv` is a 404 and
`fixtures.csv` only carries the mmz4281 leagues. So upcoming matches come from ESPN, which
means the two sources name teams differently and have to be reconciled.

Protocol. Tuned on calendar 2024 only. 2025 opened once. 2026 onward is the live forward
test. Frozen at K=16, HA=90, carry=0.7, promoted init -100, C=0.05, a0=0.75. Those differ
from Big 5 in ways that match the leagues rather than looking like overfitting: a much
larger home edge, faster between-season regression, and a gentler penalty on new teams,
which is right where MLS expands rather than promotes and Liga MX has no relegation.

2025 holdout, n=946 decided:

    model 0.6043   raw Elo 0.6138   base rate 0.6597

    Brasileirao  n=281  +0.0166 over its own Elo  (t=+2.87, p=0.0044)
    LigaMX       n=260  +0.0145                   (t=+2.64, p=0.0089)
    MLS          n=405  +0.0012                   (t=+0.19, p=0.8479)

Brasileirao and LigaMX board. MLS does not: its head adds nothing over the rating it sits
on, which is the same signature that retired the CS2 series-length head and MLB F5. It is
still predicted, logged and graded on every run, so the decision stays under measurement
rather than being deleted. Same treatment as the NCAAF selection rule.
"""
import datetime as dt
import io
import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd
import requests

VERSION = "2026-08-03"

H = {"User-Agent": "Mozilla/5.0"}          # football-data.co.uk, fine with a UA
# ESPN 403s ANY custom User-Agent and only serves the requests default. Measured
# 2026-08-05 on the WNBA scoreboard: no header 200, "Mozilla/5.0" 403, an honest
# identifier with contact details 403, a full browser header set 403. mbm-wnba has
# always worked because it never set one. Do not "fix" this by adding a UA.
ESPN_H = {}
SRC = {"USA": "MLS", "MEX": "LigaMX", "BRA": "Brasileirao"}
ESPN_SLUG = {"MLS": "usa.1", "LigaMX": "mex.1", "Brasileirao": "bra.1"}
ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer"


def espn_scoreboard(lg, slug, start, end, limit=300):
    """ESPN scoreboard for a date range, as the parsed JSON: {"events": [...]}.

    ESPN began rejecting the ranged form (dates=YYYYMMDD-YYYYMMDD) with HTTP 400 in
    mid-September 2026. Big 5 got a day-by-day fallback then; this file did not, and
    because the old code called .json() on the 400 without looking at the status it read
    an error body as "no events". From 2026-09-21 the result backfill and the fixture feed
    both returned nothing, the store trailed ESPN by a day or two on every run, and the
    freshness guard withheld the Americas board for two months (2026-11-26 audit).

    Tries the range once, then walks the days. Raises only if every single call fails, so
    the caller's existing "feed unavailable" message still fires on a real outage.
    """
    rng = f"{start:%Y%m%d}-{end:%Y%m%d}"
    try:
        r = requests.get(f"{ESPN}/{slug}/scoreboard", headers=ESPN_H, timeout=45,
                         params={"dates": rng, "limit": limit})
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    events, seen, ok, day = [], set(), 0, start
    while day <= end:
        try:
            r = requests.get(f"{ESPN}/{slug}/scoreboard", headers=ESPN_H, timeout=45,
                             params={"dates": day.strftime("%Y%m%d"), "limit": limit})
            if r.status_code == 200:
                ok += 1
                for e in r.json().get("events", []):
                    if e.get("id") not in seen:
                        seen.add(e.get("id")); events.append(e)
        except Exception:
            pass
        day += dt.timedelta(days=1)
    if not ok:
        raise RuntimeError(f"ESPN refused every call for {lg} {rng}")
    print(f"  {lg}: ESPN ranged call refused; {len(events)} event(s) read day by day")
    return {"events": events}
FD = "https://www.football-data.co.uk/new"

# Shots are absent from every one of these leagues, so `sot_d` and `sh_d` sit at exactly
# zero and are left out rather than carried as dead columns. MLS is the reference level
# for the league dummies.
FEATS = ["elo_d", "gnetf_d", "gnets_d", "rest_d", "h_gp", "a_gp",
         "h_prom", "a_prom", "cong_h", "cong_a", "lg_LigaMX", "lg_Brasileirao"]

# Club-type words only. An earlier version also stripped "atletico" and "deportivo", which
# is wrong: those distinguish teams rather than decorate them, and it collapsed
# Atletico-MG, Athletico-PR and Atletico GO toward each other.
_STRIP = r"\b(fc|cf|ec|sc|afc|ac|cd|ca|club|clube|de|do|da|futbol|futebol)\b"


def norm_team(s):
    """Fold a club name to a comparison key.

    Checked against every name in all three leagues, 94 clubs, and produces no collisions,
    so it is safe as the primary matcher between ESPN and football-data. Anything it misses
    goes in the alias file rather than being handled by loosening this.
    """
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(_STRIP, " ", s)
    return re.sub(r"[^a-z0-9]", "", s)


def refresh_results(path):
    """Pull every league's results and rewrite the store.

    The `new/` files also carry sixty-odd closing-price columns. They are dropped here and
    never enter the pipeline.
    """
    frames = []
    for code, lg in SRC.items():
        try:
            r = requests.get(f"{FD}/{code}.csv", headers=H, timeout=60)
            if r.status_code != 200 or len(r.text) < 200:
                print(f"  {lg}: results feed unavailable (HTTP {r.status_code})")
                continue
            d = pd.read_csv(io.StringIO(r.text), on_bad_lines="skip",
                            encoding_errors="ignore")
        except Exception as ex:
            print(f"  {lg}: results fetch failed ({type(ex).__name__})")
            continue
        d.columns = [c.encode("ascii", "ignore").decode() for c in d.columns]
        d["date"] = pd.to_datetime(d.Date, dayfirst=True, errors="coerce")
        d = d.dropna(subset=["date", "HG", "AG"])
        frames.append(pd.DataFrame({
            "date": d.date, "season": d.Season.astype(str), "league": lg,
            "HomeTeam": d.Home.astype(str).str.strip(),
            "AwayTeam": d.Away.astype(str).str.strip(),
            "FTHG": d.HG.astype(float), "FTAG": d.AG.astype(float),
            "HS": np.nan, "AS": np.nan, "HST": np.nan, "AST": np.nan}))
    if not frames:
        return pd.read_csv(path, parse_dates=["date"]) if os.path.exists(path) \
            else pd.DataFrame()
    m = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    m.to_csv(path, index=False)
    print(f"  americas results: {len(m)} matches, "
          + ", ".join(f"{k} {v}" for k, v in m.league.value_counts().items()))
    return m


def fetch_fixtures(known, alias, horizon_days=2, leagues=None):
    """Upcoming matches from ESPN, mapped onto football-data's club names.

    `known` is the set of names already in the results store, per league. `alias` is scoped
    per league too, because a flat map would bridge leagues: "Santos" is a Brasileirao club
    and "Santos Laguna" is a Liga MX one, and they are the only pair in the three that
    collide on a bare name.

    A fixture whose teams cannot be mapped is skipped and reported by name, because a
    silently dropped fixture and a silently mis-joined one look identical on the board and
    only one of them is safe.
    """
    today = dt.date.today()
    rng = f"{today:%Y%m%d}-{today + dt.timedelta(days=horizon_days):%Y%m%d}"
    rows, unmapped = [], []
    for lg, slug in ESPN_SLUG.items():
        if leagues and lg not in leagues:
            continue
        try:
            r = espn_scoreboard(lg, slug, today, today + dt.timedelta(days=horizon_days), limit=200)
        except Exception as ex:
            print(f"  {lg}: fixture feed unavailable ({type(ex).__name__})")
            continue
        idx = {norm_team(n): n for n in known.get(lg, set())}
        amap = alias.get(lg, {}) if isinstance(alias.get(lg), dict) else {}

        def match(disp, short):
            for cand in (amap.get(disp), amap.get(short), disp, short):
                if cand is None:
                    continue
                hit = idx.get(norm_team(cand))
                if hit:
                    return hit
            return None

        for e in r.get("events", []):
            # "pre" only. completed=False includes matches currently being played.
            if e.get("status", {}).get("type", {}).get("state") != "pre":
                continue
            c = e["competitions"][0]
            t = {x["homeAway"]: x["team"] for x in c.get("competitors", [])}
            if "home" not in t or "away" not in t:
                continue
            h = match(t["home"].get("displayName", ""), t["home"].get("shortDisplayName", ""))
            a = match(t["away"].get("displayName", ""), t["away"].get("shortDisplayName", ""))
            if not h or not a:
                unmapped.append(f"{lg}: {t['away'].get('displayName')} @ "
                                f"{t['home'].get('displayName')}")
                continue
            rows.append({"league": lg, "date": pd.Timestamp(e["date"]).tz_localize(None).normalize(),
                         "HomeTeam": h, "AwayTeam": a})
    if unmapped:
        print(f"  {len(unmapped)} fixture(s) skipped, club names not matched. Add these to "
              f"the alias file:")
        for u in unmapped[:8]:
            print(f"     {u}")
    return pd.DataFrame(rows)


def backfill_from_espn(store, alias, days=30):
    """Results football-data has not published yet, pulled from ESPN.

    football-data updates a couple of times a week, so a board that refuses to publish on
    stale data goes quiet for days at a time. Measured 2026-08-08: Brasileirao's file
    stopped on 07-31 with four matches played since, MLS on 08-02 with one. This closes
    that gap the same way the WNBA board closes the wehoop gap.

    Lossless here, unlike Big 5. The `new/` files carry no shot columns, so the engine
    already runs these leagues on goals alone and ESPN supplies exactly what is used.

    Returns rows in the store's own shape. Empty on any failure, and the caller then keeps
    whatever it had.
    """
    out = []
    for lg, slug in ESPN_SLUG.items():
        sub = store[store.league == lg]
        if not sub.empty:
            last = pd.to_datetime(sub.date).max().date()
        else:
            last = dt.date.today() - dt.timedelta(days=days)
        start = max(last + dt.timedelta(days=1), dt.date.today() - dt.timedelta(days=days))
        if start > dt.date.today():
            continue
        idx = {norm_team(n): n for n in (set(sub.HomeTeam) | set(sub.AwayTeam))}
        amap = alias.get(lg, {}) if isinstance(alias.get(lg), dict) else {}

        def match(disp, short):
            for cand in (amap.get(disp), amap.get(short), disp, short):
                if cand and norm_team(cand) in idx:
                    return idx[norm_team(cand)]
            return None

        rng = f"{start:%Y%m%d}-{dt.date.today():%Y%m%d}"
        try:
            r = espn_scoreboard(lg, slug, start, dt.date.today())
        except Exception as ex:
            print(f"  {lg}: result backfill unavailable ({type(ex).__name__})")
            continue
        skipped = []
        for e in r.get("events", []):
            if not e.get("status", {}).get("type", {}).get("completed"):
                continue
            c = e["competitions"][0]
            t = {x["homeAway"]: x for x in c.get("competitors", [])}
            if "home" not in t or "away" not in t:
                continue
            h = match(t["home"]["team"].get("displayName", ""),
                      t["home"]["team"].get("shortDisplayName", ""))
            a = match(t["away"]["team"].get("displayName", ""),
                      t["away"]["team"].get("shortDisplayName", ""))
            try:
                hg, ag = float(t["home"]["score"]), float(t["away"]["score"])
            except (TypeError, ValueError):
                continue
            if not h or not a:
                skipped.append(f"{t['away']['team'].get('displayName')} @ "
                               f"{t['home']['team'].get('displayName')}")
                continue
            when = pd.Timestamp(e["date"]).tz_localize(None).normalize()
            out.append({"date": when, "season": season_of(lg, when, store), "league": lg,
                        "HomeTeam": h, "AwayTeam": a, "FTHG": hg, "FTAG": ag,
                        "HS": np.nan, "AS": np.nan, "HST": np.nan, "AST": np.nan})
        if skipped:
            print(f"  {lg}: {len(skipped)} result(s) skipped, club names not matched: "
                  + ", ".join(skipped[:4]))
    if not out:
        return pd.DataFrame()
    add = pd.DataFrame(out)
    have = set(store.league + "|" + pd.to_datetime(store.date).dt.strftime("%Y-%m-%d")
               + "|" + store.HomeTeam + "|" + store.AwayTeam)
    key = (add.league + "|" + add.date.dt.strftime("%Y-%m-%d") + "|"
           + add.HomeTeam + "|" + add.AwayTeam)
    add = add[~key.isin(have)]
    if len(add):
        print(f"  backfilled {len(add)} result(s) from ESPN that football-data has not "
              f"published: " + ", ".join(f"{k} {v}" for k, v in
                                         add.league.value_counts().items()))
    return add


def season_of(league, when, store):
    """Which season label a future fixture belongs to.

    Taken from whatever the store already calls the matches nearest that date, rather than
    derived, because the three leagues disagree: MLS and Brasileirao label by calendar
    year, Liga MX by a split "2026/2027" spanning Apertura and Clausura.
    """
    g = store[store.league == league]
    if g.empty:
        return str(when.year)
    near = g.iloc[(pd.to_datetime(g.date) - when).abs().argsort()[:1]]
    return str(near.season.iloc[0])


def build_features(store, fixtures, cfg, E):
    """Per-league state pass, then one frame.

    Each league gets its own ordered season list, so promotion detection compares a club to
    that league's previous season rather than to whichever league happens to sit next in a
    shared list. It also keeps Elo and the league-mean decays per league, which is correct:
    these three never play each other.
    """
    g_all = pd.concat([store, fixtures], ignore_index=True) if len(fixtures) else store
    g_all["date"] = pd.to_datetime(g_all.date)
    out = []
    for lg, g in g_all.groupby("league"):
        E.SEASONS = g.groupby("season").date.min().sort_values().index.tolist()
        out.append(E.add_diffs(E.build(g.sort_values("date").reset_index(drop=True), cfg)))
    f = pd.concat(out, ignore_index=True)
    f["date"] = pd.to_datetime(f.date)
    for lg in ("LigaMX", "Brasileirao"):
        f[f"lg_{lg}"] = (f.league == lg).astype(int)
    return f


def load_alias(path):
    """Load the per-league alias map, dropping the comment key."""
    if not os.path.exists(path):
        return {}
    try:
        return {k: v for k, v in json.load(open(path)).items()
                if not k.startswith("_")}
    except Exception as ex:
        print(f"  alias file unreadable ({type(ex).__name__}), matching on names alone")
        return {}
