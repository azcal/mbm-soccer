"""
Big 5 results from ESPN, for when football-data has not published.

football-data has been the Big 5 source since the model was built, and on 2026-08-16 it had
published no 2026-27 file for any of the five leagues. Worse, its server 301-redirected the
missing files to whatever name was closest -- E0.csv to EC.csv (National League), SP1.csv to
P1.csv (Primeira Liga) -- so the board was quietly ingesting the wrong division until the
Div check caught it. With the store cleaned, Big 5 had no source at all and went dark.

This fills that gap from ESPN, and unlike the Americas equivalent it is **lossless**. The
Americas `new/` files carry no shot columns, so that board runs on goals alone. Big 5 does
not: `engine.FEATS` includes `sot_d` and `sh_d`, and the spec puts the shotless penalty at
+0.0071 log loss, which is the whole reason Big 5 keeps its own head. ESPN's soccer summary
endpoint carries `totalShots` and `shotsOnTarget` per team, so the same four columns arrive.

Verified 2026-08-16 against six EPL matches football-data already held: shots and shots on
target agreed exactly, 6 of 6, with the home side first in `boxscore.teams`.

Club names come from `big5_alias.json`, hand-written and verified so that all 96 current Big
5 clubs resolve. Hand-written rather than fuzzy-matched on purpose: a similarity pass on the
same names proposed Coventry City -> Man City, Hull City -> Man City, Malaga -> Mallorca and
Le Mans -> Lens. A wrong join here is the same class of silent contamination the Div check
exists to stop, so a name that does not map exactly is skipped and printed, never guessed.

These rows are provisional. `refresh_results` replaces any row whose league|date|home|away
key it also fetches, so when football-data does publish, its version -- with referee, half
time and the rest -- overwrites this one. The gap heals itself.
"""
import datetime as dt
import json
import os
import time

import unicodedata

import numpy as np
import pandas as pd
import requests

VERSION = "2026-09-16"

# ESPN 403s any custom User-Agent. See the note in espn_roster.py.
H = {}
ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer"
SLUG = {"EPL": "eng.1", "LaLiga": "esp.1", "SerieA": "ita.1",
        "Bundesliga": "ger.1", "Ligue1": "fra.1"}

def _scoreboard(league, rng, timeout=45, retries=3):
    """One scoreboard pull with backoff, and a per-day fallback when the ranged request keeps
    failing. ESPN throttles GitHub runners now and then (429/5xx); the freshness probe has
    retried single days since August and kept working through the same hours this fetcher
    failed. Returns the merged events list, or None when nothing could be fetched."""
    import time
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(f"{ESPN}/{SLUG[league]}/scoreboard", headers=H, timeout=timeout,
                             params={"dates": rng, "limit": 500})
            if r.status_code == 200:
                return r.json().get("events", [])
            last = f"HTTP {r.status_code}"
        except Exception as ex:
            last = type(ex).__name__
        time.sleep(1.5 * (2 ** attempt))
    # per-day fallback
    events, ok = [], 0
    try:
        a, b = rng.split("-"); d0 = dt.datetime.strptime(a, "%Y%m%d").date(); d1 = dt.datetime.strptime(b, "%Y%m%d").date()
    except Exception:
        print(f"  {league}: ESPN unavailable ({last})"); return None
    day = d0
    while day <= d1:
        try:
            r = requests.get(f"{ESPN}/{SLUG[league]}/scoreboard", headers=H, timeout=timeout,
                             params={"dates": day.strftime("%Y%m%d")})
            if r.status_code == 200:
                events += r.json().get("events", []); ok += 1
            time.sleep(0.4)
        except Exception:
            pass
        day += dt.timedelta(days=1)
    if ok == 0:
        print(f"  {league}: ESPN unavailable ({last}; per-day fallback also failed)"); return None
    print(f"  {league}: ESPN ranged call failed ({last}); recovered {len(events)} event(s) day by day")
    return events



def load_alias(path):
    """Load the club-name map, and SAY SO if it cannot be loaded.

    The first version swallowed every exception and returned {}, so a missing or misplaced
    file looked exactly like a file with no entries -- and the backfill then skipped every
    fixture on names the alias explicitly contained, reporting them as unmatched. That is
    the silent-empty pattern, inside the module written to close a silent-empty. It cost a
    full run to diagnose and would have cost more if the skipped names had been less
    obviously wrong.
    """
    if not os.path.exists(path):
        print(f"  big5 alias NOT FOUND at {path} -- every club name will be matched raw, "
              f"so anything ESPN spells differently will be skipped")
        return {}
    try:
        raw = json.load(open(path, encoding="utf-8"))
    except Exception as ex:
        print(f"  big5 alias at {path} could not be read ({type(ex).__name__}: {ex})")
        return {}
    a = {k: v for k, v in raw.items() if not k.startswith("_")}
    print(f"  big5 alias loaded: {sum(len(v) for v in a.values())} entries across "
          f"{len(a)} league(s)")
    return a


def _fold(s):
    """Accent-folded, case-folded club name.

    Deterministic, not fuzzy: it strips combining marks and lowercases, nothing else.
    Verified 2026-08-16 that no two clubs in any Big 5 store fold together, so a fold match
    is exact rather than a guess. This is what lets the alias carry only genuine name
    differences (Man City, Nott'm Forest) instead of also carrying every accent, and it
    means a missing alias file degrades to "accents still work" rather than to nothing.
    """
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).strip().lower()


def _resolve(name, amap, known, folded):
    """Store name for an ESPN club, or None. Exact, then alias, then accent fold."""
    if name in known:
        return name
    mapped = amap.get(name)
    if mapped and mapped in known:
        return mapped
    return folded.get(_fold(mapped or name))


def _stat(team, name):
    for s in team.get("statistics", []):
        if s.get("name") == name:
            try:
                return float(s.get("displayValue"))
            except (TypeError, ValueError):
                return np.nan
    return np.nan


def fetch(league, since, until=None, known=None, alias=None, timeout=45, pause=0.4):
    """Completed matches for one league after `since`, in matches.csv's own shape.

    `known` is the set of club names already in the store for this league. A fixture whose
    clubs do not both resolve into it is skipped and named, never guessed: the store is the
    authority on what a club is called, and inventing a name here would write a phantom club
    into the Elo.
    """
    until = until or dt.date.today()
    start = since + dt.timedelta(days=1)
    if start > until:
        return pd.DataFrame(), []
    amap = (alias or {}).get(league, {})
    known = known or set()
    folded = {_fold(k): k for k in known}
    rng = f"{start:%Y%m%d}-{until:%Y%m%d}"
    _ev = _scoreboard(league, rng, timeout=timeout)
    if _ev is None:
        return pd.DataFrame(), []
    j = {"events": _ev}

    rows, skipped = [], []
    for e in j.get("events", []):
        if not e.get("status", {}).get("type", {}).get("completed"):
            continue
        for c in e.get("competitions", []):
            sides = {x.get("homeAway"): x for x in c.get("competitors", [])}
            if "home" not in sides or "away" not in sides:
                continue
            rawh = sides["home"]["team"].get("displayName", "")
            rawa = sides["away"]["team"].get("displayName", "")
            hn, an = _resolve(rawh, amap, known, folded), _resolve(rawa, amap, known, folded)
            if hn is None or an is None:
                bad = [r for r, v in ((rawh, hn), (rawa, an)) if v is None]
                skipped.append(f"{league}: no match for {bad} (in {rawa} @ {rawh})")
                continue
            try:
                hg, ag = float(sides["home"]["score"]), float(sides["away"]["score"])
            except (TypeError, ValueError):
                continue
            when = pd.Timestamp(e["date"]).tz_localize(None).normalize()

            # Shots need the per-match summary; the scoreboard does not carry them. One
            # extra call per completed match, which is a handful a week per league.
            hs = as_ = hst = ast = np.nan
            try:
                s = requests.get(f"{ESPN}/{SLUG[league]}/summary", headers=H,
                                 params={"event": e["id"]}, timeout=timeout).json()
                ts = s.get("boxscore", {}).get("teams", [])
                if len(ts) == 2:
                    # Identify by team id rather than position: the ordering of
                    # boxscore.teams is not documented and must not be assumed.
                    hid = str(sides["home"]["team"].get("id"))
                    ht = next((t for t in ts if str(t.get("team", {}).get("id")) == hid), None)
                    at = next((t for t in ts if t is not ht), None)
                    if ht is not None and at is not None:
                        hs, hst = _stat(ht, "totalShots"), _stat(ht, "shotsOnTarget")
                        as_, ast = _stat(at, "totalShots"), _stat(at, "shotsOnTarget")
                time.sleep(pause)
            except Exception:
                pass                       # goals still land; engine tolerates NaN shots

            rows.append({"Date": when.strftime("%d/%m/%Y"), "Time": "",
                         "HomeTeam": hn, "AwayTeam": an,
                         "FTHG": hg, "FTAG": ag,
                         "FTR": "H" if hg > ag else ("A" if ag > hg else "D"),
                         "HS": hs, "AS": as_, "HST": hst, "AST": ast,
                         "league": league, "date": when})
    return pd.DataFrame(rows), skipped


def fixtures(m, divs, alias_path, horizon_days, timeout=45):
    """Upcoming Big 5 fixtures from ESPN, in the shape daily_soccer's pseudo-rows expect.

    football-data's fixtures.csv is a next-48-hours feed: measured 2026-08-18 it carried
    three rows across two days, so a seven-day horizon bought nothing and EPL -- three days
    from its opening weekend -- was simply absent. ESPN listed ten EPL fixtures in the same
    window and twenty within a fortnight.

    Seven days is roughly when books post soccer lines, so a fixture surfaces about when it
    becomes shoppable rather than the night before.

    Only state == "pre": a match under way or finished must never reach a pre-match board.
    Club names resolve through the same alias and accent fold as the results backfill, and
    an unresolved name is skipped and printed rather than guessed.
    """
    alias = load_alias(alias_path)
    today = dt.date.today()
    rng = f"{today:%Y%m%d}-{today + dt.timedelta(days=horizon_days):%Y%m%d}"
    out, skipped = [], []
    for league in divs.values():
        sub = m[(m.league == league) & m.FTHG.notna()]
        if not len(sub):
            continue
        known = set(sub.HomeTeam) | set(sub.AwayTeam)
        folded = {_fold(k): k for k in known}
        amap = alias.get(league, {})
        _ev = _scoreboard(league, rng, timeout=timeout)
        if _ev is None:
            print(f"  {league}: ESPN fixtures unavailable")
            continue
        j = {"events": _ev}
        for e in j.get("events", []):
            if e.get("status", {}).get("type", {}).get("state") != "pre":
                continue
            for c in e.get("competitions", []):
                sides = {x.get("homeAway"): x for x in c.get("competitors", [])}
                if "home" not in sides or "away" not in sides:
                    continue
                rawh = sides["home"]["team"].get("displayName", "")
                rawa = sides["away"]["team"].get("displayName", "")
                hn = _resolve(rawh, amap, known, folded)
                an = _resolve(rawa, amap, known, folded)
                if hn is None or an is None:
                    bad = [x for x, v in ((rawh, hn), (rawa, an)) if v is None]
                    skipped.append(f"{league}: no match for {bad} (in {rawa} @ {rawh})")
                    continue
                out.append({"league": league,
                            "date": pd.Timestamp(e["date"]).tz_localize(None).normalize(),
                            "HomeTeam": hn, "AwayTeam": an})
    if skipped:
        print(f"  {len(skipped)} Big 5 fixture(s) skipped, club names not matched:")
        for x in skipped[:6]:
            print(f"     {x}")
    if not out:
        return pd.DataFrame(columns=["league", "date", "HomeTeam", "AwayTeam"])
    f = pd.DataFrame(out).drop_duplicates(subset=["league", "date", "HomeTeam", "AwayTeam"])
    print("  Big 5 fixtures from ESPN: "
          + ", ".join(f"{k} {v}" for k, v in f.league.value_counts().items()))
    return f


def backfill(m, divs, alias_path, season_code_fn):
    """Everything the store is missing across all five leagues.

    `m` is the loaded matches frame; each league's own newest played date decides where its
    fill starts, so a league mid-season and a league yet to kick off are handled the same
    way without special-casing.
    """
    alias = load_alias(alias_path)
    out, skips = [], []
    for league in divs.values():
        sub = m[(m.league == league) & m.FTHG.notna()]
        if not len(sub):
            continue
        last = pd.to_datetime(sub.date).max().date()
        known = set(sub.HomeTeam) | set(sub.AwayTeam)
        got, sk = fetch(league, last, known=known, alias=alias)
        skips += sk
        if len(got):
            got["season"] = [season_code_fn(d.date()) for d in got.date]
            out.append(got)
    if skips:
        print(f"  {len(skips)} Big 5 result(s) skipped, club names not matched:")
        for s in skips[:6]:
            print(f"     {s}")
    if not out:
        return pd.DataFrame()
    add = pd.concat(out, ignore_index=True)
    have = set(m.league + "|" + pd.to_datetime(m.date).dt.strftime("%Y-%m-%d")
               + "|" + m.HomeTeam + "|" + m.AwayTeam)
    key = (add.league + "|" + add.date.dt.strftime("%Y-%m-%d")
           + "|" + add.HomeTeam + "|" + add.AwayTeam)
    add = add[~key.isin(have)]
    if len(add):
        n_sh = int(add.HS.notna().sum())
        print(f"  backfilled {len(add)} Big 5 result(s) from ESPN that football-data has "
              f"not published ({n_sh} with shot data): "
              + ", ".join(f"{k} {v}" for k, v in add.league.value_counts().items()))
    return add
