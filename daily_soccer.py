"""
MBM Big-5 soccer DNB daily runner (rebuild v1, frozen spec).

- Refreshes current-season results from football-data.co.uk (tolerates preseason 404s)
- Ingests upcoming Big-5 fixtures (7-day horizon), predicts P(home | no draw)
- Discord push routed through tg_delta so refreshes only report what moved
- Grades logged predictions (draws = push), rolling metrics + last-300 tripwire

Train window frozen through 2025-26; 2026-27 is the forward test.
Usage: python daily_soccer.py   (env TEST_FIXTURES=path.csv to inject a test slate)
"""
import io, json, os, sys, datetime as dt

import numpy as np, pandas as pd, requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tg_delta
import board_format as BF
import notify
import engine as E
import soccer_americas as AM
import big5_espn as B5
import freshness as FR
import mbm_checks as CHK

# Shared files must be at least this new. A minimum rather than an exact match, so a later
# improvement to a shared file does not force an edit here. Raise it only for a change this
# runner depends on.
#   2026-08-03  board_format.line() field separator. Big 5 club names run to 18 characters
#               and a matchup string to 35, so without it the widest row on every board
#               renders as an unstyled run-on line.
SHARED_MIN = "2026-08-03"
for _mod in (BF, notify, tg_delta, AM, B5, FR, CHK):
    _have = getattr(_mod, "VERSION", "")
    if _have < SHARED_MIN:
        raise ImportError(
            f"{_mod.__name__}.py is version {_have or 'unstamped'}, this runner needs "
            f"{SHARED_MIN} or newer. Copy the current shared files into this repo.")

H = {"User-Agent": "Mozilla/5.0"}
DIVS = {"E0": "EPL", "SP1": "LaLiga", "I1": "SerieA", "D1": "Bundesliga", "F1": "Ligue1"}
MATCHES = os.path.join(HERE, "data", "matches.csv")
LOGS = os.path.join(HERE, "logs")
SPEC = json.load(open(os.path.join(HERE, "spec.json")))
CFG, C, A0 = SPEC["cfg"], SPEC["C"], SPEC["a0"]
TRAIN_THROUGH = "2526"

# Printed on every pick. The model's number is P(home wins | no draw), which is only a
# valid price against a draw-no-bet market. Taken on the 1X2 moneyline instead, a draw
# loses outright rather than pushing, and draws are about a quarter of Big 5 matches. The
# embed title carries the market, but the row is what gets acted on, so it carries it too.
MARKET_TAG = SPEC.get("market_tag", "DNB")

# Fixtures are published weeks out, so the horizon controls repetition rather than
# coverage. Seven days, matching daily_nfl and daily_ncaaf: a Tuesday run reaches the
# following Monday, so a whole weekend slate is on the board from the start of the week
# rather than appearing on the Friday. Repetition is not the cost it looks like -- the
# delta gate suppresses a row that has not moved, so a fixture posts once and then only
# when its price changes.
#
# The old value was 2, which on 2026-08-14 would have caught 4 of the 5 upcoming Big 5
# fixtures and missed a Sunday match entirely.
HORIZON_DAYS = SPEC.get("horizon_days", 7)

# The Americas boards. Big 5 above is untouched by all of this: separate spec, separate
# feature set, separate frozen holdout. See soccer_americas.py for why they are not pooled.
NEW = SPEC.get("new_leagues", {})
NEW_BOARD = set(NEW.get("board", []))        # Brasileirao, LigaMX
NEW_RETIRED = set(NEW.get("retired", []))    # MLS: predicted and graded, never pushed
NEW_MATCHES = os.path.join(HERE, "data", "americas_matches.csv")
NEW_ALIAS = os.path.join(HERE, "data", "espn_soccer_alias.json")
BIG5_ALIAS = os.path.join(HERE, "data", "big5_alias.json")

# Country codes throughout, so every soccer row names its competition. BRA and MEX were
# already country codes, so the Big 5 follow the same convention rather than a mix of
# country codes and league abbreviations.
# Heading shown above that league's fixtures. Grouping beats a per-row code: a reader
# scanning for one competition finds it as a block rather than picking prefixes out of a
# mixed list.
LG_NAME = {"EPL": "Premier League", "LaLiga": "La Liga", "SerieA": "Serie A",
           "Bundesliga": "Bundesliga", "Ligue1": "Ligue 1",
           "Brasileirao": "Brasileirao Serie A", "LigaMX": "Liga MX", "MLS": "MLS"}
# Order they appear in, so the board reads the same way every night.
LG_ORDER = ["EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1",
            "Brasileirao", "LigaMX", "MLS"]

# Tracks board_format rather than restating it. This is the console-side copy, and a
# local 0.05 here while board_format ran at 0.00 meant the Actions log and Discord
# quoted different prices for the same game.
MIN_EDGE = BF.MIN_EDGE


def min_odds(p):
    import math
    p = min(max(float(p), 0.01), 0.99)
    return math.ceil((1 + MIN_EDGE) / p * 100) / 100.0


def season_code(d):
    y = d.year % 100
    return f"{y:02d}{y+1:02d}" if d.month >= 7 else f"{y-1:02d}{y:02d}"


def extend_seasons(code):
    while code not in E.SEASONS:
        last = E.SEASONS[-1]
        a = int(last[:2]) + 1
        E.SEASONS.append(f"{a:02d}{a+1:02d}")


def refresh_results():
    """Pull this season's results, trusting the FILE's own Div column rather than the URL.

    Measured 2026-08-16: mmz4281/2627/E0.csv returned HTTP 200 carrying National League
    matches (Div "EC": Altrincham, Boreham Wood) and 2627/SP1.csv returned Primeira Liga
    (Div "P1": Benfica, Estoril). Both cleared the old status-and-length gate, and because
    the runner labelled every row with the league it ASKED for, Altrincham went into the
    store as EPL and Benfica as La Liga -- feeding the Elo, the form decays and every
    price. Nothing said a word; the only symptom was a suspiciously small "+21 rows".

    So: read the Div column and keep only rows carrying the division actually requested. A
    file serving the wrong league now contributes nothing and says so. That also covers the
    HTTP 300 multiple-choice pages football-data returns for a season file that does not
    exist yet (I1, D1, F1 today) -- the old length gate caught those by luck, and a
    slightly longer redirect page would have sailed through.

    Discarding rather than raising is deliberate: the league simply does not update, and if
    matches were played the freshness guard catches it downstream. A source serving the
    wrong data must not be able to poison the store, and must not be able to take the whole
    board down either.
    """
    m = pd.read_csv(MATCHES, parse_dates=["date"])
    m["season"] = m.season.astype(str).str.zfill(4)
    cur = season_code(dt.date.today())
    extend_seasons(cur)
    new, wrong = [], []
    for div, lg in DIVS.items():
        try:
            r = requests.get(f"https://www.football-data.co.uk/mmz4281/{cur}/{div}.csv",
                             headers=H, timeout=60)
            if r.status_code != 200 or len(r.text) < 200:
                continue
            # A 301 to a different filename means the file we asked for does not exist and
            # Apache's MultiViews picked the nearest name it could find. Measured
            # 2026-08-16: 2627/E0.csv -> 2627/EC.csv (National League) and 2627/SP1.csv ->
            # 2627/P1.csv (Primeira Liga), both HTTP 200 after the redirect. Catching it
            # here says exactly what happened; the Div check below is the backstop for a
            # substituted file that happens to carry a plausible division.
            if r.url.rsplit("/", 1)[-1] != f"{div}.csv":
                wrong.append(f"{div}.csv does not exist, the server redirected to "
                             f"{r.url.rsplit('/', 1)[-1]}")
                continue
            r.encoding = "utf-8-sig"     # same BOM trap as fixtures.csv
            d = pd.read_csv(io.StringIO(r.text), on_bad_lines="skip",
                            encoding_errors="ignore")
        except Exception as ex:
            print(f"  {lg}: results feed unavailable ({type(ex).__name__})")
            continue
        d.columns = [str(c).strip().lstrip("\ufeff") for c in d.columns]
        if "Div" not in d.columns:
            wrong.append(f"{div}.csv has no Div column, got {list(d.columns)[:5]}")
            continue
        served = sorted(set(d.Div.dropna().astype(str)))
        d = d[d.Div.astype(str) == div]
        if not len(d):
            wrong.append(f"asked {div} ({lg}), file contains {served}")
            continue
        if served != [div]:
            print(f"  {lg}: {div}.csv also contained {[x for x in served if x != div]}, "
                  f"those rows dropped")
        keep = [c for c in ("Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
                            "HS", "AS", "HST", "AST") if c in d.columns]
        d = d[keep].dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        d["league"] = lg
        d["season"] = cur
        new.append(d)
    if wrong:
        print("SOURCE MISMATCH, these leagues contributed nothing this run:")
        for w in wrong:
            print(f"  {w}")
        print("  (football-data is serving another division at that URL. Rows were "
              "discarded rather than filed under the wrong league.)")
    if new:
        n = pd.concat(new, ignore_index=True)
        n["date"] = pd.to_datetime(n.Date, format="mixed", dayfirst=True)
        key = lambda x: x.league + "|" + x.date.dt.strftime("%Y-%m-%d") + "|" + x.HomeTeam + "|" + x.AwayTeam
        m = m[~key(m).isin(set(key(n)))]
        m = pd.concat([m, n], ignore_index=True).sort_values("date").reset_index(drop=True)
        print(f"results refreshed: +{len(n)} rows this season")
    else:
        print("no current-season results yet (preseason)")

    # Fill whatever football-data has not published, from ESPN. Lossless: ESPN carries
    # totalShots and shotsOnTarget, so HS/AS/HST/AST arrive too and the Big 5 head runs on
    # the feature set it was trained on. These rows are provisional -- the dedup above
    # replaces any key football-data later serves, so its version wins when it appears.
    try:
        add = B5.backfill(m, DIVS, BIG5_ALIAS, season_code)
        if len(add):
            m = pd.concat([m, add], ignore_index=True).sort_values("date").reset_index(drop=True)
    except Exception as ex:
        print(f"  Big 5 ESPN backfill failed ({type(ex).__name__}: {ex}), "
              f"continuing on football-data alone")
    m.to_csv(MATCHES, index=False)
    return m


def fetch_fixtures(horizon_days=HORIZON_DAYS):
    """Upcoming Big 5 fixtures. Returns (frame, feed_ok).

    feed_ok is False when the pull failed or came back in a shape we could not read, so the
    caller can tell an outage from a genuinely empty week. This function used to return a
    bare empty frame in both cases and the board printed "no upcoming fixtures", which is
    exactly incident 3.

    That was not hypothetical. Measured 2026-08-14: requests decodes fixtures.csv as
    ISO-8859-1, so the UTF-8 byte-order mark arrives as the three characters "i>>?" rather
    than one \ufeff. `lstrip("\ufeff")` could not match it, the first column stayed
    "<BOM>Div", the `"Div" not in d.columns` test fired, and 127 fixtures -- including four
    La Liga matches on the opening weekend -- became zero with nothing printed. Forcing
    utf-8-sig makes requests strip the BOM itself.
    """
    tf = os.environ.get("TEST_FIXTURES")
    if tf and os.path.exists(tf):
        d = pd.read_csv(tf)
    else:
        try:
            r = requests.get("https://www.football-data.co.uk/fixtures.csv",
                             headers=H, timeout=60)
            r.raise_for_status()
            # Not lstrip: the BOM may already have been mangled by the decode. Naming the
            # encoding makes requests handle it before we ever see the text.
            r.encoding = "utf-8-sig"
            d = pd.read_csv(io.StringIO(r.text), on_bad_lines="skip",
                            encoding_errors="ignore")
        except Exception as ex:
            print(f"Big 5 fixture feed unavailable: {type(ex).__name__}")
            return pd.DataFrame(), False
    # A readable file with no Div column is a shape change, not an empty week. Say so.
    d.columns = [str(c).strip().lstrip("\ufeff") for c in d.columns]
    if "Div" not in d.columns:
        print(f"Big 5 fixture feed has no Div column, got {list(d.columns)[:6]} -- "
              f"the file's shape changed, this is not an empty slate")
        return pd.DataFrame(), False
    if d.empty:
        return pd.DataFrame(), True
    d = d[d.Div.isin(DIVS)].copy()
    if d.empty:
        return d, True
    d["league"] = d.Div.map(DIVS)
    d["date"] = pd.to_datetime(d.Date, format="mixed", dayfirst=True)
    today = pd.Timestamp(dt.date.today())
    d = d[(d.date >= today) & (d.date <= today + pd.Timedelta(days=horizon_days))]
    return d[["league", "date", "HomeTeam", "AwayTeam"]], True


def run_americas(chk):
    """Board Brasileirao and LigaMX; predict, log and grade MLS without pushing it.

    Returns (rows, alerts). A row is a dict the caller folds into the shared width pass,
    so both boards share one column layout and one delta gate.

    MLS is retired rather than deleted. Its head measured +0.0012 over its own Elo floor on
    the 2025 holdout at p=0.85, so publishing its prices would be publishing a number with
    no demonstrated content. It is still forecast and graded on every run, which keeps the
    decision under measurement instead of resting on one holdout forever. Same treatment as
    the NCAAF selection rule.
    """
    cfg, C_, a0_ = NEW["cfg"], NEW["C"], NEW["a0"]
    store = AM.refresh_results(NEW_MATCHES)
    if store is None or not len(store):
        print("  americas: no results store, skipping")
        return [], [], None, "Americas withheld: no results store."
    store["date"] = pd.to_datetime(store.date)
    alias = AM.load_alias(NEW_ALIAS)
    # football-data publishes a couple of times a week, so fill the tail from ESPN before
    # anything is measured or modelled. Lossless for these leagues: the new/ files carry no
    # shot columns, so goals are all the engine uses and goals are what ESPN gives.
    _add = AM.backfill_from_espn(store, alias)
    if len(_add):
        store = pd.concat([store, _add], ignore_index=True).sort_values("date")
        store.to_csv(NEW_MATCHES, index=False)
    known = {lg: set(g.HomeTeam) | set(g.AwayTeam) for lg, g in store.groupby("league")}
    # Freshness for THESE leagues, not Big 5's. The first version checked only the Big 5
    # slugs, so a board publishing Brasileirao carried a footer describing English data.
    # season_types is None because ESPN puts a league id in that field for soccer.
    am_slug = {"Brasileirao": "soccer/bra.1", "LigaMX": "soccer/mex.1", "MLS": "soccer/usa.1"}
    am_entries, am_worst, am_newest = [], 0, None
    for _lg in sorted(set(store.league) & set(am_slug)):
        _sub = store[store.league == _lg]
        if not len(_sub):
            continue
        _nw = pd.to_datetime(_sub.date).max().date()
        _m, _, _probe_failed = FR.missed_games(am_slug[_lg], _nw, season_types=None)
        am_worst = max(am_worst, abs(_m))
        am_newest = _nw if am_newest is None else max(am_newest, _nw)
        am_entries.append((_lg, _nw, am_slug[_lg]))
    if am_entries:
        # No reporter= here. check_multi registers a FAIL on stale, and finish(strict=True)
        # raises on any FAIL -- which would abort the run AFTER the board was built and
        # BEFORE it posted. One board's source outage must not silence the other, and that
        # is true in both directions. The single registration below records that the guards
        # ran; see the note there.
        _ok, _lines = FR.check_multi(am_entries, season_types=None, grace_days=14)
        for _l in _lines:
            print("  " + _l)
        if not _ok and not FR.override():
            _bad = [e[0] for e, _m in zip(am_entries, _lines)
                    if "STALE" in _m or "UNVERIFIED" in _m]
            _why = ("Americas withheld: " + ", ".join(_bad or ["stale"]) + " not current.")
            print(f"  {_why}")
            return [], [], None, _why
    AM_FRESH = FR.board_line(am_newest, checked=(am_newest, am_worst)) if am_newest else None

    fx = AM.fetch_fixtures(known, alias, HORIZON_DAYS)
    if len(fx):
        fx = fx.assign(
            season=[AM.season_of(r.league, r.date, store) for r in fx.itertuples()],
            FTHG=np.nan, FTAG=np.nan, HS=np.nan, AS=np.nan, HST=np.nan, AST=np.nan)
    print(f"  americas fixtures in {HORIZON_DAYS}d: {len(fx)}")
    f = AM.build_features(store, fx, cfg, E)
    tr = (f.date < pd.Timestamp("2026-01-01")) & (f.date >= pd.Timestamp("2014-01-01")) \
         & f.hg.notna() & (~f.draw)
    sc = StandardScaler().fit(f.loc[tr, AM.FEATS].values)
    mdl = LogisticRegression(C=C_, max_iter=2000).fit(
        sc.transform(f.loc[tr, AM.FEATS].values), f.loc[tr, "y"].values)
    base = float(f.loc[tr, "y"].mean())
    pred_path = os.path.join(LOGS, "americas_predictions.csv")
    log = pd.read_csv(pred_path) if os.path.exists(pred_path) else pd.DataFrame(
        columns=["id", "date", "league", "home", "away", "p_home", "p_elo",
                 "boarded", "model"])
    rows = []
    up = f[f.hg.isna()]
    if len(up):
        p = AM.np.asarray(mdl.predict_proba(sc.transform(up[AM.FEATS].values))[:, 1])
        p = E.shrink(p, up.min_gp.values, a0_)
        pe = 1 / (1 + 10 ** (-((up.h_elo + cfg["HA"]) - up.a_elo) / 400))
        nb = pd.DataFrame({
            "id": up.league + "|" + pd.to_datetime(up.date).dt.strftime("%Y-%m-%d")
                  + "|" + up.home + "|" + up.away,
            "date": pd.to_datetime(up.date).dt.strftime("%Y-%m-%d"),
            "league": up.league.values, "home": up.home.values, "away": up.away.values,
            "p_home": p, "p_elo": pe.values,
            "boarded": [int(l in NEW_BOARD) for l in up.league],
            "model": ["v1" if l in NEW_BOARD else "v1-retired" for l in up.league]})
        log = pd.concat([log[~log.id.isin(nb.id)], nb], ignore_index=True)
        for _, r in nb.sort_values(["date", "league"]).iterrows():
            if not r.boarded:
                continue
            pick = r.home if r.p_home >= .5 else r.away
            rows.append({"date": r.date, "id": r.id, "p": max(r.p_home, 1 - r.p_home),
                         # League prefix, because the Big 5 rows carry no league and a
                         # channel showing both needs to say which is which.
                         "game": f"{r.home} vs {r.away}",
                         "pick": f"{pick} {MARKET_TAG}"})
        n_ret = int((~nb.boarded.astype(bool)).sum())
        print(f"  americas board: {len(rows)} pushed, {n_ret} logged only (retired league)")

    # ---- grade ----
    res = store.assign(id=store.league + "|"
                       + pd.to_datetime(store.date).dt.strftime("%Y-%m-%d")
                       + "|" + store.HomeTeam + "|" + store.AwayTeam)
    got = res.set_index("id")[["FTHG", "FTAG"]]
    got = got[~got.index.duplicated(keep="last")]
    j = log.id.map(got.FTHG), log.id.map(got.FTAG)
    log["result"] = np.where(j[0].isna(), np.nan,
                             np.where(j[0] == j[1], -1.0, (j[0] > j[1]).astype(float)))
    os.makedirs(LOGS, exist_ok=True)
    log.to_csv(pred_path, index=False)
    alerts = []
    dec = log[log.result.isin([0.0, 1.0])]
    if len(dec) >= 50:
        print(f"\n  americas rolling grade, decided only "
              f"(pushes: {(log.result == -1).sum():.0f}):")
        for lg in sorted(dec.league.unique()):
            d = dec[dec.league == lg]
            if len(d) < 25:
                continue
            y = d.result.astype(int).values
            pc = np.clip(d.p_home.values.astype(float), 1e-6, 1 - 1e-6)
            pf = np.clip(d.p_elo.values.astype(float), 1e-6, 1 - 1e-6)
            tag = "" if lg in NEW_BOARD else "  [RETIRED, not boarded]"
            print(f"    {lg:12} n={len(y):>4}  LL={log_loss(y, pc):.4f}  "
                  f"elo={log_loss(y, pf):.4f}  edge={log_loss(y, pf) - log_loss(y, pc):+.4f}  "
                  f"hit={np.mean((pc > .5) == (y == 1)):.3f}{tag}")
        w = dec[dec.league.isin(NEW_BOARD)].tail(300)
        if len(w) >= 300:
            yy = w.result.astype(int).values
            edge = log_loss(yy, np.full(len(yy), base)) - \
                log_loss(yy, np.clip(w.p_home.values.astype(float), 1e-6, 1 - 1e-6))
            print(f"    last-300 boarded edge vs base: {edge:+.4f}  "
                  f"[{'OK' if edge > 0 else 'TRIPWIRE: bench review'}]")
            if edge <= 0:
                alerts.append("Americas DNB")
        dec.to_csv(os.path.join(LOGS, "americas_graded.csv"), index=False)
    return rows, alerts, AM_FRESH, None


def run():
    chk = CHK.Reporter("Soccer", dt.date.today().strftime("%Y%m%d"))
    m = refresh_results()
    fx, _fx_ok = fetch_fixtures()
    print(f"Big 5 fixtures in {HORIZON_DAYS}d: {len(fx)}")
    # WARN not FAIL: a fixture-feed outage means no Big 5 board this run, but it must not
    # kill the Americas board or the grading, both of which are independent of this feed.
    chk.no_silent_empty("big5_fixtures", is_empty=(len(fx) == 0),
                        call_failed=not _fx_ok, warn_only=True)
    pseudo = []
    for r in fx.itertuples():
        pseudo.append(dict(Date=r.date.strftime("%d/%m/%Y"), HomeTeam=r.HomeTeam,
                           AwayTeam=r.AwayTeam, FTHG=np.nan, FTAG=np.nan, HS=np.nan,
                           AS=np.nan, HST=np.nan, AST=np.nan, league=r.league,
                           season=season_code(r.date.date()), date=r.date))
    g_all = pd.concat([m, pd.DataFrame(pseudo)], ignore_index=True) if pseudo else m
    for c in ("HS", "AS", "HST", "AST"):
        g_all[c] = pd.to_numeric(g_all.get(c), errors="coerce")
    g_all = g_all.sort_values("date").reset_index(drop=True)
    for s in g_all.season.unique():
        extend_seasons(str(s).zfill(4))
    # Big 5 leagues, each probed separately: a board that publishes five competitions is
    # only as current as its worst one. season_types is None because ESPN puts a league id
    # in that field for soccer, not the 1/2/3 US convention, so the usual filter would
    # reject every match.
    _slug = {"EPL": "soccer/eng.1", "LaLiga": "soccer/esp.1", "SerieA": "soccer/ita.1",
             "Bundesliga": "soccer/ger.1", "Ligue1": "soccer/fra.1"}
    _played = g_all[g_all.hg.notna()] if "hg" in g_all.columns else g_all[g_all.FTHG.notna()]
    _entries, _worst = [], 0
    for _lg, _sl in _slug.items():
        _sub = _played[_played.league == _lg] if "league" in _played.columns else _played
        if not len(_sub):
            continue
        _nw = pd.to_datetime(_sub.date).max().date()
        _m, _, _probe_failed = FR.missed_games(_sl, _nw, season_types=None)
        _worst = max(_worst, abs(_m))
        _entries.append((_lg, _nw, _sl))
    # Stale Big 5 suppresses the BIG 5 BOARD, it does not kill the run. run_americas is
    # already wrapped in try/except so a broken Americas board cannot take Big 5 down; this
    # is the same protection in the other direction, and it was missing.
    #
    # It matters right now rather than hypothetically: football-data has published no
    # 2026-27 Big 5 file for any of the five leagues, so this check will fail on every run
    # until they do, possibly for weeks. Raising here would take Brasileirao and Liga MX
    # off Discord the whole time, and those two are healthy and had 14 boardable fixtures
    # on the last good run.
    #
    # The guard still does its real job: no Big 5 price is published on stale data. It just
    # stops a sourceless board holding a working one hostage.
    BIG5_OK, BIG5_WHY = True, None
    if _entries:
        _ok, _lines = FR.check_multi(_entries, season_types=None, grace_days=14)
        for _l in _lines:
            print(_l)
        if not _ok and not FR.override():
            BIG5_OK = False
            _bad = [e[0] for e, m in zip(_entries, _lines) if "STALE" in m or "UNVERIFIED" in m]
            BIG5_WHY = ("Big 5 withheld: " + ", ".join(_bad or ["stale"])
                        + " not current. Americas below is unaffected.")
            print(f"  {BIG5_WHY}")
        _newest = max(e[1] for e in _entries)
        FRESH_LINE = FR.board_line(_newest, checked=(_newest, _worst))
    else:
        FRESH_LINE = None
    SEASON_LINE = None

    f = E.add_diffs(E.build(g_all, CFG))
    tr = f.season.apply(lambda s: E.SEASONS.index(str(s).zfill(4)) <= E.SEASONS.index(TRAIN_THROUGH)) \
        & f.hg.notna() & (~f.draw)
    sc = StandardScaler().fit(f.loc[tr, E.FEATS].values)
    mdl = LogisticRegression(C=C, max_iter=2000).fit(
        sc.transform(f.loc[tr, E.FEATS].values), f.loc[tr, "y"].values)
    base = float(f.loc[tr, "y"].mean())
    alerts = []
    os.makedirs(LOGS, exist_ok=True)
    pred_path = os.path.join(LOGS, "predictions.csv")
    log = pd.read_csv(pred_path) if os.path.exists(pred_path) else \
        pd.DataFrame(columns=["id", "date", "league", "home", "away", "p_home", "p_elo", "model"])
    if pseudo and BIG5_OK:
        rows = f[f.hg.isna()].copy()
        p = mdl.predict_proba(sc.transform(rows[E.FEATS].values))[:, 1]
        p = E.shrink(p, rows.min_gp.values, A0)
        # engine.shrink ramps to full weight at 6 games; say so while it is still ramping.
        SEASON_LINE = FR.season_note(float(rows.min_gp.min()), 6, "the thinner side")
        pe = 1/(1+10**(-((rows.h_elo + CFG["HA"]) - rows.a_elo)/400))
        b = pd.DataFrame({"id": rows.league + "|" + pd.to_datetime(rows.date).dt.strftime("%Y-%m-%d")
                                + "|" + rows.home + "|" + rows.away,
                          "date": pd.to_datetime(rows.date).dt.strftime("%Y-%m-%d"),
                          "league": rows.league, "home": rows.home, "away": rows.away,
                          "p_home": p, "p_elo": pe.values, "model": "v1"})
        log = pd.concat([log[~log.id.isin(b.id)], b], ignore_index=True)
    elif pseudo:
        print("Big 5 fixtures present but the board is suppressed on stale data")
    else:
        print("no upcoming Big-5 fixtures in horizon")
    res = m.assign(id=m.league + "|" + m.date.dt.strftime("%Y-%m-%d") + "|" + m.HomeTeam + "|" + m.AwayTeam)
    exact = res.set_index("id")[["FTHG", "FTAG"]]

    def grade(row):
        if row.id in exact.index:
            hg, ag = exact.loc[row.id]
        else:
            c = res[(res.league == row.league) & (res.HomeTeam == row.home) & (res.AwayTeam == row.away)]
            c = c[(c.date - pd.Timestamp(row.date)).abs() <= pd.Timedelta(days=14)]
            if c.empty:
                return np.nan
            hg, ag = c.iloc[0][["FTHG", "FTAG"]]
        return -1.0 if hg == ag else float(hg > ag)

    log["result"] = log.apply(grade, axis=1)
    log.to_csv(pred_path, index=False)
    dec = log[log.result.isin([0.0, 1.0])]
    if len(dec) >= 30:
        y = dec.result.astype(int).values
        print(f"\nrolling grade, decided only (pushes: {(log.result == -1).sum():.0f}):")
        for lbl, p in (("model v1 ", dec.p_home), ("elo floor", dec.p_elo),
                       ("base rate", pd.Series(np.full(len(y), base)))):
            pc = np.clip(p.values.astype(float), 1e-6, 1-1e-6)
            print(f"{lbl}: n={len(y)}  LL={log_loss(y, pc):.4f}  Brier={brier_score_loss(y, pc):.4f}  "
                  f"hit={np.mean((pc > .5) == (y == 1)):.3f}")
        w = dec.tail(300)
        if len(w) >= 300:
            yy = w.result.astype(int).values
            edge = log_loss(yy, np.full(len(yy), base)) - \
                log_loss(yy, np.clip(w.p_home.values.astype(float), 1e-6, 1-1e-6))
            print(f"last-300 edge vs base: {edge:+.4f}  [{'OK' if edge > 0 else 'TRIPWIRE: bench review'}]")
            if edge <= 0:
                alerts.append("Soccer DNB")
        dec.to_csv(os.path.join(LOGS, "graded.csv"), index=False)

    # ---------------------------------------------------------------- Americas
    new_rows, new_alerts, AM_FRESH, AM_WHY = [], [], None, None
    if NEW:
        try:
            new_rows, new_alerts, AM_FRESH, AM_WHY = run_americas(chk)
        except Exception as ex:
            # Never let a second board take the first one down with it.
            print(f"americas board failed: {type(ex).__name__}: {ex}")
            AM_WHY = f"Americas withheld: the board errored ({type(ex).__name__})."

    # One registration for both boards, after both have been evaluated. register_guard is
    # what require("freshness") trusts, and both guards genuinely ran -- the difference is
    # that a stale verdict here suppresses that board rather than aborting the run, so it
    # is not a check FAILURE. The invariant this asserts is the one that matters: no stale
    # price reached Discord. A withheld board is that invariant HOLDING, and the reason is
    # posted to the channel below rather than buried.
    _held = [x for x in (BIG5_WHY, AM_WHY) if x]
    chk.register_guard("freshness", True,
                       "; ".join(_held) if _held else "all boards current, nothing withheld",
                       "unverified" if _held else "fresh")
    alerts = alerts + new_alerts

    # Both boards go through one width pass and one delta gate, so the two share a column
    # layout and a single push cache rather than drifting apart.
    entries = []
    if pseudo and BIG5_OK:
        entries += [(r.date, f"{r.home} vs {r.away}",
                     f"{r.home if r.p_home >= .5 else r.away} {MARKET_TAG}",
                     max(r.p_home, 1 - r.p_home), r.id) for _, r in b.iterrows()]
    entries += [(r["date"], r["game"], r["pick"], r["p"], r["id"]) for r in new_rows]

    push, skipped = {}, 0
    if entries:
        wg = max([len(x[1]) for x in entries] + [10])
        wp = max([len(x[2]) for x in entries] + [8])
        for d_, gm, pk, q, gid in entries:
            if BF.min_odds(q) < BF.PRICE_FLOOR:
                skipped += 1
                continue
            push[gid] = {"line": BF.line(d_, gm, pk, q, w_game=wg, w_pick=wp),
                         "min_odds": BF.min_odds(q),
                         # Carries the pick, so a side flipping between runs is reported
                         # even when the price barely moves. The id already pins the
                         # fixture, so the matchup itself cannot change under it.
                         "match": pk}
    print("\n=== MBM Soccer DNB board ===")
    for _v in push.values():
        print(_v["line"])
    if skipped:
        print(f"({skipped} fixture{'s' if skipped != 1 else ''} skipped, "
              f"price floor {BF.PRICE_FLOOR:.2f} / {BF.american(BF.PRICE_FLOOR)})")
    # Keyed, so whatever survives the gate can be regrouped under its league. On a delta
    # push only some fixtures go out, so headings must be built from what was actually
    # sent rather than from the full board.
    chk.finish(strict=True, require=["freshness"])
    mode, keyed = tg_delta.gate(LOGS, dt.date.today().strftime("%Y%m%d"), push,
                                with_keys=True)
    by_lg = {}
    for _k, _ln in keyed:
        by_lg.setdefault(str(_k).split("|")[0], []).append(_ln)
    lines = []
    for _lg in LG_ORDER + [x for x in by_lg if x not in LG_ORDER]:
        if _lg not in by_lg:
            continue
        if lines:
            lines.append("")
        lines.append(f"__{LG_NAME.get(_lg, _lg)}__")
        lines.extend(by_lg[_lg])
    if lines or alerts or BIG5_WHY or AM_WHY:
        body = list(lines) if lines else ["no price changes since the last push"]
        foot = (f"{skipped} fixture{'s' if skipped != 1 else ''} skipped, "
                f"price floor {BF.PRICE_FLOOR:.2f} / {BF.american(BF.PRICE_FLOOR)}")\
               if skipped else None
        title = "MBM Soccer DNB" + (" · changes since the last push"
                                    if mode == "delta" and lines else "")
        colour = notify.GREEN
        if alerts:
            body = body + ["", "TRIPWIRE: " + ", ".join(alerts)
                           + " below base rate, stop betting this board"]
            colour = notify.RED
        # The board carries its own provenance. A reader should not have to trust that
        # the data was current; the row says so.
        # Each board's own provenance. A run that publishes only the Americas must not
        # carry a Big 5 freshness line, and vice versa.
        # Posted, not just logged. A board that silently stops looks exactly like a quiet
        # week, and Big 5 will be down for as long as its source has no 2026-27 files.
        for _w in (BIG5_WHY, AM_WHY):
            if _w:
                body = body + ["", _w]
        _fresh = []
        if pseudo and BIG5_OK and FRESH_LINE:
            _fresh.append("Big 5 " + FRESH_LINE)
        if new_rows and AM_FRESH:
            _fresh.append("Americas " + AM_FRESH)
        foot = " · ".join([x for x in (" · ".join(_fresh), SEASON_LINE, foot) if x])
        notify.send(title, body, foot, colour)
    elif not push:
        print("no board to send")
    else:
        print("board unchanged since the last push, nothing sent")


if __name__ == "__main__":
    run()
