import requests, time, os, sys
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"}
COMPS = {"EPL": (9, "Premier-League"), "LaLiga": (12, "La-Liga"), "SerieA": (11, "Serie-A"),
         "Bundesliga": (20, "Bundesliga"), "Ligue1": (13, "Ligue-1")}
SEASONS = [f"{y}-{y+1}" for y in range(2019, 2026)]
BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 400
t0 = time.time()
for lg, (cid, name) in COMPS.items():
    for s in SEASONS:
        fn = f"soccer/html/{lg}_{s}.html"
        if os.path.exists(fn) and os.path.getsize(fn) > 50000:
            continue
        if time.time() - t0 > BUDGET:
            print("budget hit"); sys.exit(0)
        url = f"https://fbref.com/en/comps/{cid}/{s}/schedule/{s}-{name}-Scores-and-Fixtures"
        r = requests.get(url, headers=H, timeout=60)
        if r.status_code == 429:
            print("429, backing off 65s"); time.sleep(65)
            r = requests.get(url, headers=H, timeout=60)
        print(lg, s, r.status_code, len(r.text))
        if r.status_code == 200:
            open(fn, "w").write(r.text)
        time.sleep(7)
print("done all")
