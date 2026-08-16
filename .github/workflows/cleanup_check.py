"""Show what the wrong-league fetch put into matches.csv, and optionally remove it.

Run with no arguments to inspect. Run with --clean to delete the current season's rows so
the fixed refresh_results can rebuild them from a verified source.
"""
import sys
import pandas as pd

m = pd.read_csv("data/matches.csv", parse_dates=["date"])
m["season"] = m.season.astype(str).str.zfill(4)
cur = m[m.season == "2627"]

print(f"matches.csv: {len(m):,} rows total, {len(cur)} in season 2627\n")
for lg, g in cur.groupby("league"):
    clubs = sorted(set(g.HomeTeam) | set(g.AwayTeam))
    print(f"{lg}: {len(g)} rows, {g.date.min().date()} to {g.date.max().date()}")
    print(f"   {clubs}\n")

if "--clean" in sys.argv:
    keep = m[m.season != "2627"]
    keep.to_csv("data/matches.csv", index=False)
    print(f"removed {len(cur)} season-2627 rows, {len(keep):,} remain")
    print("run daily_soccer.py to rebuild them from a Div-verified source")
else:
    print("inspect the club lists above. Anything that is not a top-flight club of that")
    print("league came from the wrong-division fetch. Re-run with --clean to remove all")
    print("season-2627 rows; the fixed refresh_results will re-add the legitimate ones.")
