# MBM Big-5 soccer DNB daily engine (rebuild v1)

Standalone predictive model, no market inputs. Target: P(home wins | no draw); draws
(~25%) grade as pushes. Frozen through 2025-26; 2026-27 is the forward test.
Frozen 2025-26 holdout: LL 0.5753 / Brier 0.1964 / hit 69.8% (base 0.6773, Elo 0.5895).

Data: football-data.co.uk (results + fixtures, same team names, no key). Runner idles
through preseason and wakes when fixtures post (typically the week of matchdays).
Promoted teams are handled automatically (Elo init -200, promoted flag, state shrinkage).

## Setup: private GitHub repo, upload folder, enable Actions. Optional TG_TOKEN/TG_CHAT.
## Manual: pip install -r requirements.txt && python daily_soccer.py
## Tripwire: last-300 decided-game log-loss edge vs base rate printed every run.
