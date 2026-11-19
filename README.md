# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/cendelobening/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/cendelobening/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one: for every PlayStation 5 trophy, five quantiles of its Exophase earn rate at 2026-11-26T00:00Z, from the case history of Exophase reads.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Model

Pure-stdlib Python (`main.py`, no packages, no network). Per trophy, the logit earn rate is fitted as a straight line in `log(1 + hours since 2026-11-19T00:00Z)` over the case history, weighted by `sqrt(players tracked)` — a rate measured over a large tracked base is far more precise, and the fit is effectively in earned-count space (`rate * tracked / 100`), so dilution of rates by a growing tracked base and progression of earned counts are both carried by the data. The fitted slope is shrunk toward a tier prior (story bronze rises in week one, platinum falls as dilution wins), the median is extrapolated to the target with a cap on the total logit move, and a saturated trophy (100%) is drifted down only. The spread is horizon-scaled in logit space: a random-walk floor of `0.65 * sqrt(horizon in days)`, widened by the fit's residual uncertainty and extrapolation leverage and by rarity, floored at 0.5 logit units — much wider than the house baseline's band, because a launch-week refresh can move every rate at once.

---

<sub>Entered on [make.fun](https://make.fun/?ref=jme44qjv&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
