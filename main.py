"""GTA VI trophy rarity forecast — count-space logit-trend model (pure stdlib).

Forecasts each trophy's Exophase earn rate at the case target from the case
history. The key structure of the data, from the competition rules:

* Exophase refreshes on its own schedule; a read shows rates as of its last
  refresh, and rates are earned / tracked. As tracked players grow, new
  joiners with zero trophies dilute every rate down, while progression pushes
  earned counts up. So the model works in COUNT space: per-trophy earned
  counts are rate * tracked / 100, and the rate is the ratio of two growing
  quantities.
* A trophy's logit earn rate is therefore modelled as a straight line in
  x = log(1 + hours since 2026-11-19T00:00Z) — the adoption/progression clock
  the house curve baseline also uses — fitted per trophy by least squares
  over the case history, with the slope shrunk toward a tier prior (story
  bronze trophies rise in week one, platinum/collectible/grind trophies fall
  or stay flat as the tracked base dilutes them).
* The spread is an honest, horizon-scaled logit spread: a random-walk floor
  of 0.65 * sqrt(horizon in days), widened by the fit's residual uncertainty
  (extrapolation leverage to the target) and by rarity, with an absolute
  floor of 0.5 logit units — far wider than the house baseline's narrow band,
  because a launch-week refresh can move every rate at once.

Deterministic: no randomness, no network, no clock reads; two runs on the
same input write byte-identical output. Pure standard library only.

Usage (local): mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
Grader: python3 -P -s /bot/main.py, reading $MAKEFUN_INPUT_DIR (else /input)
and writing $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Adoption/progression clock: hours since the launch-night epoch.
EPOCH = "2026-11-19T00:00:00Z"

# Tier prior for d(logit rate) / d(log(1 + hours)): story bronze rises as
# players progress through week one; platinum is earned by almost nobody in
# week one so dilution wins; gold/silver sit in between.
TIER_PRIOR_SLOPE = {
    "platinum": -0.35,
    "gold": -0.10,
    "silver": 0.0,
    "bronze": 0.12,
}
# Fallback median rate (percent) per tier when a trophy has no usable history.
TIER_FALLBACK_RATE = {
    "platinum": 0.5,
    "gold": 1.0,
    "silver": 2.0,
    "bronze": 1.0,
}
# How strongly the per-trophy fitted slope is shrunk toward the tier prior:
# weight of the prior is equivalent to PRIOR_WEIGHT extra reads.
PRIOR_WEIGHT = 8.0
# Random-walk spread per sqrt(day) of horizon — wider than the house 0.5.
SIGMA_RW = 0.65
SIGMA_FLOOR = 0.5
SIGMA_CAP = 2.5
# Cap on how far the median may move from the last read, in logit units.
MAX_MEDIAN_MOVE = 3.0
# A trophy already at 100% in the last read can only move one way: new
# tracked players who have not earned it yet dilute it down. Give it a
# negative drift instead of its tier's prior slope.
SATURATED_RATE = 99.9
SATURATED_SLOPE = -0.5


def clamp_rate(pct: float) -> float:
    return min(99.99, max(0.01, pct))


def logit(pct: float) -> float:
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up (matches the starter's rounding)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(text: str) -> float:
    """ISO 8601 UTC timestamp -> epoch seconds (stdlib, no datetime tz pain)."""
    # Format is always like 2026-11-19T03:00:00.309Z or ...:00Z.
    date_part, _, time_part = text.partition("T")
    year, month, day = (int(v) for v in date_part.split("-"))
    time_part = time_part.rstrip("Z")
    if "." in time_part:
        time_part, frac = time_part.split(".", 1)
        frac_s = float("0." + frac)
    else:
        frac_s = 0.0
    hour, minute, second = (int(v) for v in time_part.split(":"))
    days = _days_since_epoch(year, month, day)
    return days * 86400.0 + hour * 3600.0 + minute * 60.0 + second + frac_s


def _days_since_epoch(year: int, month: int, day: int) -> int:
    """Days from 1970-01-01 to the given civil date (Howard Hinnant's algorithm)."""
    y = year - (1 if month <= 2 else 0)
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    mp = (month + 9) % 12
    doy = (153 * mp + 2) // 5 + day - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


EPOCH_SECONDS = parse_time(EPOCH)


def hours_since_epoch(at: str) -> float:
    return (parse_time(at) - EPOCH_SECONDS) / 3600.0


def fit_line(xs, ys, ws=None):
    """Weighted least squares y = a + b*x. Returns (a, b, sxx, x_mean, n).

    Weights default to 1. A read's weight is sqrt(players tracked): a rate
    measured over 3,868 players is far more precise than one over 638, and
    the earned COUNT behind a rate is rate * tracked / 100, so the fit is
    effectively a count-space fit — dilution of the rate by a growing
    tracked base is carried by the rates themselves, while the weights keep
    the small-base launch-night reads from dominating the line.
    """
    n = len(xs)
    if ws is None:
        ws = [1.0] * n
    w_sum = sum(ws)
    x_mean = sum(w * x for w, x in zip(ws, xs)) / w_sum
    y_mean = sum(w * y for w, y in zip(ws, ys)) / w_sum
    sxx = sum(w * (x - x_mean) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0:
        return y_mean, 0.0, 0.0, x_mean, n
    sxy = sum(w * (x - x_mean) * (y - y_mean) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    return y_mean - b * x_mean, b, sxx, x_mean, n


def forecast_trophy(trophy, history, horizon_minutes, target_at):
    """Return the five quantiles for one trophy."""
    tier = trophy.get("tier", "bronze")
    prior_slope = TIER_PRIOR_SLOPE.get(tier, 0.0)

    xs, ys, ws = [], [], []
    last_rate = None
    last_x = None
    idx = trophy["_index"]
    for read in history:
        rates = read.get("rates") or []
        if idx >= len(rates):
            continue
        rate = rates[idx]
        if rate is None:
            continue
        try:
            rate = float(rate)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(rate):
            continue
        x = math.log1p(max(0.0, hours_since_epoch(read["at"])))
        xs.append(x)
        ys.append(logit(rate))
        # Precision weight from the tracked base behind this read; the
        # earned count is rate * tracked / 100, so a bigger base means a
        # better-measured rate.
        tracked = read.get("players_tracked")
        try:
            tracked = float(tracked)
        except (TypeError, ValueError):
            tracked = 0.0
        ws.append(math.sqrt(tracked) if tracked and tracked > 0 else 1.0)
        last_rate = rate
        last_x = x

    if last_rate is None:
        last_rate = TIER_FALLBACK_RATE.get(tier, 1.0)
        last_logit = logit(last_rate)
        median_logit = last_logit
        sigma = SIGMA_FLOOR
    else:
        last_logit = logit(last_rate)
        x_target = math.log1p(max(0.0, hours_since_epoch(target_at)))
        if len(xs) >= 3:
            a, b_fit, sxx, x_mean, n = fit_line(xs, ys, ws)
            # Residual spread of the fit (logit units), weighted.
            if n > 2:
                w_sum = sum(ws)
                resid = math.sqrt(
                    sum(w * (y - (a + b_fit * x)) ** 2 for w, x, y in zip(ws, xs, ys))
                    / (w_sum * (n - 2) / n)
                )
            else:
                resid = 0.0
            # Shrink the fitted slope toward the tier prior.
            b = (n * b_fit + PRIOR_WEIGHT * prior_slope) / (n + PRIOR_WEIGHT)
            median_logit = a + b * x_target
            # Extrapolation leverage: how far the target sits outside the
            # observed x-range, in units of the fit's own scatter.
            if sxx > 0:
                leverage = (x_target - x_mean) ** 2 / sxx
            else:
                leverage = 0.0
            fit_unc = resid * math.sqrt(1.0 + 1.0 / n + leverage)
            fit_unc = min(fit_unc, 1.5)
        elif len(xs) >= 1:
            # Too few reads for a fit: drift from the last rate at the tier
            # prior slope, over the remaining clock to the target.
            resid = 0.5
            b = prior_slope
            median_logit = last_logit + b * max(0.0, x_target - last_x)
            fit_unc = 0.5
        else:
            median_logit = last_logit
            fit_unc = 0.5
            resid = 0.5

        # Never let the median run away from the last observed rate: a
        # 4.5-day launch-week move is large, but a 20x move is not credible.
        move = median_logit - last_logit
        if move > MAX_MEDIAN_MOVE:
            median_logit = last_logit + MAX_MEDIAN_MOVE
        elif move < -MAX_MEDIAN_MOVE:
            median_logit = last_logit - MAX_MEDIAN_MOVE

        # Saturated trophy: every new tracked player who has not earned it
        # yet dilutes the rate down, so override the tier drift with a
        # negative drift over the remaining clock to the target.
        if last_rate >= SATURATED_RATE:
            median_logit = last_logit + SATURATED_SLOPE * max(
                0.0, x_target - last_x
            )

        # Horizon-scaled spread: random-walk floor, plus fit uncertainty,
        # plus a rarity term (rare trophies move most in logit space).
        horizon_days = horizon_minutes / 1440.0
        rw = SIGMA_RW * math.sqrt(max(0.0, horizon_days))
        if last_rate < 0.5:
            rare_extra = 0.35
        elif last_rate < 2.0:
            rare_extra = 0.25
        elif last_rate < 5.0:
            rare_extra = 0.15
        else:
            rare_extra = 0.05
        sigma = math.sqrt(rw * rw + fit_unc * fit_unc + rare_extra * rare_extra)
        sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)

    qs = []
    for z in Z:
        q = clamp_rate(inv_logit(median_logit + z * sigma))
        qs.append(round4(q))
    # Enforce non-decreasing after rounding (rounding can only tie, never
    # invert, but keep the invariant explicit).
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    horizon_minutes = float(case["horizon_minutes"])
    target_at = case["target"]
    trophies = case["trophies"]
    for i, trophy in enumerate(trophies):
        trophy["_index"] = i
    rows = []
    for trophy in trophies:
        rows.append({"id": trophy["id"], "q": forecast_trophy(trophy, history, horizon_minutes, target_at)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")
        os.chmod(out_path, 0o644)


if __name__ == "__main__":
    main()
