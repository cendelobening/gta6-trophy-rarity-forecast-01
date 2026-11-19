"""Local scorer: compare our forecast against the house persistence baseline.

The practice cases cannot validate the final model (their histories are flat
pre-refresh and their answers are the post-refresh page, which is not in the
kit), so this script checks what the practice cases CAN check: that our
spread is much wider than the house baseline's 90-minute band, that our
medians stay sane against the last read, and that the outputs are valid.

Usage: python3 tools/scorer.py <kit-input-dir> <our-output-dir>
"""

import json
import math
import sys
from pathlib import Path

Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
SIGMA = 0.5


def clamp(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def house_row(rate, horizon_minutes):
    spread = SIGMA * math.sqrt(horizon_minutes / 1440.0)
    lg = logit(rate)
    return [100.0 / (1.0 + math.exp(-(lg + z * spread))) for z in Z]


def main():
    in_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    for case_path in sorted(in_dir.glob("*.in")):
        case = json.loads(case_path.read_text(encoding="utf-8"))
        ours = json.loads((out_dir / f"{case_path.stem}.out").read_text(encoding="utf-8"))
        last = case["history"][-1]
        rows = {r["id"]: r["q"] for r in ours["trophies"]}
        assert len(rows) == len(case["trophies"]) == 55, "row count"
        widths_ours, widths_house, med_moves = [], [], []
        for i, trophy in enumerate(case["trophies"]):
            rate = last["rates"][i]
            q = rows[trophy["id"]]
            assert len(q) == 5 and all(q[j] <= q[j + 1] for j in range(4)), "shape"
            assert all(0 <= v <= 100 for v in q), "range"
            h = house_row(rate, case["horizon_minutes"])
            widths_ours.append(logit(q[4]) - logit(q[0]))
            widths_house.append(logit(h[4]) - logit(h[0]))
            med_moves.append(abs(logit(q[2]) - logit(rate)))
        import statistics

        print(f"{case['id']}: n=55 valid")
        print(f"  median 90% logit width ours={statistics.median(widths_ours):.3f} "
              f"house={statistics.median(widths_house):.3f} "
              f"(ratio {statistics.median(widths_ours)/statistics.median(widths_house):.1f}x wider)")
        print(f"  median |median - last rate| in logit: {statistics.median(med_moves):.3f} "
              f"(max {max(med_moves):.3f})")
        # Show the platinum, the saturated 100% trophy, and a typical bronze.
        for tid in (1, 2, 4):
            i = tid - 1
            print(f"  trophy {tid} ({case['trophies'][i]['tier']}, last {last['rates'][i]}%): "
                  f"ours={rows[tid]} house={[round(v, 4) for v in house_row(last['rates'][i], case['horizon_minutes'])]}")


if __name__ == "__main__":
    main()
