import json, time, sys
import numpy as np
from dgp import draw, params_for_corr, DECK_SLOPE, DECK_SIG
from methods import slearner, dml

# Quick pass: fewer reps and a lower size cap than the deck's mc_conv.py (R=30, up to 200k).
# NS starts small (100) so the DML convergence curve is visible at small-portfolio sizes too.
LEVELS = {"deck87": 0.87, "corr35": 0.35, "corr15": 0.15}
NS = [100, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000]
R = 10

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else None
    levels = {tag: LEVELS[tag]} if tag else LEVELS
    for name, target_corr in levels.items():
        slope, sig_v = (DECK_SLOPE, DECK_SIG) if target_corr == 0.87 else params_for_corr(target_corr)
        print(f"=== {name}: target corr {target_corr}, slope {slope:.6f}, sig_v {sig_v:.4f} ===", flush=True)
        out = {}
        t0 = time.time()
        for n in NS:
            rows = []
            for r in range(R):
                A, T, Y, p = draw(n, 5000 + r, slope, sig_v)
                s = slearner(A, T, Y)
                th, se = dml(A, T, Y)
                rows.append([s, th, se])
            out[n] = rows
            json.dump(out, open(f"out/mc_{name}.json", "w"))
            a = np.array(rows)
            print(n, a[:, 0].mean().round(3), a[:, 0].std().round(3),
                  a[:, 1].mean().round(3), a[:, 1].std().round(3),
                  f"{time.time() - t0:.0f}s", flush=True)
