import json
import numpy as np
import matplotlib.pyplot as plt

THETA = 1.0
NS = [100, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000]
ORANGE, BLUE, INK, ZONE = "#EB6834", "#2A78D6", "#14213D", "#F0F0EA"
Y0, Y1 = -0.3, 1.6   # same clipping window as the deck's own why_convergence chart

PANELS = [
    ("corr ≈ 0.87 (deck)", "out/mc_deck87.json"),
    ("corr ≈ 0.35 (moderate)", "out/mc_corr35.json"),
    ("corr ≈ 0.15 (weak)", "out/mc_corr15.json"),
]


def load(path):
    raw = json.load(open(path))
    return {int(k): np.array(v) for k, v in raw.items() if int(k) in NS}


def rate(ns, bias, n_min=1000):
    """log-log slope of |median bias| vs n, restricted to n >= n_min (below that, MC noise
    dominates and the slope is meaningless - see out/mc_*.json n=100/250 rows)."""
    ns, bias = np.asarray(ns, float), np.abs(np.asarray(bias, float))
    ok = (bias > 1e-6) & (ns >= n_min)
    return np.polyfit(np.log(ns[ok]), np.log(bias[ok]), 1)[0]


def line(ax, ns, vals, col, z):
    inside = (vals >= Y0) & (vals <= Y1)
    ax.plot(ns, np.clip(vals, Y0, Y1), color=col, lw=2, zorder=z, solid_capstyle="round")
    ax.scatter(np.asarray(ns)[inside], vals[inside], s=36, color=col, edgecolor="white", lw=1.2, zorder=z + 0.5)
    off = ~inside
    if off.any():
        edge = np.where(vals[off] < Y0, Y0 + 0.045, Y1 - 0.045)
        marker = np.where(vals[off] < Y0, "v", "^")
        for x, y, mk in zip(np.asarray(ns)[off], edge, marker):
            ax.scatter([x], [y], marker=mk, s=70, color=col, lw=0, zorder=z + 0.5, clip_on=False)


fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
print(f"{'panel':<22}{'LightGBM rate (n>=1k)':>24}{'DML rate (n>=1k)':>20}")
for ax, (title, path) in zip(axes, PANELS):
    data = load(path)
    sm = np.array([np.median(data[n][:, 0]) for n in NS])
    dmn = np.array([np.median(data[n][:, 1]) for n in NS])
    slo, shi = np.array([np.clip(np.percentile(data[n][:, 0], [10, 90]), Y0, Y1) for n in NS]).T
    dlo, dhi = np.array([np.clip(np.percentile(data[n][:, 1], [10, 90]), Y0, Y1) for n in NS]).T

    ax.set_xscale("log")
    ax.set_ylim(Y0, Y1)
    ax.axvspan(NS[0] / 1.3, 300, color=ZONE, lw=0, zorder=0)
    ax.text(np.sqrt(NS[0] / 1.3 * 300), Y1 - 0.06, "Too little\ndata", ha="center", va="top",
            fontsize=10, color="#8A8A82", linespacing=1.15)
    ax.axhline(THETA, color=INK, lw=1.4, ls=(0, (5, 2.5)), zorder=6)
    ax.fill_between(NS, slo, shi, color=ORANGE, alpha=0.15, lw=0)
    line(ax, NS, sm, ORANGE, 5)
    ax.fill_between(NS, dlo, dhi, color=BLUE, alpha=0.15, lw=0)
    line(ax, NS, dmn, BLUE, 6)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Portfolio size")
    ax.set_xticks(NS)
    ax.set_xticklabels([f"{n // 1000}k" if n >= 1000 else str(n) for n in NS], rotation=45, ha="right")
    ax.minorticks_off()

    r_s, r_d = rate(NS, sm - THETA), rate(NS, dmn - THETA)
    print(f"{title:<22}{r_s:>24.2f}{r_d:>20.2f}")

axes[0].set_ylabel("Estimated price effect")
axes[0].plot([], [], color=ORANGE, marker="o", lw=2, label="LightGBM")
axes[0].plot([], [], color=BLUE, marker="o", lw=2, label="Double ML")
axes[0].legend(loc="lower right", fontsize=10, frameon=False)
fig.tight_layout()
fig.savefig("out/bias_convergence_compare.png", dpi=150)
print("saved out/bias_convergence_compare.png")
