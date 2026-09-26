import json
import numpy as np
import matplotlib.pyplot as plt

THETA = 1.0
NS = [2000, 5000, 10000, 20000, 50000]
ORANGE, BLUE, INK = "#EB6834", "#2A78D6", "#14213D"

PANELS = [
    ("corr ≈ 0.87 (deck)", "../../out/mc_conv.json"),
    ("corr ≈ 0.35 (moderate)", "out/mc_corr35.json"),
    ("corr ≈ 0.15 (weak)", "out/mc_corr15.json"),
]


def load(path):
    raw = json.load(open(path))
    return {int(k): np.array(v) for k, v in raw.items() if int(k) in NS}


def rate(ns, bias):
    ns, bias = np.asarray(ns, float), np.abs(np.asarray(bias, float))
    ok = bias > 1e-6
    return np.polyfit(np.log(ns[ok]), np.log(bias[ok]), 1)[0]


fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
print(f"{'panel':<22}{'LightGBM rate':>16}{'DML rate':>12}")
for ax, (title, path) in zip(axes, PANELS):
    data = load(path)
    sm = np.array([np.median(data[n][:, 0]) for n in NS])
    dmn = np.array([np.median(data[n][:, 1]) for n in NS])
    slo, shi = np.array([np.percentile(data[n][:, 0], [10, 90]) for n in NS]).T
    dlo, dhi = np.array([np.percentile(data[n][:, 1], [10, 90]) for n in NS]).T

    ax.set_xscale("log")
    ax.axhline(THETA, color=INK, lw=1.4, ls=(0, (5, 2.5)))
    ax.fill_between(NS, slo, shi, color=ORANGE, alpha=0.15, lw=0)
    ax.plot(NS, sm, color=ORANGE, marker="o", lw=2, label="LightGBM")
    ax.fill_between(NS, dlo, dhi, color=BLUE, alpha=0.15, lw=0)
    ax.plot(NS, dmn, color=BLUE, marker="o", lw=2, label="Double ML")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Portfolio size")
    ax.set_xticks(NS)
    ax.set_xticklabels([f"{n // 1000}k" for n in NS])
    ax.minorticks_off()

    r_s, r_d = rate(NS, sm - THETA), rate(NS, dmn - THETA)
    print(f"{title:<22}{r_s:>16.2f}{r_d:>12.2f}")

axes[0].set_ylabel("Estimated price effect")
axes[0].legend(loc="lower right", fontsize=10, frameon=False)
axes[0].set_ylim(0, 1.6)
fig.tight_layout()
fig.savefig("out/bias_convergence_compare.png", dpi=150)
print("saved out/bias_convergence_compare.png")
