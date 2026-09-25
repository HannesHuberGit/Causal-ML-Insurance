"""Reference mock-up of the proposed chart changes, rendered as build layers.

Every chart is drawn ONCE on a figure with a FIXED axes rectangle (no tight_layout, no bbox_inches="tight"),
then saved once per build layer with only that layer's artists visible (transparent PNG, identical pixel size).
Layer 0 carries the axes; later layers switch the axes decorations off.
Outputs: review2/visual_editor/layers/<chart>_L<k>.png  and  preview/<chart>_step<k>.png (cumulative composite).
Run from the scratchpad root after data_cache.py:  python3 review2/visual_editor/mock_layers.py
"""
import json, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from PIL import Image
sys.path.insert(0, ".")
from cont import g, m, THETA, SIG_V

OUT = "final"; os.makedirs(f"{OUT}/layers", exist_ok=True); os.makedirs(f"{OUT}/preview", exist_ok=True)
for w in ["400", "500", "600"]: fm.fontManager.addfont(f"fonts/plex-{w}.ttf")
INK, BODY, LABEL, GRID, SURF = "#14213D", "#4A5568", "#6B7280", "#E4E4DE", "#FBFBF8"
BLUE, ORANGE, AQUA, VIOLET = "#2A78D6", "#EB6834", "#18A070", "#4A3AA7"   # AQUA: 3:1 step of #1BAF7A
DATA, OBS = "#9AA3AF", "#ECECE5"            # grey data points; "observed region" fill (same on every chart)
AGE = ["#8792A6", "#66728A", "#47556F", "#2C3A56", "#14213D"]  # slate ramp, older = darker (no blue)
LW = 2.6
plt.rcParams.update({
    "font.family": ["IBM Plex Sans", "DejaVu Sans"], "font.size": 15, "text.color": BODY,
    "axes.edgecolor": GRID, "axes.labelcolor": LABEL, "axes.labelsize": 15, "axes.linewidth": 1,
    "xtick.color": LABEL, "ytick.color": LABEL, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "xtick.major.pad": 8, "ytick.major.pad": 8,
    "savefig.transparent": True})
C = np.load("review2/visual_editor/cache.npz")
pct = lambda v: (np.exp(v) - 1) * 100
sgn = lambda v, _: (f"{v:+.0f}%".replace("-", "−") if v else "0%")
neg = lambda fmt: (lambda v, _: fmt.format(v).replace("-", "−"))


class Chart:
    """Fixed-geometry figure. w,h = displayed size on the slide in px; PNG is 2x. rect = axes [l,b,w,h] in figure fraction."""
    def __init__(self, name, w, h, rect):
        self.name, self.w, self.h = name, w, h
        self.f = plt.figure(figsize=(w / 100, h / 100), dpi=200)
        self.ax = self.f.add_axes(rect)
        for s in ["top", "right", "left"]: self.ax.spines[s].set_visible(False)
        self.ax.grid(axis="y", color=GRID, lw=0.8); self.ax.set_axisbelow(True)
        self.ax.patch.set_alpha(0)
        self.layer = {}; self.cur = 0
    def on(self, k): self.cur = k; return self
    def add(self, *arts):
        for a in arts:
            for b in (a if isinstance(a, (list, tuple)) else [a]): self.layer[b] = self.cur
        return arts[0] if len(arts) == 1 else arts
    def ytitle(self, s):  # horizontal y-axis title, flush-left with the figure edge (= the HTML chart heading), 20px above the plot (clears the top tick label)
        self.ax.set_ylabel("")
        top = self.ax.get_position().y1
        self.layer[self.f.text(2 / self.w, top + 20 / self.h, s, ha="left", va="bottom", color=LABEL, fontsize=15)] = 0
    def save(self):
        n = max(self.layer.values()) + 1
        for k in range(n):
            for a, L in self.layer.items(): a.set_visible(L == k)
            self.ax.set_axis_on() if k == 0 else self.ax.set_axis_off()
            self.f.savefig(f"{OUT}/layers/{self.name}_L{k}.png", dpi=200, transparent=True)
        for a in self.layer: a.set_visible(True)
        plt.close(self.f)
        base = Image.new("RGBA", (self.w * 2, self.h * 2), SURF)
        for k in range(n):
            base = Image.alpha_composite(base, Image.open(f"{OUT}/layers/{self.name}_L{k}.png").convert("RGBA"))
            base.convert("RGB").save(f"{OUT}/preview/{self.name}_step{k}.png")
        return n


def ring(ax, x, y, color, s=110, z=8):
    return ax.scatter([x], [y], s=s, facecolor=SURF, edgecolor=color, lw=LW, zorder=z, clip_on=False)


def dot(ax, x, y, color, s=90, z=8):
    return ax.scatter([x], [y], s=s, color=color, edgecolor=SURF, lw=1.8, zorder=z, clip_on=False)


# ------------------------------------------------------------------ Decision: margin with markers (900 x 520)
def decision():
    xs, tru, lgm, dm, lgl = C["xs"], C["tru"], C["lgm"], C["dm"], C["lgl"]
    k40 = xs <= 0.4001; x, tru, lgm, dm = xs[k40] * 100, tru[k40], lgm[k40], dm[k40]
    c = Chart("decision_margin", 900, 520, [0.085, 0.14, 0.895, 0.72]); ax = c.ax
    ax.set_xlim(-11, 44); ax.set_ylim(80, 130); ax.set_yticks(range(90, 131, 10))
    ax.set_xticks([-10, 0, 10, 20, 30, 40]); ax.xaxis.set_major_formatter(sgn)
    ax.set_xlabel("Price change for all customers"); c.ytitle("Expected margin (index, today = 100)")
    it = int(np.argmax(tru)); xo, yo = x[it], tru[it]
    # L0 truth + its optimum
    c.on(0).add(ax.axhline(100, color=LABEL, lw=1, zorder=1))
    truth = lambda: ax.plot(x, tru, color=INK, lw=LW, ls=(0, (5, 2.5)), zorder=5, solid_capstyle="round")
    c.add(truth()); c.add(dot(ax, xo, yo, INK))
    c.add(ax.text(xo, yo + 5.6, f"True optimum {xo:+.0f}%", ha="center", va="bottom", fontsize=15, color=INK, fontweight=600))
    # L1 LightGBM: leaves the frame, no optimum
    c.on(1).add(ax.plot(x, lgm, color=ORANGE, lw=LW, zorder=4, solid_capstyle="round"))
    xe = float(np.interp(130, lgm, x))
    c.add(ax.scatter([xe], [130.9], marker="^", s=120, color=ORANGE, zorder=9, clip_on=False, lw=0))
    c.add(ax.text(xe + 1.6, 126.8, "LightGBM: no optimum", ha="left", va="center", fontsize=15, color=INK, fontweight=600))
    # L2 Double ML: curve + its 95% band (margin curves at the two ends of the CI of theta); truth redrawn on top
    blo, bhi = np.minimum(C["dlo"][k40], C["dhi"][k40]), np.maximum(C["dlo"][k40], C["dhi"][k40])
    c.on(2).add(ax.fill_between(x, blo, bhi, color=BLUE, alpha=0.13, lw=0, zorder=3))
    c.add(ax.plot(x, dm, color=BLUE, lw=LW, zorder=4, solid_capstyle="round")); c.add(truth()); c.add(dot(ax, xo, yo, INK))
    c.add(ax.text(43.5, blo[-1] - 2.0, "Double ML, 95% band", ha="right", va="top", fontsize=15, color=INK, fontweight=600))
    return c.save()


# ------------------------------------------------------------------ Why: convergence (900 x 520), n >= 500
def convergence(style="band"):
    mc = json.load(open("out/mc_conv.json")); ns = np.array(sorted(int(k) for k in mc))
    arr = {n: np.array(mc[str(n)]) for n in ns}
    sm = np.array([np.median(arr[n][:, 0]) for n in ns]); dmn = np.array([np.median(arr[n][:, 1]) for n in ns])   # median: DML is skewed at small n
    slo, shi = np.array([np.percentile(arr[n][:, 0], [10, 90]) for n in ns]).T
    dlo, dhi = np.array([np.percentile(arr[n][:, 1], [10, 90]) for n in ns]).T
    c = Chart("why_convergence", 900, 520, [0.075, 0.14, 0.78, 0.72]); ax = c.ax
    Y0, Y1 = -0.3, 1.6
    ax.set_xscale("log"); ax.set_xlim(ns[0] / 1.3, ns[-1] * 1.3); ax.set_ylim(Y0, Y1)
    tk = [n for n in ns if n not in (300, 400)]
    ax.set_xticks(tk); ax.set_xticklabels([f"{n//1000}k" if n >= 1000 else str(n) for n in tk]); ax.minorticks_off()
    ax.set_yticks([0, 0.5, 1.0, 1.5]); ax.yaxis.set_major_formatter(neg("{:.1f}"))
    ax.set_xlabel("Portfolio size (policies, log scale)"); c.ytitle("Estimated price effect (pp per +1% price)")
    X1 = ns[-1] * 1.5
    truth = lambda: ax.plot([ns[0] / 1.3, ns[-1] * 1.3], [THETA, THETA], color=INK, lw=1.6, ls=(0, (5, 2.5)), zorder=6)
    ZE = np.sqrt(200 * 300)   # "too little data" zone: 100 and 200 policies
    c.on(0).add(ax.axvspan(ns[0] / 1.3, ZE, color="#F0F0EA", lw=0, zorder=0))
    c.add(ax.text(np.sqrt(ns[0] / 1.3 * ZE), Y1 - 0.06, "Too little\ndata", ha="center", va="top", fontsize=15, color=LABEL, linespacing=1.2))
    c.add(truth()); c.add(ax.text(X1, THETA, "Truth", va="center", fontsize=15, color=LABEL, clip_on=False))
    def band(lo, hi, col): return [ax.fill_between(ns, np.maximum(lo, Y0), np.minimum(hi, Y1), color=col, alpha=0.12, lw=0, zorder=2)]
    def line(vals, col, z):
        inside = vals >= Y0
        arts = [ax.plot(ns, np.maximum(vals, Y0), color=col, lw=LW, zorder=z, solid_capstyle="round")[0],
                ax.scatter(ns[inside], vals[inside], s=49, color=col, edgecolor=SURF, lw=1.5, zorder=z + 0.5)]
        if (~inside).any():   # off-scale point: marker on the bottom edge
            arts.append(ax.scatter(ns[~inside], np.full((~inside).sum(), Y0 + 0.045), marker="v", s=110, color=col, lw=0, zorder=z + 0.5, clip_on=False))
        return arts
    lg = lambda: line(sm, ORANGE, 5)
    c.on(1).add(band(slo, shi, ORANGE)); c.add(lg())
    c.add(ax.text(X1, sm[-1], "LightGBM", va="center", fontsize=15, color=INK, fontweight=600, clip_on=False))
    j = list(ns).index(50000)
    c.add(ax.annotate("", xy=(ns[j], THETA - 0.02), xytext=(ns[j], sm[j] + 0.025),
                      arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.3, mutation_scale=14, shrinkA=0, shrinkB=0)))
    c.add(ax.text(ns[j] * 1.1, (THETA + sm[j]) / 2, "Bias", va="center", fontsize=15, color=INK, fontweight=600))
    c.on(2).add(band(dlo, dhi, BLUE)); c.add(lg()); c.add(line(dmn, BLUE, 6)); c.add(truth())
    c.add(ax.text(X1, 1.17, "Double ML", va="center", fontsize=15, color=INK, fontweight=600, clip_on=False))
    return c.save()


# ------------------------------------------------------------------ How Double ML works: three steps (520 x 250 each)
def steps():
    A, T, Y, lh, mh, Yr, Tr, th, S2 = (C[k] for k in ["A", "T", "Y", "lh", "mh", "Yr", "Tr", "th", "S2"])
    rect = [0.13, 0.22, 0.84, 0.58]
    # a) lapse model
    c = Chart("how_step1", 520, 250, rect); ax = c.ax
    ab = np.arange(25, 75); am = ab + 0.5
    ly = [Y[(A >= a) & (A < a + 1)].mean() * 100 for a in ab]; lf = [lh[(A >= a) & (A < a + 1)].mean() * 100 for a in ab]
    c.add(ax.scatter(am, ly, s=26, color=DATA, edgecolor=SURF, lw=0.8, zorder=3)); c.add(ax.plot(am, lf, color=AQUA, lw=LW, zorder=4))
    ax.set_xlim(25, 75); ax.set_ylim(15, 40); ax.set_yticks([20, 30, 40]); ax.set_xlabel("Age"); c.ytitle("Lapse rate (%)"); c.save()
    # b) price model
    c = Chart("how_step2", 520, 250, rect); ax = c.ax
    c.add(ax.scatter(A[S2], pct(T[S2]), s=5, color=DATA, alpha=0.55, lw=0, zorder=2))
    o = np.argsort(A[S2]); c.add(ax.plot(A[S2][o], pct(mh[S2][o]), color=VIOLET, lw=LW, zorder=4))
    ax.set_xlim(25, 75); ax.set_ylim(-7, 20); ax.set_yticks([0, 10, 20]); ax.yaxis.set_major_formatter(sgn)
    ax.set_xlabel("Age"); c.ytitle("Price vs age-25 base (%)"); c.save()
    # c) residual on residual
    c = Chart("how_step3", 520, 250, rect); ax = c.ax
    q = np.quantile(Tr, np.linspace(0.01, 0.99, 21)); idx = np.digitize(Tr, q)
    bx = [Tr[idx == k].mean() * 100 for k in range(1, 21)]; by = [Yr[idx == k].mean() * 100 for k in range(1, 21)]
    xx = np.linspace(-0.05, 0.05, 10)
    c.add(ax.scatter(bx, by, s=30, color=DATA, edgecolor=SURF, lw=0.8, zorder=3))
    c.add(ax.plot(xx * 100, th * xx * 100, color=BLUE, lw=LW, zorder=4))   # no truth line here: it would hide the fit
    ax.set_xlim(-5, 5); ax.set_ylim(-5.5, 5.5); ax.set_yticks([-5, 0, 5])
    ax.xaxis.set_major_formatter(neg("{:.0f}")); ax.yaxis.set_major_formatter(neg("{:.0f}"))
    ax.set_xlabel("Price residual (%)"); c.ytitle("Lapse residual (pp)"); c.save()


# ------------------------------------------------------------------ Confounding (1000 x 540): grey -> by age
def confounding():
    A, T, Y = C["A"], C["T"], C["Y"]
    c = Chart("conf_simpson", 1000, 540, [0.07, 0.13, 0.92, 0.77]); ax = c.ax
    pi_ = np.polyfit(T, Y, 1); xx = np.linspace(np.quantile(T, .01), np.quantile(T, .99), 50)
    c.on(0).add(ax.plot(pct(xx), np.polyval(pi_, xx) * 100, color="#C3C8D0", lw=7, solid_capstyle="round", zorder=1))
    c.add(ax.text(pct(xx[-1]) + 0.5, np.polyval(pi_, xx[-1]) * 100, "All ages together", ha="left", va="center",
                  fontsize=15, color=BODY))
    pts = []
    for i, a in enumerate([30, 40, 50, 60, 70]):
        s = np.abs(A - a) < 2.5
        q = np.quantile(T[s], np.linspace(0, 1, 7)); idx = np.digitize(T[s], q[1:-1])
        bx = [pct(T[s][idx == k].mean()) for k in range(6)]; by = [Y[s][idx == k].mean() * 100 for k in range(6)]
        pts.append((bx, by, s))
        c.on(0).add(ax.scatter(bx, by, s=50, color=DATA, edgecolor=SURF, linewidth=1.5, zorder=4))
    for i, (a, (bx, by, s)) in enumerate(zip([30, 40, 50, 60, 70], pts)):
        sl, ic = np.polyfit(T[s], Y[s], 1); lo, hi = np.quantile(T[s], [0.03, 0.97]); xx = np.linspace(lo, hi, 20)
        c.on(1).add(ax.plot(pct(xx), (sl * xx + ic) * 100, color=AGE[i], lw=LW, solid_capstyle="round", zorder=3))
        c.add(ax.scatter(bx, by, s=50, color=AGE[i], edgecolor=SURF, linewidth=1.5, zorder=4))
        c.add(ax.text(pct(hi) + 0.35, (sl * hi + ic) * 100, f"Age ≈ {a}", ha="left", va="center", fontsize=15, color=INK, fontweight=500))
    ax.set_xlim(-4, 21.5); ax.set_ylim(13, 41); ax.xaxis.set_major_formatter(sgn)
    ax.set_xlabel("Price vs age-25 base (%)"); c.ytitle("Lapse rate (%)")
    return c.save()


# ------------------------------------------------------------------ Extrapolation (800 x 440 each)
def extrapolation():
    A, T, S = C["A"], C["T"], C["S"]; S = S[:900]; aa = np.linspace(25, 75, 200); sh = np.log1p(0.20)
    c = Chart("extra_support", 800, 440, [0.10, 0.15, 0.88, 0.74]); ax = c.ax
    c.on(0).add(ax.fill_between(aa, pct(m(aa) - 2 * SIG_V), pct(m(aa) + 2 * SIG_V), color=OBS, lw=0, zorder=1))
    c.add(ax.scatter(A[S], pct(T[S]), s=9, color=DATA, alpha=0.8, lw=0, zorder=2))
    c.add(ax.text(74.5, 4.5, "Observed customers", color=BODY, fontsize=15, ha="right", va="top"))
    c.on(1).add(ax.scatter(A[S], pct(T[S] + sh), s=10, facecolor="none", edgecolor=INK, alpha=0.45, lw=0.7, zorder=3))
    c.add(ax.annotate("", xy=(30, pct(m(30) + sh) - 1.2), xytext=(30, pct(m(30)) + 3.5),
                      arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.0, mutation_scale=20)))
    c.add(ax.text(26, 50, "Same customers after +20%:\nno observed customer looks like this", color=INK, fontsize=15,
                  fontweight=600, va="top", linespacing=1.3))
    ax.set_xlim(25, 75); ax.set_ylim(-9, 50); ax.set_yticks([0, 10, 20, 30, 40]); ax.yaxis.set_major_formatter(sgn)
    ax.set_xlabel("Age"); c.ytitle("Price vs age-25 base (%)"); c.save()

    xs, tl, ll = C["xs"], C["truth_l"], C["lgbm_l"]; k = xs <= 0.4001; x, tl, ll = xs[k] * 100, tl[k] * 100, ll[k] * 100
    c = Chart("extra_whatif", 800, 440, [0.10, 0.15, 0.74, 0.74]); ax = c.ax
    c.on(0).add(ax.axvspan(-4, 4, color=OBS, lw=0, zorder=0))
    c.add(ax.text(0, 63.5, "Observed\nprice variation", ha="center", va="top", fontsize=15, color=BODY, linespacing=1.2))
    truth = lambda: ax.plot(x, tl, color=INK, lw=LW, ls=(0, (5, 2.5)), zorder=5)
    c.add(truth()); c.add(ax.text(41.5, tl[-1], "Truth", va="center", fontsize=15, color=INK, fontweight=600, clip_on=False))
    c.on(1).add(ax.plot(x, ll, color=ORANGE, lw=LW, zorder=4, solid_capstyle="round")); c.add(truth())
    c.add(ax.text(41.5, ll[-1], "LightGBM", va="center", fontsize=15, color=INK, fontweight=600, clip_on=False))
    ax.set_xlim(-11, 40.5); ax.set_ylim(12, 66); ax.set_yticks([20, 30, 40, 50, 60]); ax.xaxis.set_major_formatter(sgn)
    ax.set_xlabel("Price change for all customers"); c.ytitle("Lapse rate (%)"); c.save()


if __name__ == "__main__":
    print("decision layers", decision()); print("convergence layers", convergence())
    steps(); print("confounding layers", confounding()); extrapolation(); print("done")
