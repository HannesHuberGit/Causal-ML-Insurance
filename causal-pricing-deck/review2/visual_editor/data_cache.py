"""Recompute the chart data exactly as charts2.py does (same seeds) and cache arrays for the mock-ups."""
import sys, numpy as np, warnings, time
sys.path.insert(0, "."); warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from cont import g, m, THETA, SIG_V
from conv import fit_es
t0 = time.time()
COST = 0.60; rng = np.random.default_rng(7); N = 200_000
A = rng.uniform(25, 75, N); V = rng.normal(0, SIG_V, N); T = m(A) + V
p = np.clip(g(A) + THETA * T, 0, 1); Y = rng.binomial(1, p).astype(float)
mod = fit_es(np.c_[A, T], Y)
X = A.reshape(-1, 1); lh, mh = np.zeros(N), np.zeros(N)
for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
    lh[te] = fit_es(X[tr], Y[tr]).predict(X[te]); mh[te] = fit_es(X[tr], T[tr]).predict(X[te])
Yr, Tr = Y - lh, T - mh; th = (Tr @ Yr) / (Tr @ Tr); e = Yr - th * Tr
se = np.sqrt(np.mean(Tr ** 2 * e ** 2)) / np.mean(Tr ** 2) / np.sqrt(N)
# consume rng exactly like charts2.py (slide-3 sample, slide-4 sample)
S = rng.choice(N, 2200, replace=False); S2 = rng.choice(N, 1500, replace=False)
xs = np.round(np.linspace(-0.10, 0.50, 61), 4)   # extended to +50% for label room (only mock-up)
truth_l = np.array([np.clip(g(A) + THETA * (T + np.log1p(x)), 0, 1).mean() for x in xs])
lgbm_l = np.array([mod.predict(np.c_[A, T + np.log1p(x)]).mean() for x in xs])
P0 = np.exp(T)
def margin(lap, x): return np.mean((1 - lap) * P0 * (1 + x - COST))
base_m = margin(p, 0.0)
tru = np.array([margin(np.clip(g(A) + THETA * (T + np.log1p(x)), 0, 1), x) for x in xs]) / base_m * 100
lgm = np.array([margin(np.clip(mod.predict(np.c_[A, T + np.log1p(x)]), 0, 1), x) for x in xs]) / base_m * 100
def lin_curve(t): return np.array([margin(np.clip(lh + t * np.log1p(x), 0, 1), x) for x in xs]) / base_m * 100
dm, dlo, dhi = lin_curve(th), lin_curve(th + 1.96 * se), lin_curve(th - 1.96 * se)
# "LightGBM linearised": LightGBM's own slope 0.82 applied linearly on top of its in-sample prediction
d = np.log1p(.02) - np.log1p(-.02)
s_lgbm = (mod.predict(np.c_[A, T + np.log1p(.02)]).mean() - mod.predict(np.c_[A, T + np.log1p(-.02)]).mean()) / d
base_pred = mod.predict(np.c_[A, T])
lgl = np.array([margin(np.clip(base_pred + s_lgbm * np.log1p(x), 0, 1), x) for x in xs]) / base_m * 100
np.savez("review2/visual_editor/cache.npz", A=A, T=T, Y=Y, lh=lh, mh=mh, Yr=Yr, Tr=Tr, th=th, se=se, S=S, S2=S2, xs=xs,
         truth_l=truth_l, lgbm_l=lgbm_l, tru=tru, lgm=lgm, dm=dm, dlo=dlo, dhi=dhi, lgl=lgl, s_lgbm=s_lgbm)
i = lambda x: int(np.argmin(abs(xs - x)))
print("theta", th, "se", se, "s_lgbm", s_lgbm, "time", round(time.time() - t0))
print("opt truth", xs[np.argmax(tru)], tru.max(), "opt dml", xs[np.argmax(dm)], "dml lo/hi opt", xs[np.argmax(dlo)], xs[np.argmax(dhi)])
print("lgbm-lin argmax", xs[np.argmax(lgl)], "its promise", lgl.max(), "truth there", tru[np.argmax(lgl)])
print("lgbm margin at", [(x, round(lgm[i(x)], 1)) for x in [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.3, 0.4]])
print("truth at 39", tru[i(.39)], "at 40", tru[i(.4)])
