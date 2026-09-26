# Causal pricing deck: LightGBM vs Double ML

An 8-slide deck on causal ML in insurance pricing. It uses a toy renewal/lapse portfolio with a known truth.

Live deck (Claude Slides artifact, private): https://claude.ai/artifact/TakwyxEBZxjuxXRuNt7iBw

Story: the question → confounding (age) → extrapolation (LightGBM with price as a feature freezes) → how Double ML works → pricing decision → why Double ML is better (convergence) → our setup: LightGBM first stage, then the final-stage options and why the causal forest.

## Toy world (`cont.py`)

- Age A ~ U(25, 75)
- Price (log) T = 0.0025·(A − 25) + V, with V ~ N(0, 0.02)
- Lapse probability p = g(A) + θ·T, where g(A) = 0.35 − 0.006·(A − 25) + 0.015·sin((A − 25)/6), and θ = 1
- Margin = (1 − lapse)·(premium·(1 + x) − claims), with claims = 60% of today's premium

## Methods (`conv.py`)

- **LightGBM (S-learner):**
  - fit on (age, price) with early stopping on a 20% hold-out (lr 0.05, 31 leaves, min_child_samples 50);
  - the price effect is the average ±2% what-if slope.
- **Double ML (partially linear model):**
  - the same LightGBM for ℓ(X) = E[Y|X] and m(X) = E[T|X];
  - 5-fold cross-fitting, then a residual-on-residual regression.

## Pipeline (run from this folder)

```
pip install -r requirements.txt
python3 review2/visual_editor/data_cache.py   # 200k-policy fits -> review2/visual_editor/cache.npz (already included)
python3 mc_conv.py 30; python3 mc_small.py; python3 mc_mid.py   # Monte Carlo -> out/mc_conv.json (already included; slow)
python3 rate_prod.py                           # measured DML product-term rate -> log in out/rate_prod.log
python3 final/charts_final.py                  # chart build layers -> final/layers/*_L<k>.png
python3 final/formulas_final.py                # formula PNGs -> final/layers/f_*.png
python3 render/render.py deck shots --builds   # local screenshots of every build step (Playwright/Chromium)
```

## Deck files

- `deck/project/deck.json` and `deck/project/slides/*.html` are the Claude Slides source.
  - Images are referenced as `/_blob/<id>`.
  - `render/blobmap.json` maps each id to the local PNG in `final/layers/`.
- To change a chart:
  1. Regenerate the chart.
  2. Upload the PNG as an artifact asset.
  3. Swap the `/_blob/<id>` in the slide HTML.
  4. Republish the slide file to the artifact URL above.
- Speaker notes live in each slide's `<aside>`.
- `screenshots/` holds the current render of every build step.

## Key numbers (200k policies unless noted)

- LightGBM price effect: 0.82 (truth 1.00).
- Double ML: θ̂ = 1.02, SE 0.049, 95% CI 0.92–1.11.
- Margin optimum:
  - truth +24% (index 113);
  - Double ML +24% (CI +19% to +30%);
  - LightGBM: no optimum (flat extrapolation).
- Convergence (medians of 30 portfolios per size):
  - Double ML ≈ 0.8 from 300 to 2k policies, then on the truth from about 5k;
  - LightGBM 0.16 at 2k, 0.63 at 20k, 0.82 at 200k.
- Measured rates:
  - LightGBM bias ≈ n^−0.3;
  - Double ML product term E[(m̂−m)(ℓ̂−ℓ)]/E[V²] ≈ n^−0.8;
  - on the slide, as typical orders: n^−1/3 and n^−2/3.
