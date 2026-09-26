# Side check: does the bias/convergence story survive weaker age→price confounding?

Not part of the deck pipeline. Nothing in `causal-pricing-deck/{cont,conv,mc_*,rate_prod}.py`,
`final/`, `deck/`, `out/`, or `screenshots/` is touched by this folder.

## Question

The deck's toy world (`../cont.py`) sets `corr(age, price) ≈ 0.87` — very strong confounding.
Real motor-tariff age effects are usually one rating factor among several (bonus-malus, region,
vehicle power, discounts, competition), so an R² of "age alone explains most of price" is a
stress-test, not a realistic base case. This experiment dials confounding down to more modest
levels and checks whether the LightGBM S-learner still converges much slower than Double ML,
and whether the DML bias/SE story still holds.

No single published number for "the" correlation between age and premium exists (premium is a
multi-factor tariff, not just an age function), so the literature only supports a qualitative
call: age is *a* factor, not *the* factor. We test two levels on that basis:

- **moderate**: corr(age, price) ≈ 0.35
- **weak**: corr(age, price) ≈ 0.15

against the deck's own high-confounding run (reused as-is from `../../out/mc_conv.json`, corr ≈ 0.87)
as the reference.

## How confounding is diluted

`dgp.py` keeps the age tariff slope fixed at the deck's value (`+0.25%` price per year of age —
a real, unremarkable tariff effect) and instead widens the idiosyncratic price noise `V` so that
age explains a smaller share of total price variation. `THETA` (the true causal price→lapse
effect we're trying to recover) is unchanged at 1.0. `sig_for_corr()` solves analytically for the
noise SD that hits a target correlation.

`methods.py` is a straight copy of `../conv.py` (LightGBM S-learner + Double ML with early
stopping) — same fitting code, so any difference in results is attributable to confounding
strength alone, not to a different estimator.

## Run

```
cd causal-pricing-deck/experiments/low_confounding
python3 sweep.py          # quick pass: n up to 50k, 10 reps/level (~2 min total)
python3 plot_compare.py   # -> out/bias_convergence_compare.png + printed rate estimates
```

## Findings (this run: R=10 reps/level, n up to 50k — a quick pass, not deck-grade precision)

Medians (p10–p90 band) of the estimated price effect, truth = 1.0:

| n     | corr 0.87 (deck) LightGBM | corr 0.87 DML     | corr 0.35 LightGBM | corr 0.35 DML     | corr 0.15 LightGBM | corr 0.15 DML     |
|-------|---------------------------|--------------------|---------------------|---------------------|----------------------|----------------------|
| 2k    | 0.16 [0.05, 0.58]         | 0.79 [0.42, 1.39]  | 0.74 [0.63, 0.90]   | 1.03 [0.96, 1.08]   | 0.74 [0.60, 0.82]    | 0.85 [0.79, 0.87]    |
| 5k    | 0.46 [0.10, 0.82]         | 1.00 [0.53, 1.31]  | 0.73 [0.63, 0.80]   | 0.98 [0.93, 1.01]   | 0.73 [0.70, 0.81]    | 0.86 [0.85, 0.87]    |
| 10k   | 0.55 [0.28, 0.75]         | 0.98 [0.76, 1.14]  | 0.80 [0.73, 0.86]   | 0.97 [0.93, 1.00]   | 0.78 [0.75, 0.80]    | 0.85 [0.84, 0.86]    |
| 20k   | 0.63 [0.48, 0.76]         | 0.95 [0.77, 1.10]  | 0.84 [0.80, 0.92]   | 0.99 [0.97, 1.02]   | 0.80 [0.77, 0.82]    | 0.86 [0.85, 0.87]    |
| 50k   | 0.67 [0.55, 0.81]         | 0.98 [0.82, 1.14]  | 0.91 [0.87, 0.94]   | 0.98 [0.97, 1.01]   | 0.83 [0.81, 0.84]    | 0.86 [0.85, 0.87]    |

Chart: `out/bias_convergence_compare.png` (LightGBM orange, Double ML blue, truth dashed — same
convention as the deck's `why_convergence` chart, one panel per confounding level).

**The headline story survives at moderate confounding (corr ≈ 0.35):** LightGBM is still visibly
slow and still meaningfully biased at 50k policies (0.91, not 1.00), while Double ML is within a
few percent of the truth by 5k policies and stays there, with a much tighter band throughout.
Qualitatively the same plot as the deck's, just less dramatic.

**At weak confounding (corr ≈ 0.15) the story changes in an interesting way.** LightGBM is still
clearly biased and slow, as expected. But Double ML also stops converging to the truth — it
settles at a *stable* ≈0.86, with a very tight band, from 2k policies all the way to 50k. That's
not the "keeps closing in on 1.0" pattern from the other two panels; it's a small but persistent
bias that doesn't shrink with more data in this range.

Best guess at the mechanism (not verified further here): at corr ≈ 0.15, age explains very little
of price's variance (`kappa ≈ 0.02` — see `dgp.py`'s diagnostic print). The nuisance model for
price given age, `m̂(age)`, is fit with the same early-stopping settings tuned for the deck's much
stronger signal; when the true age→price signal is this faint relative to noise, early stopping
likely halts before the model picks up much of that real-but-faint slope, so `m̂(age)` under-fits
and a sliver of the confounding leaks through the residual — a first-stage regularization-bias
issue, not the classic "no adjustment at all" omitted-variable bias. If you want to chase this
further, the natural next step is re-tuning (or cross-validating) the nuisance learner's
early-stopping patience for this weaker-signal regime and re-running the corr-0.15 sweep, rather
than treating it as evidence that Double ML fails under weak confounding in general.

**Bottom line:** yes, the deck's qualitative point (LightGBM lags and stays biased longer; Double
ML gets close to the truth much faster and with tighter uncertainty) holds up under a realistic,
much weaker level of age→price confounding. But this run also surfaced a real caveat worth
knowing about before reusing this toy world for anything beyond the deck's slide: Double ML's own
reliability leans on the nuisance models being tuned to the actual signal strength, and that
tuning doesn't automatically transfer when you dial confounding down a lot.
