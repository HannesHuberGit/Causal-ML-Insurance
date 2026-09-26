import numpy as np

# Same toy world as ../cont.py. Confounding strength (corr(age, price)) is diluted by
# reallocating price variance between the age-driven part and idiosyncratic noise V,
# while holding TOTAL price variance fixed at the deck's own level. Earlier versions of
# this experiment instead inflated V alone with the age slope held fixed - that blew up
# var(T) itself, pushed ~14% of simulated lapse probabilities outside [0,1] at low
# confounding, and the resulting clipping (not nuisance under-fitting) was what caused
# Double ML to plateau away from the truth. Holding var(T) fixed removes that artifact.
THETA = 1.0
AGE_LO, AGE_HI = 25, 75
VAR_A = (AGE_HI - AGE_LO) ** 2 / 12   # Var(A) for A ~ U(AGE_LO, AGE_HI)

DECK_SLOPE = 0.0025      # deck's own "+0.25% price per year of age" tariff (../cont.py)
DECK_SIG = 0.02          # deck's own price noise SD (../cont.py)
DECK_VAR_T = DECK_SLOPE ** 2 * VAR_A + DECK_SIG ** 2   # total price variance to hold fixed


def g(A):
    return 0.35 - 0.006 * (A - 25) + 0.015 * np.sin((A - 25) / 6)


def m(A, slope=DECK_SLOPE):
    return slope * (A - 25)


def params_for_corr(target_corr, var_t=DECK_VAR_T):
    """(slope, sig) that hit corr(age, price) == target_corr while holding var(price) == var_t."""
    if not 0 < target_corr < 1:
        raise ValueError("target_corr must be in (0, 1)")
    slope = target_corr * np.sqrt(var_t / VAR_A)
    sig = np.sqrt(var_t * (1 - target_corr ** 2))
    return slope, sig


def draw(n, seed, slope, sig_v):
    r = np.random.default_rng(seed)
    A = r.uniform(AGE_LO, AGE_HI, n)
    V = r.normal(0, sig_v, n)
    T = slope * (A - 25) + V
    p = np.clip(g(A) + THETA * T, 0, 1)
    Y = r.binomial(1, p).astype(float)
    return A, T, Y, p


if __name__ == "__main__":
    print(f"deck var(T) held fixed at {DECK_VAR_T:.5f} (slope {DECK_SLOPE}, sig {DECK_SIG})")
    for target in [0.87, 0.35, 0.15]:
        slope, sig = (DECK_SLOPE, DECK_SIG) if target == 0.87 else params_for_corr(target)
        A, T, Y, p = draw(500_000, 7, slope, sig)
        p_raw = g(A) + THETA * T
        print(f"target corr {target:.2f} -> slope {slope:.6f} sig {sig:.4f} | measured corr "
              f"{np.corrcoef(A, T)[0, 1]:.3f} | var(T) {T.var():.5f} | frac clipped "
              f"{np.mean((p_raw < 0) | (p_raw > 1)):.4f}")
