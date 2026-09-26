import numpy as np

# Same toy world as ../cont.py, but SIG_V is derived from a target corr(age, price)
# instead of being fixed at 0.02. The age-tariff slope and the causal effect THETA
# are left exactly as in the deck: only the confounding strength changes.
THETA = 1.0
M_SLOPE = 0.0025          # deck's "+0.25% price per year of age" tariff, unchanged
AGE_LO, AGE_HI = 25, 75
VAR_A = (AGE_HI - AGE_LO) ** 2 / 12   # Var(A) for A ~ U(AGE_LO, AGE_HI)


def g(A):
    return 0.35 - 0.006 * (A - 25) + 0.015 * np.sin((A - 25) / 6)


def m(A, slope=M_SLOPE):
    return slope * (A - 25)


def sig_for_corr(target_corr, slope=M_SLOPE):
    """Noise SD that makes corr(age, price) == target_corr, holding the age slope fixed."""
    if not 0 < target_corr < 1:
        raise ValueError("target_corr must be in (0, 1)")
    var_m = slope ** 2 * VAR_A
    return (slope * np.sqrt(VAR_A)) * np.sqrt(1 - target_corr ** 2) / target_corr


def draw(n, seed, sig_v):
    r = np.random.default_rng(seed)
    A = r.uniform(AGE_LO, AGE_HI, n)
    V = r.normal(0, sig_v, n)
    T = m(A) + V
    p = np.clip(g(A) + THETA * T, 0, 1)
    Y = r.binomial(1, p).astype(float)
    return A, T, Y, p


if __name__ == "__main__":
    for target in [0.87, 0.35, 0.15]:
        sig_v = sig_for_corr(target)
        A, T, Y, p = draw(200_000, 7, sig_v)
        kappa = m(A).var() / sig_v ** 2
        print(f"target corr {target:.2f} -> sig_v {sig_v:.4f} | measured corr "
              f"{np.corrcoef(A, T)[0, 1]:.3f} | kappa {kappa:.3f}")
