r"""
hurst_dfa.py - Hurst exponent by Detrended Fluctuation Analysis (DFA-1).

PHASE 2 PREP. Tested on SIMULATED data only. Do not apply it to the real
S&P 500 data until the baseline is finished and frozen.

What it measures
    H ~ 0.5 : no memory (each step independent, like a fair coin)
    H > 0.5 : persistent (big tends to follow big)
    H < 0.5 : anti-persistent (big tends to follow small)

Apply it to an INCREMENT series (returns, |returns|), never to prices:
a price path is already integrated, and DFA on it gives ~1.5.

Method (Peng et al. 1994)
    1. profile  Y(k) = cumulative sum of (x - mean x)
    2. cut Y into windows of length n, fit a straight line in each window
    3. F(n) = root-mean-square of what is left after removing the lines
    4. F(n) grows like n^H, so H = slope of log F(n) against log n

Run `python python\hurst_dfa.py` to execute the self-tests.
"""

import numpy as np


def dfa(x, min_window=10, max_window=None, n_windows=20):
    """
    Returns (H, window_sizes, fluctuations).
    x: 1-D increment series (e.g. daily log returns), at least 100 points.
    """
    x = np.asarray(x, dtype=float).ravel()
    if np.isnan(x).any():
        raise ValueError("NaN values found")
    N = x.size
    if N < 100:
        raise ValueError("Need at least 100 observations for a stable estimate")
    if max_window is None:
        max_window = N // 4
    if not (4 <= min_window < max_window <= N // 2):
        raise ValueError("Window range must satisfy 4 <= min < max <= N/2")

    profile = np.cumsum(x - x.mean())
    sizes = np.unique(np.floor(np.logspace(np.log10(min_window),
                                           np.log10(max_window),
                                           n_windows)).astype(int))
    fluct = np.empty(sizes.size)
    for i, n in enumerate(sizes):
        n_seg = N // n
        segs = profile[: n_seg * n].reshape(n_seg, n)
        t = np.arange(n)
        # least-squares straight line in every window at once
        t_c = t - t.mean()
        slope = (segs - segs.mean(axis=1, keepdims=True)) @ t_c / (t_c @ t_c)
        trend = segs.mean(axis=1, keepdims=True) + slope[:, None] * t_c
        fluct[i] = np.sqrt(np.mean((segs - trend) ** 2))

    H = np.polyfit(np.log(sizes), np.log(fluct), 1)[0]
    return float(H), sizes, fluct


def hurst(x, **kwargs):
    return dfa(x, **kwargs)[0]


# --------------------------------------------------------------------------
# Simulators for the self-tests
# --------------------------------------------------------------------------
def fractional_gaussian_noise(n, H, rng):
    """Exact fGn with a known Hurst exponent (Davies-Harte circulant method)."""
    k = np.arange(n + 1)
    gamma = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H)
                   + np.abs(k - 1) ** (2 * H))
    row = np.concatenate([gamma, gamma[-2:0:-1]])
    eig = np.fft.fft(row).real
    if (eig < 0).any():
        eig = np.maximum(eig, 0)
    m = row.size
    w = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    z = np.fft.fft(np.sqrt(eig / m) * w)
    return z.real[:n]


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    N, reps = 4000, 30
    passed = 0

    def avg_H(make):
        return np.mean([hurst(make()) for _ in range(reps)])

    # 1. Independent steps (random-walk increments) -> H = 0.5
    h = avg_H(lambda: rng.standard_normal(N))
    assert abs(h - 0.5) < 0.03, h
    print(f"Test 1 passed: random-walk steps       H = {h:.3f}  (theory 0.5)"); passed += 1

    # 2-3. Known persistent and anti-persistent memory is recovered
    for H_true in (0.7, 0.3):
        h = avg_H(lambda: fractional_gaussian_noise(N, H_true, rng))
        assert abs(h - H_true) < 0.05, (H_true, h)
        kind = "persistent     " if H_true > 0.5 else "anti-persistent"
        print(f"Test {2 if H_true > 0.5 else 3} passed: {kind} noise   H = {h:.3f}  (theory {H_true})"); passed += 1

    # 4. The price path itself is NOT what to feed in: it gives ~1.5
    h = avg_H(lambda: np.cumsum(rng.standard_normal(N)))
    assert abs(h - 1.5) < 0.08, h
    print(f"Test 4 passed: price-like path         H = {h:.3f}  (theory 1.5, so use returns)"); passed += 1

    # 5. Volatility clustering: returns look memoryless, their SIZE does not
    def clustered_returns():
        lv = np.zeros(N)
        for t in range(1, N):
            lv[t] = 0.98 * lv[t - 1] + 0.15 * rng.standard_normal()
        return np.exp(lv) * rng.standard_normal(N)
    hr, ha = [], []
    for _ in range(reps):
        r = clustered_returns()
        hr.append(hurst(r)); ha.append(hurst(np.abs(r)))
    hr, ha = np.mean(hr), np.mean(ha)
    assert abs(hr - 0.5) < 0.05 and ha > 0.6, (hr, ha)
    print(f"Test 5 passed: clustered returns       H = {hr:.3f},  |returns| H = {ha:.3f}"); passed += 1

    # 6. Shuffling destroys memory -> back to 0.5
    x = fractional_gaussian_noise(N, 0.8, rng)
    h_shuf = np.mean([hurst(rng.permutation(x)) for _ in range(reps)])
    assert abs(h_shuf - 0.5) < 0.04, h_shuf
    print(f"Test 6 passed: shuffled H=0.8 series   H = {h_shuf:.3f}  (memory destroyed)"); passed += 1

    # 7. Bad input rejected
    for bad in (np.ones(50), np.array([1.0] * 200 + [np.nan])):
        try:
            hurst(bad); raise AssertionError("accepted bad input")
        except ValueError:
            pass
    print("Test 7 passed: too-short series and NaN rejected"); passed += 1

    print(f"\nAll {passed} tests passed.")
