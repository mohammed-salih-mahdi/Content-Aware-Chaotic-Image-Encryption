"""
nist_randomness_tests.py
-------------------------------------------------------------------
Full NIST SP 800-22 Rev. 1a statistical test suite (15 tests, 34
individual sub-results once the two multi-state tests are expanded).

This module is completely independent of the encryption code: it
only operates on a 1-D numpy array of bits (0/1 values), so it can be
imported and unit-tested on its own, or reused to test the randomness
of any other bit stream.

Implementation notes / practical simplifications
--------------------------------------------------
- Non-overlapping / Overlapping Template Matching use the single
  canonical 9-bit templates "000000001" / "111111111" (the default
  templates used in NIST's own reference implementation for a quick
  single-template run), rather than iterating the full set of 148
  aperiodic 9-bit templates the complete STS uses.
- Linear Complexity caps the number of 500-bit blocks processed at
  300 (still above NIST's recommended minimum of N=200) to keep the
  pure-Python Berlekamp-Massey step from dominating runtime.

Each individual `nist_XX_*` function returns a p-value (float) or
`None` when the input bit stream is too short for that test to be
statistically meaningful (per NIST's own minimum-length guidance).
A bit stream passes a test at significance level `alpha` when its
p-value >= alpha (NIST's own recommended default is alpha = 0.01).

Usage
-----
    import numpy as np
    from nist_randomness_tests import nist_sp800_22_full_suite

    bits = np.unpackbits(np.frombuffer(some_bytes, dtype=np.uint8))
    results, verdict = nist_sp800_22_full_suite(bits, alpha=0.01)
"""

import math

import numpy as np
from scipy.special import gammaincc, erfc as sp_erfc
from scipy.stats import norm as sp_norm


# =====================================================================
# Internal helpers
# =====================================================================

def _get_block_counts(bits, m):
    """Vectorized count of every overlapping m-bit pattern (circular)."""
    n = len(bits)
    if m == 0:
        return np.array([n])
    padded = np.concatenate([bits, bits[:m - 1]]) if m > 1 else bits
    vals = np.zeros(n, dtype=np.int64)
    for j in range(m):
        vals = (vals << 1) | padded[j:j + n].astype(np.int64)
    return np.bincount(vals, minlength=2 ** m)


def _gf2_rank(mat):
    """Rank of a binary matrix over GF(2) via Gaussian elimination."""
    mat = mat.copy().astype(np.uint8)
    rows, cols = mat.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for r in range(rank, rows):
            if mat[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != rank:
            mat[[rank, pivot]] = mat[[pivot, rank]]
        for r in range(rows):
            if r != rank and mat[r, col] == 1:
                mat[r] ^= mat[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def _berlekamp_massey(bits):
    """Linear complexity of a bit sequence (Berlekamp-Massey algorithm)."""
    n = len(bits)
    c = np.zeros(n, dtype=np.uint8); c[0] = 1
    b = np.zeros(n, dtype=np.uint8); b[0] = 1
    L, m = 0, -1
    for N in range(n):
        d = int(bits[N])
        for i in range(1, L + 1):
            d ^= int(c[i]) & int(bits[N - i])
        if d == 1:
            t = c.copy()
            shift = N - m
            if shift + len(b) <= n:
                c[shift:shift + len(b)] ^= b
            else:
                c[shift:] ^= b[:n - shift]
            if L <= N / 2:
                L = N + 1 - L
                m = N
                b = t
    return L


# =====================================================================
# Individual NIST SP 800-22 tests
# =====================================================================

def nist_01_monobit(bits):
    """Test 1: Frequency (Monobit) test."""
    n = len(bits)
    s = np.sum(2 * bits.astype(np.int64) - 1)
    s_obs = abs(s) / math.sqrt(n)
    return float(sp_erfc(s_obs / math.sqrt(2)))


def nist_02_block_frequency(bits, M=128):
    """Test 2: Frequency within a block."""
    n = len(bits)
    N = n // M
    if N == 0:
        return None
    blocks = bits[:N * M].reshape(N, M)
    pi = blocks.mean(axis=1)
    chi2 = 4 * M * np.sum((pi - 0.5) ** 2)
    return float(gammaincc(N / 2, chi2 / 2))


def nist_03_runs(bits):
    """Test 3: Runs test."""
    n = len(bits)
    pi_hat = bits.mean()
    if abs(pi_hat - 0.5) >= (2 / math.sqrt(n)):
        return 0.0  # prerequisite (monobit) not satisfied
    v_obs = 1 + np.sum(bits[1:] != bits[:-1])
    num = abs(v_obs - 2 * n * pi_hat * (1 - pi_hat))
    den = 2 * math.sqrt(2 * n) * pi_hat * (1 - pi_hat)
    return float(sp_erfc(num / den)) if den > 0 else 0.0


def nist_04_longest_run_of_ones(bits):
    """Test 4: Longest run of ones in a block."""
    n = len(bits)
    if n < 128:
        return None
    if n < 6272:
        M, K, N = 8, 3, 16
        pi = [0.2148, 0.3672, 0.2305, 0.1875]
        cat = lambda v: 0 if v <= 1 else (1 if v == 2 else (2 if v == 3 else 3))
    elif n < 750000:
        M, K, N = 128, 5, 49
        pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1892]
        cat = lambda v: 0 if v <= 4 else (1 if v == 5 else (2 if v == 6 else (3 if v == 7 else 4)))
    else:
        M, K, N = 10000, 6, 75
        pi = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
        cat = lambda v: (0 if v <= 10 else 1 if v == 11 else 2 if v == 12 else 3 if v == 13
                          else 4 if v == 14 else 5 if v == 15 else 6)
    if N * M > n:
        return None
    blocks = bits[:N * M].reshape(N, M)
    v_counts = np.zeros(len(pi))
    for row in blocks:
        max_run = cur = 0
        for b in row:
            cur = cur + 1 if b == 1 else 0
            max_run = max(max_run, cur)
        v_counts[cat(max_run)] += 1
    pi = np.array(pi)
    chi2 = np.sum((v_counts - N * pi) ** 2 / (N * pi))
    return float(gammaincc(K / 2, chi2 / 2))


def nist_05_binary_matrix_rank(bits, M=32, Q=32):
    """Test 5: Binary matrix rank test."""
    n = len(bits)
    N = n // (M * Q)
    if N < 38:  # NIST recommends N >= 38 matrices for this test size
        return None
    mats = bits[:N * M * Q].reshape(N, M, Q)
    ranks = np.array([_gf2_rank(m) for m in mats])
    FM = np.sum(ranks == M)
    FM1 = np.sum(ranks == M - 1)
    N_rest = N - FM - FM1
    p_full, p_m1, p_rest = 0.2888, 0.5776, 0.1336
    chi2 = ((FM - p_full * N) ** 2) / (p_full * N) \
        + ((FM1 - p_m1 * N) ** 2) / (p_m1 * N) \
        + ((N_rest - p_rest * N) ** 2) / (p_rest * N)
    return float(math.exp(-chi2 / 2))


def nist_06_spectral_dft(bits):
    """Test 6: Discrete Fourier Transform (Spectral) test."""
    n = len(bits)
    X = 2 * bits.astype(np.float64) - 1
    S = np.fft.fft(X)
    M = np.abs(S[:n // 2])
    T = math.sqrt(math.log(1 / 0.05) * n)
    N0 = 0.95 * n / 2
    N1 = np.sum(M < T)
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4)
    return float(sp_erfc(abs(d) / math.sqrt(2)))


def nist_07_nonoverlapping_template(bits, template=None, N=8):
    """Test 7: Non-overlapping template matching test."""
    if template is None:
        template = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.uint8)
    m = len(template)
    n = len(bits)
    M = n // N
    if M <= m:
        return None
    mu = (M - m + 1) / (2 ** m)
    var = M * (1 / 2 ** m - (2 * m - 1) / 2 ** (2 * m))
    W = np.zeros(N)
    for i in range(N):
        block = bits[i * M:(i + 1) * M]
        count = 0
        j = 0
        while j <= M - m:
            if np.array_equal(block[j:j + m], template):
                count += 1
                j += m
            else:
                j += 1
        W[i] = count
    chi2 = np.sum((W - mu) ** 2 / var)
    return float(gammaincc(N / 2, chi2 / 2))


def nist_08_overlapping_template(bits, template=None, M=1032):
    """Test 8: Overlapping template matching test."""
    if template is None:
        template = np.ones(9, dtype=np.uint8)
    m = len(template)
    n = len(bits)
    N = n // M
    if N == 0:
        return None
    K = 5
    # Standard NIST reference pi values for m=9, M=1032
    pi = np.array([0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865])
    v = np.zeros(K + 1)
    for i in range(N):
        block = bits[i * M:(i + 1) * M]
        count = 0
        for j in range(M - m + 1):
            if np.array_equal(block[j:j + m], template):
                count += 1
        v[min(count, K)] += 1
    chi2 = np.sum((v - N * pi) ** 2 / (N * pi))
    return float(gammaincc(K / 2, chi2 / 2))


_UNIVERSAL_L_TABLE = [(387840, 6), (904960, 7), (2068480, 8), (4654080, 9), (10430464, 10),
                      (22753280, 11), (49643520, 12), (107560960, 13), (231669760, 14),
                      (496435200, 15), (1059061760, 16)]
_UNIVERSAL_EXPECTED = {6: 5.2177052, 7: 6.1962507, 8: 7.1836656, 9: 8.1764248, 10: 9.1723243,
                       11: 10.170032, 12: 11.168765, 13: 12.168070, 14: 13.167693,
                       15: 14.167488, 16: 15.167379}
_UNIVERSAL_VARIANCE = {6: 2.954, 7: 3.125, 8: 3.238, 9: 3.311, 10: 3.356, 11: 3.384,
                        12: 3.401, 13: 3.410, 14: 3.416, 15: 3.419, 16: 3.421}


def nist_09_universal(bits):
    """Test 9: Maurer's universal statistical test."""
    n = len(bits)
    L = None
    for min_n, L_candidate in _UNIVERSAL_L_TABLE:
        if n >= min_n:
            L = L_candidate
    if L is None:
        return None  # sequence too short for a reliable Universal test
    Q = 10 * 2 ** L
    total_blocks = n // L
    K = total_blocks - Q
    if K <= 0:
        return None
    powers = 1 << np.arange(L - 1, -1, -1)
    blocks = bits[:total_blocks * L].reshape(total_blocks, L)
    ints = blocks.astype(np.int64) @ powers
    T = np.zeros(2 ** L, dtype=np.int64)
    T[ints[:Q]] = np.arange(1, Q + 1)
    total = 0.0
    for i in range(Q, Q + K):
        val = ints[i]
        if T[val] != 0:
            total += math.log2(i + 1 - T[val])
        T[val] = i + 1
    fn = total / K
    c = 0.7 - 0.8 / L + (4 + 32 / L) * (K ** (-3 / L)) / 15
    sigma = c * math.sqrt(_UNIVERSAL_VARIANCE[L] / K)
    if sigma == 0:
        return None
    return float(sp_erfc(abs(fn - _UNIVERSAL_EXPECTED[L]) / (math.sqrt(2) * sigma)))


def nist_10_linear_complexity(bits, M=500, max_blocks=300):
    """Test 10: Linear complexity test."""
    n = len(bits)
    N = min(n // M, max_blocks)
    if N < 200:
        return None  # NIST recommends N >= 200 blocks
    K = 6
    pi = np.array([0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833])
    mu = M / 2 + (9 + (-1) ** (M + 1)) / 36 - (M / 3 + 2 / 9) / 2 ** M
    v = np.zeros(7)
    for i in range(N):
        block = bits[i * M:(i + 1) * M]
        L = _berlekamp_massey(block)
        Tval = ((-1) ** M) * (L - mu) + 2 / 9
        if Tval <= -2.5: v[0] += 1
        elif Tval <= -1.5: v[1] += 1
        elif Tval <= -0.5: v[2] += 1
        elif Tval <= 0.5: v[3] += 1
        elif Tval <= 1.5: v[4] += 1
        elif Tval <= 2.5: v[5] += 1
        else: v[6] += 1
    chi2 = np.sum((v - N * pi) ** 2 / (N * pi))
    return float(gammaincc(K / 2, chi2 / 2))


def nist_11_serial(bits, m=3):
    """Test 11: Serial test. Returns a (p1, p2) tuple of p-values."""
    n = len(bits)
    if m < 2 or n < 2 ** (m + 1):
        return None, None

    def psi_sq(mm):
        if mm <= 0:
            return 0.0
        counts = _get_block_counts(bits, mm)
        return (2 ** mm / n) * np.sum(counts.astype(np.float64) ** 2) - n

    psi_m, psi_m1, psi_m2 = psi_sq(m), psi_sq(m - 1), psi_sq(m - 2)
    del1 = psi_m - psi_m1
    del2 = psi_m - 2 * psi_m1 + psi_m2
    p1 = float(gammaincc(2 ** (m - 1) / 2, del1 / 2)) if del1 >= 0 else None
    p2 = float(gammaincc(2 ** (m - 2) / 2, del2 / 2)) if del2 >= 0 else None
    return p1, p2


def nist_12_approximate_entropy(bits, m=2):
    """Test 12: Approximate entropy test."""
    n = len(bits)
    if n < 2 ** (m + 2):
        return None

    def phi(mm):
        if mm == 0:
            return 0.0
        counts = _get_block_counts(bits, mm)
        c = counts.astype(np.float64) / n
        c = c[c > 0]
        return float(np.sum(c * np.log(c)))

    ap_en = phi(m) - phi(m + 1)
    chi2 = 2 * n * (math.log(2) - ap_en)
    df = 2 ** m
    return float(gammaincc(df / 2, chi2 / 2))


def nist_13_cumulative_sums(bits, mode="forward"):
    """Test 13: Cumulative sums (Cusum) test, forward or backward mode."""
    n = len(bits)
    x = 2 * bits.astype(np.int64) - 1
    if mode == "backward":
        x = x[::-1]
    S = np.cumsum(x)
    z = int(np.max(np.abs(S)))
    if z == 0:
        return 1.0
    sqrt_n = math.sqrt(n)
    total1 = 0.0
    for k in range(int((-n / z + 1) / 4), int((n / z - 1) / 4) + 1):
        total1 += sp_norm.cdf(((4 * k + 1) * z) / sqrt_n) - sp_norm.cdf(((4 * k - 1) * z) / sqrt_n)
    total2 = 0.0
    for k in range(int((-n / z - 3) / 4), int((n / z - 1) / 4) + 1):
        total2 += sp_norm.cdf(((4 * k + 3) * z) / sqrt_n) - sp_norm.cdf(((4 * k + 1) * z) / sqrt_n)
    p = 1 - total1 + total2
    return float(min(max(p, 0.0), 1.0))


def _pi_x_k(x, k):
    """State-occupation probability used by the Random Excursions tests."""
    ax = abs(x)
    if k == 0:
        return 1 - 1 / (2 * ax)
    if k < 5:
        return (1 / (4 * ax ** 2)) * (1 - 1 / (2 * ax)) ** (k - 1)
    # k == 5 ("k>=5" category): remaining probability mass
    return (1 / (2 * ax)) * (1 - 1 / (2 * ax)) ** 4


def nist_14_15_random_excursions(bits):
    """Tests 14 & 15: Random Excursions and Random Excursions Variant.

    Returns (excursions_p_by_state, variant_p_by_state); either may be
    None if there are too few cycles (J < 500, per NIST recommendation).
    """
    x = 2 * bits.astype(np.int64) - 1
    S = np.cumsum(x)
    S = np.concatenate([[0], S, [0]])
    zero_pos = np.where(S == 0)[0]
    if len(zero_pos) < 2:
        return None, None
    cycles = [S[zero_pos[i]:zero_pos[i + 1] + 1] for i in range(len(zero_pos) - 1)]
    J = len(cycles)
    if J < 500:
        return None, None

    states = [-4, -3, -2, -1, 1, 2, 3, 4]
    excursions_p = {}
    for s in states:
        v = np.zeros(6)
        for cyc in cycles:
            cnt = int(np.sum(cyc == s))
            v[min(cnt, 5)] += 1
        chi2 = 0.0
        for k in range(6):
            pk = _pi_x_k(s, k)
            expected = J * pk
            if expected > 0:
                chi2 += (v[k] - expected) ** 2 / expected
        excursions_p[s] = float(gammaincc(5 / 2, chi2 / 2))

    variant_states = [s for s in range(-9, 10) if s != 0]
    variant_p = {}
    for s in variant_states:
        xi = sum(int(np.sum(cyc == s)) for cyc in cycles)
        denom = math.sqrt(2 * J * (4 * abs(s) - 2))
        variant_p[s] = float(sp_erfc(abs(xi - J) / denom)) if denom > 0 else None
    return excursions_p, variant_p


def nist_sp800_22_full_suite(bits, alpha=0.01):
    """Runs all 15 NIST SP 800-22 tests on a bit array.

    Returns
    -------
    (results, verdict) : tuple of dicts
        `results` maps a test label to its p-value (or None if the
        stream was too short for that test). `verdict` maps the same
        label to "PASS", "FAIL", or "N/A (insufficient data for this
        test length)".
    """
    results = {}

    results["01. Frequency (Monobit)"] = nist_01_monobit(bits)
    results["02. Block Frequency"] = nist_02_block_frequency(bits)
    results["03. Runs"] = nist_03_runs(bits)
    results["04. Longest Run of Ones in a Block"] = nist_04_longest_run_of_ones(bits)
    results["05. Binary Matrix Rank"] = nist_05_binary_matrix_rank(bits)
    results["06. Discrete Fourier Transform (Spectral)"] = nist_06_spectral_dft(bits)
    results["07. Non-overlapping Template Matching"] = nist_07_nonoverlapping_template(bits)
    results["08. Overlapping Template Matching"] = nist_08_overlapping_template(bits)
    results["09. Maurer's Universal Statistical"] = nist_09_universal(bits)
    results["10. Linear Complexity"] = nist_10_linear_complexity(bits)

    p1, p2 = nist_11_serial(bits)
    results["11a. Serial (delta psi^2)"] = p1
    results["11b. Serial (delta^2 psi^2)"] = p2

    results["12. Approximate Entropy"] = nist_12_approximate_entropy(bits)
    results["13a. Cumulative Sums (forward)"] = nist_13_cumulative_sums(bits, "forward")
    results["13b. Cumulative Sums (backward)"] = nist_13_cumulative_sums(bits, "backward")

    excursions_p, variant_p = nist_14_15_random_excursions(bits)
    if excursions_p is not None:
        for state, p in excursions_p.items():
            results[f"14. Random Excursions (state {state:+d})"] = p
    else:
        results["14. Random Excursions"] = None  # too few cycles (J < 500)
    if variant_p is not None:
        for state, p in variant_p.items():
            results[f"15. Random Excursions Variant (state {state:+d})"] = p
    else:
        results["15. Random Excursions Variant"] = None  # too few cycles (J < 500)

    verdict = {}
    for k, v in results.items():
        if v is None:
            verdict[k] = "N/A (insufficient data for this test length)"
        else:
            verdict[k] = "PASS" if v >= alpha else "FAIL"
    return results, verdict
