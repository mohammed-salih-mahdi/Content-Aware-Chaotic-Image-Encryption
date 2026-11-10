

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm
from skimage.metrics import structural_similarity as ssim


def channels(image: np.ndarray) -> List[np.ndarray]:
    return [image] if image.ndim == 2 else [image[:, :, k] for k in range(image.shape[2])]


def channel_names(image: np.ndarray) -> List[str]:
    return ["Gray"] if image.ndim == 2 else ["Blue", "Green", "Red"][:image.shape[2]]


def shannon_entropy(values: np.ndarray) -> float:
    hist = np.bincount(values.reshape(-1).astype(np.int64), minlength=256)
    prob = hist[hist > 0].astype(np.float64) / values.size
    return float(-np.sum(prob * np.log2(prob)))


def global_entropy(image: np.ndarray) -> Dict[str, float]:
    out = {n: shannon_entropy(ch) for n, ch in zip(channel_names(image), channels(image))}
    out["mean"] = float(np.mean(list(out.values())))
    return out


LOCAL_BLOCK = 32# 32x 32= 1936 pixels 
LOCAL_K = 25              # max. number of non-overlapping 32x32 blocks in 256x256


@lru_cache(maxsize=None)
def local_entropy_reference(tb: int = LOCAL_BLOCK * LOCAL_BLOCK, k: int = LOCAL_K, alpha: float = 0.05):
    """Exact mean / s.d. of the Shannon entropy of a block of `tb` i.i.d.
    uniform 8-bit samples (multinomial model), and the two-sided
    acceptance interval for the mean of `k` blocks at level `alpha`.
    For tb=1936 the mean is 7.902469 (identical to Wu et al., 2013)."""
    K, p, n = 256, 1.0 / 256, tb
    x = np.arange(0, 121)
    g = np.where(x > 0, -(x / n) * np.log2(np.maximum(x, 1) / n), 0.0)
    pb = np.exp(gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1) + x * np.log(p) + (n - x) * np.log1p(-p))
    X1, X2 = np.meshgrid(x, x, indexing="ij")
    rest = n - X1 - X2
    pj = np.exp(gammaln(n + 1) - gammaln(X1 + 1) - gammaln(X2 + 1) - gammaln(rest + 1)
                + (X1 + X2) * np.log(p) + rest * np.log1p(-2 * p))
    eg, eg2, egg = np.sum(pb * g), np.sum(pb * g * g), np.sum(pj * np.outer(g, g))
    mu = K * eg
    sd_block = float(np.sqrt(K * (eg2 - eg ** 2) + K * (K - 1) * (egg - eg ** 2)))
    half = norm.ppf(1 - alpha / 2) * sd_block / np.sqrt(k)
    return {"mu": float(mu), "sd_block": sd_block, "k": k, "tb": tb, "alpha": alpha,
            "lower": float(mu - half), "upper": float(mu + half)}


def local_entropy(image: np.ndarray, seed: int, block: int = LOCAL_BLOCK, k: int = LOCAL_K) -> Dict:

    rng = np.random.default_rng(seed)
    ref = local_entropy_reference(block * block, k)
    out = {"block": f"{block}x{block}", "k": k}
    vals = []
    for name, ch in zip(channel_names(image), channels(image)):
        h, w = ch.shape
        tiles = [(r, c) for r in range(0, h - block + 1, block) for c in range(0, w - block + 1, block)]
        if len(tiles) < k:
            raise ValueError(f"Only {len(tiles)} tiles of {block}x{block} available; need {k}.")
        pick = rng.choice(len(tiles), size=k, replace=False)
        ent = [shannon_entropy(ch[tiles[i][0]:tiles[i][0] + block, tiles[i][1]:tiles[i][1] + block]) for i in pick]
        m = float(np.mean(ent))
        out[name] = m
        out[f"{name}_pass"] = bool(ref["lower"] <= m <= ref["upper"])
        vals.append(m)
    out["mean"] = float(np.mean(vals))
    out["sd_over_channels"] = float(np.std(vals))
    out["all_pass"] = all(out[f"{n}_pass"] for n in channel_names(image))
    return out


def local_entropy_32_pooled(image: np.ndarray) -> Tuple[float, float]:

    img = image if image.ndim == 3 else image[:, :, None]
    v = []
    for y in range(0, img.shape[0] - 31, 32):
        for x in range(0, img.shape[1] - 31, 32):
            v.append(shannon_entropy(img[y:y + 32, x:x + 32, :]))
    return float(np.mean(v)), float(np.std(v))


def correlation(image: np.ndarray) -> Dict[str, float]:
    out = {}
    for name, ch in zip(channel_names(image), channels(image)):
        a = ch.astype(np.float64)
        pairs = {"H": (a[:, :-1], a[:, 1:]), "V": (a[:-1, :], a[1:, :]), "D": (a[:-1, :-1], a[1:, 1:])}
        for d, (u, v) in pairs.items():
            u, v = u.ravel(), v.ravel()
            out[f"{name}_{d}"] = 1.0 if (u.std() == 0 or v.std() == 0) else float(np.corrcoef(u, v)[0, 1])
    vals = list(out.values())
    out["max_abs"] = float(np.max(np.abs(vals)))
    out["mean_abs"] = float(np.mean(np.abs(vals)))
    return out


def polar_resultant(image: np.ndarray) -> Dict[str, float]:
    """R = sqrt(C^2 + S^2) of the intensity histogram mapped to angles
    theta_i = 2*pi*i/256 (R=0 isotropic / uniform, R=1 fully directional)."""
    theta = 2 * np.pi * np.arange(256) / 256
    out = {}
    for name, ch in zip(channel_names(image), channels(image)):
        p = np.bincount(ch.reshape(-1).astype(np.int64), minlength=256).astype(float)
        p /= p.sum()
        out[name] = float(np.hypot(np.sum(p * np.cos(theta)), np.sum(p * np.sin(theta))))
    return out


def npcr_uaci(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    aa, bb = a.astype(np.int16), b.astype(np.int16)
    return float(np.mean(aa != bb) * 100.0), float(np.mean(np.abs(aa - bb)) / 255.0 * 100.0)


def psnr_ssim(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    psnr = float("inf") if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse)))
    s = float(np.mean([ssim(x, y, data_range=255) for x, y in zip(channels(a), channels(b))]))
    return psnr, s
