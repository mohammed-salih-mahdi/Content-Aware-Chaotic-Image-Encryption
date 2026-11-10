

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import numpy as np  
from PIL import Image  

from .core import REFERENCE_PARAMS, trajectory  
from .metrics import polar_resultant  

DPI = 600
ORDER = ["Baboon", "Peppers", "Car", "Brain", "Chest", "Lung", "Two Macaws"]


def _rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img


def _show(ax, img, title=None):
    ax.imshow(_rgb(img), cmap=None if img.ndim == 3 else "gray", vmin=0, vmax=255)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)


def _save(fig, path: Path, dpi: int = DPI):

    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with Image.open(path.with_suffix(".png")) as im:
        im.convert("RGB").save(path.with_suffix(".tif"), compression="tiff_lzw", dpi=(dpi, dpi))


def fig2_visual(res, out: Path):

    names = [n for n in ORDER if n in res]
    fig, axes = plt.subplots(3, len(names), figsize=(7.5, 3.9), squeeze=False)
    for j, n in enumerate(names):
        a = res[n]["_arrays"]
        for i, k in enumerate(("original", "cipher", "recovered")):
            _show(axes[i, j], a[k], n if i == 0 else None)
    for i, t in enumerate(("Original", "Encrypted", "Decrypted")):
        axes[i, 0].text(-0.12, 0.5, t, transform=axes[i, 0].transAxes, rotation=90, va="center", ha="right",
                        fontsize=9)
    fig.tight_layout(w_pad=0.3, h_pad=0.3)
    _save(fig, out / "Fig2_visual_results", dpi=300)


def fig3_histograms(res, out: Path):
    names = [n for n in ("Baboon", "Peppers") if n in res]
    fig, axes = plt.subplots(len(names), 2, figsize=(7.5, 2.8 * len(names)))
    axes = np.atleast_2d(axes)
    for i, n in enumerate(names):
        for j, k in enumerate(("original", "cipher")):
            img = res[n]["_arrays"][k]
            ax = axes[i, j]
            for c, col in enumerate(("tab:blue", "tab:green", "tab:red")):
                ax.plot(np.bincount(img[:, :, c].ravel(), minlength=256), color=col, lw=0.8, label="BGR"[c])
            ax.set_title(f"{n} - {'original' if j == 0 else 'encrypted'}", fontsize=9)
            ax.set_xlim(0, 255)
            ax.set_xlabel("Pixel value", fontsize=8)
            ax.set_ylabel("Frequency", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, out / "Fig3_histograms")


def fig4_polar(res, out: Path, name="Baboon"):
    fig = plt.figure(figsize=(7.5, 5.2))
    theta = 2 * np.pi * np.arange(256) / 256
    for row, k in enumerate(("original", "cipher")):
        img = res[name]["_arrays"][k]
        R = polar_resultant(img)
        for c, (cn, col) in enumerate(zip(("Blue", "Green", "Red"), ("tab:blue", "tab:green", "tab:red"))):
            ax = fig.add_subplot(2, 3, row * 3 + c + 1, projection="polar")
            h = np.bincount(img[:, :, c].ravel(), minlength=256).astype(float)
            ax.bar(theta, h / h.max(), width=2 * np.pi / 256, color=col, alpha=0.8)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_title(f"{'Original' if row == 0 else 'Encrypted'} {cn}\nR = {R[cn]:.4f}", fontsize=8)
    fig.tight_layout()
    _save(fig, out / "Fig4_polar_histogram_Baboon")


def fig5_corr3d(res, out: Path, name="Baboon", n=3000, seed=7):
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(7.5, 5.2))
    dirs = [("Diagonal", 1, 1), ("Horizontal", 0, 1), ("Vertical", 1, 0)]
    for row, k in enumerate(("original", "cipher")):
        img = res[name]["_arrays"][k].astype(float)
        h, w = img.shape[:2]
        ys, xs = rng.integers(0, h - 1, n), rng.integers(0, w - 1, n)
        for c, (dn, dy, dx) in enumerate(dirs):
            ax = fig.add_subplot(2, 3, row * 3 + c + 1, projection="3d")
            for ch, col in zip((2, 1, 0), ("tab:red", "tab:green", "tab:blue")):
                ax.scatter(img[ys, xs, ch], img[ys + dy, xs + dx, ch], np.full(n, ch), s=0.3, c=col, alpha=0.5)
            ax.set_title(f"({'abcdef'[row * 3 + c]}) {'Original' if row == 0 else 'Encrypted'} - {dn}", fontsize=7)
            ax.tick_params(labelsize=5)
            ax.set_zticks([0, 1, 2])
            ax.set_zticklabels(["B", "G", "R"], fontsize=5)
    fig.tight_layout()
    _save(fig, out / "Fig5_3D_correlation_Baboon")


def fig6_key(res, out: Path, name="Baboon"):
    a = res[name]["_arrays"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))
    _show(axes[0], a["recovered"], "(a) Authorized decryption (correct K)")
    _show(axes[1], a["wrong_key"], "(b) Initial state x0 + 1e-12\n(1-bit change of K: package rejected by HMAC)")
    _save(fig, out / "Fig6_key_sensitivity", dpi=300)


def fig7_8_robust(res, out: Path, name="Baboon"):
    a = res[name]["_arrays"]
    rob = {(r["attack"], r["level_percent"]): r for r in res[name]["robustness"]}
    for kind, levels, fname in (("Noise", (1, 5, 10), "Fig7_noise_diagnostic"),
                                ("Cropping", (10, 25, 50), "Fig8_cropping_diagnostic")):
        fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.9))
        for ax, lv, tag in zip(axes, levels, "abc"):
            r = rob[(kind, lv)]
            _show(ax, a[f"robust_{kind}_{lv}"], f"({tag}) {lv}%\nPSNR {r['psnr_db']:.2f} dB, SSIM {r['ssim']:.4f}")
        _save(fig, out / fname, dpi=300)


def fig9_bifurcation(params, initials, out: Path, n_b=400, transient=10000, keep=30000, dt=0.002):
    bs = np.linspace(2.5, 3.0, n_b)
    pts = []
    for b in bs:
        p = np.array(params, dtype=float)
        p[1] = b
        tr = trajectory(initials, p, transient + keep, dt)[transient:, 0]
        m = (tr[1:-1] > tr[:-2]) & (tr[1:-1] > tr[2:])
        pts.append(np.column_stack([np.full(int(m.sum()), b), tr[1:-1][m]]))
    pts = np.vstack(pts)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(pts[:, 0], pts[:, 1], ".", color="k", markersize=0.6, alpha=0.5, markeredgewidth=0)
    ax.axvline(params[1], color="0.4", ls="--", lw=0.8, label=f"optimized b = {params[1]:.4f}")
    ax.set_xlabel("Control parameter b")
    ax.set_ylabel("Local maxima of x")
    ax.legend(fontsize=8, loc="upper right")
    _save(fig, out / "Fig9_bifurcation")


def fig10_11_attractors(params, initials, out: Path, n=30000):
    for p, fname in ((REFERENCE_PARAMS, "Fig10_pre_optimization_attractor"),
                     (params, "Fig11_post_optimization_attractor")):
        tr = trajectory(initials, p, n)[5000:]
        fig = plt.figure(figsize=(7.5, 6))
        ax = fig.add_subplot(221); ax.plot(tr[:, 0], tr[:, 2], lw=0.2, c="k"); ax.set_title("(a) x-z", fontsize=9)
        ax = fig.add_subplot(222); ax.plot(tr[:, 1], tr[:, 3], lw=0.2, c="k"); ax.set_title("(b) y-w", fontsize=9)
        ax = fig.add_subplot(223, projection="3d"); ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], lw=0.15, c="k")
        ax.set_title("(c) x-y-z", fontsize=9)
        ax = fig.add_subplot(224, projection="3d"); ax.plot(tr[:, 1], tr[:, 2], tr[:, 3], lw=0.15, c="k")
        ax.set_title("(d) y-z-w", fontsize=9)
        fig.tight_layout()
        _save(fig, out / fname)


def le_convergence(res, out: Path):
    for n, r in res.items():
        hist = r["_arrays"]["le_history"]
        t = np.array([h[0] for h in hist])
        le = np.array([h[1] for h in hist])
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for i in range(4):
            ax.plot(t, le[:, i], lw=0.8, label=f"$\\lambda_{i + 1}$")
        ax.set_ylim(min(-1, le[-1, 2] - 1), max(1.5, le[-1, 0] + 0.5))
        ax.set_xlabel("Integration time")
        ax.set_ylabel("Lyapunov exponent")
        ax.legend(fontsize=7, ncol=4)
        ax.set_title(n, fontsize=9)
        fig.savefig(out / f"S_LE_convergence_{n.replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
