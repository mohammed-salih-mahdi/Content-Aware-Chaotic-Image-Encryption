
from __future__ import annotations

import hashlib
import time
from typing import Dict

import numpy as np

from . import metrics as M
from .core import (CS_CONFIG, FINAL_LE_CONFIG, FilterBubbleCipher, HyperchaoticLorenz4D as HL,
                   derived_nonce, generate_streams, image_digest, invert_cipher_core)
from .nist_sts import run_sts

NOISE_LEVELS = (0.01, 0.05, 0.10)
CROP_LEVELS = (0.10, 0.25, 0.50)
TIMING_RUNS = 10


def stable_seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("/".join(parts).encode()).digest()[:8], "big")


def salt_and_pepper(cipher: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Set a `fraction` of the pixel positions to 0 or 255 (half each), all
    channels of a selected pixel together."""
    rng = np.random.default_rng(seed)
    out = cipher.copy()
    h, w = cipher.shape[:2]
    n = int(round(fraction * h * w))
    idx = rng.choice(h * w, size=n, replace=False)
    rows, cols = np.unravel_index(idx, (h, w))
    half = n // 2
    out[rows[:half], cols[:half]] = 255
    out[rows[half:], cols[half:]] = 0
    return out


def centre_crop(cipher: np.ndarray, fraction: float) -> np.ndarray:
    """Zero a centred rectangle covering `fraction` of the ciphertext AREA."""
    out = cipher.copy()
    h, w = cipher.shape[:2]
    s = np.sqrt(fraction)
    ch, cw = int(round(h * s)), int(round(w * s))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    out[y0:y0 + ch, x0:x0 + cw] = 0
    return out


def analyze_image(name: str, image: np.ndarray, root_key: bytes, keep_arrays: bool = True) -> Dict:
    cipher_sys = FilterBubbleCipher(root_key)
    nonce = derived_nonce(root_key, f"{name}/main")

    # Session: HKDF -> S0, CS seed -> Cuckoo Search (timed separately)
    session = cipher_sys.optimize_parameters(image, nonce)
    params, initials = session["params"], session["initials"]

    # Encryption / decryption timing (Table 9): mean of 10 runs each.
    enc_t, dec_t = [], []
    package = None
    for _ in range(TIMING_RUNS):
        t0 = time.perf_counter()
        package = cipher_sys.encrypt_with_session(image, nonce, session)
        enc_t.append(time.perf_counter() - t0)
    for _ in range(TIMING_RUNS):
        t0 = time.perf_counter()
        recovered = cipher_sys.decrypt(package)
        dec_t.append(time.perf_counter() - t0)
    cipher = package["cipher"]
    lossless = bool(np.array_equal(recovered, image))

    # Lyapunov spectrum (Table 11), divergence (Eq 17), Kaplan-Yorke (Eq 19)
    le, history = HL.lyapunov(params, initials, steps=FINAL_LE_CONFIG["steps"],
                              transient=FINAL_LE_CONFIG["transient"], dt=FINAL_LE_CONFIG["dt"],
                              qr_interval=FINAL_LE_CONFIG["qr_interval"], return_convergence=True)
    div = HL.divergence(params)


    ent = M.global_entropy(cipher)
    loc = M.local_entropy(cipher, seed=stable_seed(name, "local-entropy"))
    loc32 = M.local_entropy_32_pooled(cipher)
    corr_c = M.correlation(cipher)
    corr_p = M.correlation(image)
    polar_p, polar_c = M.polar_resultant(image), M.polar_resultant(cipher)

 
    meta = package["_metadata"]
    sens = {}
    wrong = None
    for label in ("eps_1e-12", "one_ulp"):
        s_bad = np.array(meta["initials"], dtype=float)
        s_bad[0] = s_bad[0] + 1e-12 if label == "eps_1e-12" else np.nextafter(s_bad[0], np.inf)
        dec_bad = invert_cipher_core(cipher, s_bad, meta["params"], meta["shape"])
        ps_, ss_ = M.psnr_ssim(recovered, dec_bad)
        n_, u_ = M.npcr_uaci(recovered, dec_bad)
        sens[label] = {"psnr_db": ps_, "ssim": ss_, "npcr_percent": n_, "uaci_percent": u_}
        if wrong is None:
            wrong = dec_bad
    bad_key = bytearray(root_key); bad_key[0] ^= 0x01
    try:
        cipher_sys.decrypt(package, key=bytes(bad_key)); key_status = "UNEXPECTED_SUCCESS"
    except ValueError as exc:
        key_status = f"REJECTED ({exc})"


    zero = np.zeros_like(image)
    cp1 = cipher_sys.encrypt(zero, nonce=derived_nonce(root_key, f"{name}/cp1"))
    cp2 = cipher_sys.encrypt(zero, nonce=derived_nonce(root_key, f"{name}/cp2"))
    cp_entropy = M.global_entropy(cp1["cipher"])["mean"]
    cp_differ = not np.array_equal(cp1["cipher"], cp2["cipher"])
    cp_npcr, _ = M.npcr_uaci(cp1["cipher"], cp2["cipher"])


    robust, robust_imgs = [], {}
    for kind, levels in (("Noise", NOISE_LEVELS), ("Cropping", CROP_LEVELS)):
        for lv in levels:
            if kind == "Noise":
                tampered = salt_and_pepper(cipher, lv, stable_seed(name, "noise", str(lv)))
            else:
                tampered = centre_crop(cipher, lv)
            rec = invert_cipher_core(tampered, meta["initials"], meta["params"], meta["shape"])
            ps, ss = M.psnr_ssim(image, rec)
            try:
                cipher_sys.decrypt({**package, "cipher": tampered}); auth = "ACCEPTED"
            except ValueError as exc:
                auth = f"REJECTED ({exc})"
            robust.append({"attack": kind, "level_percent": round(lv * 100), "psnr_db": ps, "ssim": ss,
                           "authenticated_protocol": auth})
            robust_imgs[f"{kind}_{round(lv * 100)}"] = rec


    _, ks_d = generate_streams(image.size, initials, params)
    nist = run_sts(np.unpackbits(ks_d))

    result = {
        "image": name,
        "shape": list(image.shape),
        "sha256_plaintext": image_digest(image).hex(),
        "nonce": nonce.hex(),
        "parameters": dict(zip("abcr", map(float, params))),
        "initial_state": [float(v) for v in initials],
        "cuckoo_search": {**{k: CS_CONFIG[k] for k in ("population", "pa", "generations", "levy_beta")},
                          "seed": int(session["cs_seed"]), "fitness": float(session["fitness"]),
                          "evaluations": int(session["cs_evaluations"]), "time_sec": float(session["cs_time"]),
                          "history": session["cs_history"]},
        "timing": {"encryption_sec_mean": float(np.mean(enc_t)), "encryption_sec_sd": float(np.std(enc_t)),
                   "decryption_sec_mean": float(np.mean(dec_t)), "decryption_sec_sd": float(np.std(dec_t)),
                   "cs_sec": float(session["cs_time"]), "runs": TIMING_RUNS},
        "lossless_decryption": lossless,
        "lyapunov": {"lambda": [float(v) for v in le], "sum": float(np.sum(le)), "divergence": div,
                     "abs_error": float(abs(np.sum(le) - div)), "kaplan_yorke": HL.kaplan_yorke(le),
                     "positive": int(np.sum(le > 0)), "hyperchaos": HL.hyperchaos_status(le)},
        "entropy_global": ent,
        "entropy_local": loc,
        "entropy_local_32x32_pooled": {"mean": loc32[0], "sd": loc32[1]},
        "correlation_cipher": corr_c,
        "correlation_plain": corr_p,
        "polar_R_plain": polar_p,
        "polar_R_cipher": polar_c,
        "differential": {"npcr_percent": npcr, "uaci_percent": uaci,
                         "modified_params": [float(v) for v in package_mod["_session"]["params"]]},
        "key_sensitivity": {**sens["eps_1e-12"], "perturbation": "x0 + 1e-12",
                            "one_ulp": sens["one_ulp"], "one_bit_root_key": key_status},
        "chosen_plaintext_probe": {"all_zero_cipher_entropy": cp_entropy,
                                   "fresh_nonce_gives_different_cipher": bool(cp_differ),
                                   "npcr_between_two_sessions": cp_npcr},
        "robustness": robust,
        "nist": nist,
    }
    if keep_arrays:
        result["_arrays"] = {"original": image, "cipher": cipher, "recovered": recovered, "wrong_key": wrong,
                             "le_history": history, **{f"robust_{k}": v for k, v in robust_imgs.items()}}
    return result


def parameter_sweep(params, initials, rel=0.10):
    """Each control parameter perturbed by +/-10 % (others fixed); full
    100k-step Lyapunov spectrum for each case."""
    rows = []
    for i, nm in enumerate("abcr"):
        for sign in (-1, +1):
            p = np.array(params, dtype=float)
            p[i] = p[i] * (1 + sign * rel)
            le = HL.lyapunov(p, initials, steps=FINAL_LE_CONFIG["steps"], transient=FINAL_LE_CONFIG["transient"],
                             dt=FINAL_LE_CONFIG["dt"], qr_interval=FINAL_LE_CONFIG["qr_interval"])
            rows.append({"parameter": nm, "change": f"{sign * rel * 100:+.0f}%", "value": float(p[i]),
                         "lambda1": None if le is None else float(le[0]),
                         "lambda2": None if le is None else float(le[1]),
                         "lambda3": None if le is None else float(le[2]),
                         "lambda4": None if le is None else float(le[3]),
                         "divergence": HL.divergence(p),
                         "hyperchaos": "DIVERGED" if le is None else HL.hyperchaos_status(le)})
    return rows
