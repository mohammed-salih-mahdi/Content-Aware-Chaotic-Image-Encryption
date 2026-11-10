
"""
Round-2 Research Pipeline
Dynamic Content-Aware Image Encryption Using Filter-Bubble Logic
and Cuckoo Search-Optimized 4D Hyperchaotic Lorenz System.

This script replaces the old 5-parameter dynamical model and produces
reproducible, table-ready results. It deliberately does NOT fabricate
NIST results: NIST is reported as NOT RUN unless a compatible external
implementation is available.

Core 4-D system:
    x' = a(y-x) + w
    y' = cx - y - xz
    z' = xy - bz
    w' = -yz + rw

Secret material:
    K = independent 256-bit root key
    N = fresh 96-bit nonce per encryption
    H = SHA-256(plaintext), non-secret content-binding digest
    HKDF-SHA256 derives initial state and CS seed

Protocol note:
    AES-256-GCM protects the parameter/initial-state metadata; HMAC-SHA256
    authenticates the nonce, protected metadata, and ciphertext. The root
    secret is independent of the plaintext hash.

Outputs:
    results/raw/*.json
    results/tables/*.csv
    results/PLOS_Round2_Results.xlsx
    results/figures/*
    results/FINAL_REPORT.txt
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception:
    AESGCM = None

try:
    from openpyxl import Workbook
except Exception:
    Workbook = None


# ============================================================
# Configuration
# ============================================================

ROOT = Path("results")
RAW_DIR = ROOT / "raw"
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"

CS_CONFIG = {
    "population": 10,
    "pa": 0.25,
    "generations": 5,
    "levy_beta": 1.5,
    "quick_le_steps": 2500,
    "quick_transient": 1000,
    "dt": 0.001,
}

FINAL_LE_CONFIG = {
    "steps": 100000,
    "transient": 5000,
    "dt": 0.001,
    "qr_interval": 10,
    "convergence_blocks": 10,
}

# Established operating region around the standard reference point.
LB = np.array([9.0, 2.5, 27.0, -1.52], dtype=float)
UB = np.array([11.0, 3.0, 29.0, -0.06], dtype=float)

REFERENCE_PARAMS = np.array([10.0, 8.0 / 3.0, 28.0, -1.0], dtype=float)
DEFAULT_INITIALS = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)

# For reproducible research runs, a fixed key may be supplied through
# environment variable RESEARCH_KEY_HEX. Otherwise a new random key is used.
KEY_ENV = os.environ.get("RESEARCH_KEY_HEX", "").strip()
FIGURE_DATA = {}


# ============================================================
# Utility functions
# ============================================================

def ensure_dirs() -> None:
    for d in (ROOT, RAW_DIR, TABLE_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    """Minimal HKDF-SHA256 implementation (RFC 5869 style)."""
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    out = b""
    t = b""
    counter = 1
    while len(out) < length:
        t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha256).digest()
        out += t
        counter += 1
    return out[:length]


def bytes_to_initial_state(material: bytes) -> np.ndarray:
    vals = []
    for i in range(4):
        u = int.from_bytes(material[i * 8:(i + 1) * 8], "big")
        # Map into a compact non-equilibrium interval.
        vals.append(0.1 + (u / 2**64) * 0.8)
    return np.asarray(vals, dtype=float)


def load_root_key() -> bytes:
    if KEY_ENV:
        try:
            key = bytes.fromhex(KEY_ENV)
        except ValueError as exc:
            raise ValueError("RESEARCH_KEY_HEX must contain hexadecimal bytes.") from exc
        if len(key) != 32:
            raise ValueError("RESEARCH_KEY_HEX must represent exactly 32 bytes (256 bits).")
        return key
    return os.urandom(32)


def nonce_for_protocol() -> bytes:
    return os.urandom(12)


def nonce_for_controlled_experiment(root_key: bytes, label: str) -> bytes:
    # Only for controlled experiments (e.g. NPCR). In the real protocol,
    # use a fresh random nonce.
    return sha256_bytes(root_key + label.encode("utf-8"))[:12]


def image_bytes(image: np.ndarray) -> bytes:
    return np.ascontiguousarray(image).tobytes()


def image_digest(image: np.ndarray) -> bytes:
    return sha256_bytes(image_bytes(image))


def json_bytes(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def safe_float(x) -> float:
    return float(np.asarray(x).item())


# ============================================================
# 4-D Hyperchaotic Lorenz system
# ============================================================

class HyperchaoticLorenz4D:
    @staticmethod
    def f(state: np.ndarray, p: np.ndarray) -> np.ndarray:
        x, y, z, w = state
        a, b, c, r = p
        return np.array([
            a * (y - x) + w,
            c * x - y - x * z,
            x * y - b * z,
            -y * z + r * w
        ], dtype=float)

    @staticmethod
    def jacobian(state: np.ndarray, p: np.ndarray) -> np.ndarray:
        x, y, z, w = state
        a, b, c, r = p
        return np.array([
            [-a, a, 0.0, 1.0],
            [c - z, -1.0, -x, 0.0],
            [y, x, -b, 0.0],
            [0.0, -z, -y, r]
        ], dtype=float)

    @staticmethod
    def divergence(p: np.ndarray) -> float:
        a, b, c, r = p
        return float(-a - 1.0 - b + r)

    @classmethod
    def rk4_state(cls, state: np.ndarray, p: np.ndarray, dt: float) -> np.ndarray:
        k1 = cls.f(state, p)
        k2 = cls.f(state + 0.5 * dt * k1, p)
        k3 = cls.f(state + 0.5 * dt * k2, p)
        k4 = cls.f(state + dt * k3, p)
        return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0

    @classmethod
    def rk4_variational(
        cls, state: np.ndarray, V: np.ndarray, p: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        def rhs(x, Y):
            return cls.f(x, p), cls.jacobian(x, p) @ Y

        k1x, k1v = rhs(state, V)
        k2x, k2v = rhs(state + 0.5 * dt * k1x, V + 0.5 * dt * k1v)
        k3x, k3v = rhs(state + 0.5 * dt * k2x, V + 0.5 * dt * k2v)
        k4x, k4v = rhs(state + dt * k3x, V + dt * k3v)

        x_new = state + dt * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        V_new = V + dt * (k1v + 2*k2v + 2*k3v + k4v) / 6.0
        return x_new, V_new

    @classmethod
    def lyapunov(
        cls,
        p: np.ndarray,
        initials: np.ndarray,
        steps: int,
        transient: int,
        dt: float,
        qr_interval: int = 10,
        return_convergence: bool = False,
    ):
        x = np.asarray(initials, dtype=float).copy()

        for _ in range(transient):
            x = cls.rk4_state(x, p, dt)
            if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e6:
                return None

        V = np.eye(4, dtype=float)
        accum = np.zeros(4, dtype=float)
        total_time = 0.0
        history = []

        for step in range(1, steps + 1):
            x, V = cls.rk4_variational(x, V, p, dt)

            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(V)):
                return None
            if np.linalg.norm(x) > 1e6:
                return None

            if step % qr_interval == 0:
                Q, R = np.linalg.qr(V)
                diag = np.abs(np.diag(R))
                if np.any(diag <= 0) or not np.all(np.isfinite(diag)):
                    return None
                accum += np.log(diag)
                total_time += qr_interval * dt
                V = Q

                if return_convergence:
                    history.append((total_time, accum / total_time))

        le = accum / total_time
        return (le, history) if return_convergence else le

    @classmethod
    def kaplan_yorke(cls, le: np.ndarray) -> float:
        le = np.sort(np.asarray(le, dtype=float))[::-1]
        cumulative = np.cumsum(le)
        j = int(np.sum(cumulative >= 0))

        if j == 0:
            return 0.0
        if j >= len(le):
            return float(len(le))

        return float(j + cumulative[j - 1] / abs(le[j]))

    @classmethod
    def hyperchaos_status(cls, le: np.ndarray) -> str:
        return "PASS" if np.sum(np.asarray(le) > 0.0) >= 2 else "FAIL"


# ============================================================
# Cuckoo Search
# ============================================================

class CuckooSearch:
    def __init__(
        self,
        initials: np.ndarray,
        seed: int,
        pop_size: int = 10,
        pa: float = 0.25,
        max_iter: int = 5,
        levy_beta: float = 1.5,
        quick_steps: int = 2500,
        quick_transient: int = 1000,
        dt: float = 0.001,
    ):
        self.initials = np.asarray(initials, dtype=float)
        self.rng = np.random.default_rng(int(seed))
        self.pop_size = int(pop_size)
        self.pa = float(pa)
        self.max_iter = int(max_iter)
        self.levy_beta = float(levy_beta)
        self.quick_steps = int(quick_steps)
        self.quick_transient = int(quick_transient)
        self.dt = float(dt)
        self.lb = LB.copy()
        self.ub = UB.copy()
        self.history = []

    @staticmethod
    def stream_stats(p: np.ndarray, initials: np.ndarray, n: int = 1000) -> Tuple[float, float]:
        state = initials.copy()
        vals = np.empty(n, dtype=float)
        for i in range(n):
            state = HyperchaoticLorenz4D.rk4_state(state, p, 0.001)
            if not np.all(np.isfinite(state)):
                return 0.0, 1.0
            vals[i] = state[0]

        # Explicit signed integer conversion avoids the uint64 -> int64
        # safe-casting failure in np.bincount().
        q = np.floor(np.abs(vals) * 1e6).astype(np.int64)
        q = np.mod(q, 256)
        counts = np.bincount(q, minlength=256).astype(np.float64)
        prob = counts[counts > 0] / float(n)
        entropy = -np.sum(prob * np.log2(prob))

        if n > 2 and np.std(q[:-1]) > 0 and np.std(q[1:]) > 0:
            ac = abs(float(np.corrcoef(q[:-1], q[1:])[0, 1]))
        else:
            ac = 1.0

        return float(entropy), float(ac)

    def levy(self, size: int) -> np.ndarray:
        beta = self.levy_beta
        sigma = (
            math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
            / (
                math.gamma((1 + beta) / 2)
                * beta
                * 2 ** ((beta - 1) / 2)
            )
        ) ** (1 / beta)

        u = self.rng.normal(0, sigma, size)
        v = self.rng.normal(0, 1, size)
        return u / np.maximum(np.abs(v), 1e-12) ** (1 / beta)

    def fitness(self, p: np.ndarray) -> float:
        p = np.asarray(p, dtype=float)
        divergence = HyperchaoticLorenz4D.divergence(p)

        # Dissipativity is required.
        if divergence >= 0:
            return -1e9

        result = HyperchaoticLorenz4D.lyapunov(
            p,
            self.initials,
            steps=self.quick_steps,
            transient=self.quick_transient,
            dt=self.dt,
            qr_interval=10,
        )
        if result is None:
            return -1e9

        le = result
        positive = int(np.sum(le > 0))
        if positive < 2:
            return -1e6 + float(np.sum(le))

        entropy, autocorr = self.stream_stats(p, self.initials)

        # Fitness rewards two positive LEs, high stream entropy and low
        # adjacent dependence. This is a heuristic local optimization
        # criterion, not a proof of global optimality.
        return float(le[0] + le[1] + 0.05 * entropy - 0.5 * autocorr)

    def optimize(self, verbose=True) -> Tuple[np.ndarray, float]:
        nests = []
        fitness = []

        attempts = 0
        max_attempts = max(100, self.pop_size * 30)

        while len(nests) < self.pop_size and attempts < max_attempts:
            attempts += 1
            p = self.rng.uniform(self.lb, self.ub)
            f = self.fitness(p)
            if f > -1e6:
                nests.append(p)
                fitness.append(f)

        if len(nests) < self.pop_size:
            raise RuntimeError("Unable to obtain enough feasible hyperchaotic CS candidates.")

        nests = np.asarray(nests, dtype=float)
        fitness = np.asarray(fitness, dtype=float)

        best_idx = int(np.argmax(fitness))
        best = nests[best_idx].copy()
        best_f = float(fitness[best_idx])

        span = self.ub - self.lb

        for gen in range(1, self.max_iter + 1):
            for i in range(self.pop_size):
                step = 0.01 * self.levy(4) * span
                candidate = np.clip(nests[i] + step, self.lb, self.ub)
                f = self.fitness(candidate)

                if f > fitness[i]:
                    nests[i] = candidate
                    fitness[i] = f

            n_abandon = max(1, int(self.pa * self.pop_size))
            worst = np.argsort(fitness)[:n_abandon]

            for i in worst:
                for _ in range(20):
                    candidate = self.rng.uniform(self.lb, self.ub)
                    f = self.fitness(candidate)
                    if f > -1e6:
                        nests[i] = candidate
                        fitness[i] = f
                        break

            idx = int(np.argmax(fitness))
            if fitness[idx] > best_f:
                best = nests[idx].copy()
                best_f = float(fitness[idx])

            self.history.append({
                "generation": gen,
                "best_fitness": best_f
            })

            if verbose:
                print(f"    CS generation {gen}/{self.max_iter}: best fitness={best_f:.8f}")

        return best, best_f


# ============================================================
# Image encryption system
# ============================================================

class ResearchEncryptionSystem:
    def __init__(
        self,
        image_path: str | Path,
        root_key: bytes,
        output_root: Path = ROOT,
    ):
        self.image_path = Path(image_path)
        self.root_key = root_key
        self.output_root = Path(output_root)

        bgr = cv2.imread(str(self.image_path), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {self.image_path}")

        if bgr.ndim == 2:
            self.image = bgr
            self.channels = 1
        elif bgr.ndim == 3:
            if bgr.shape[2] == 4:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
            self.image = bgr
            self.channels = 3
        else:
            raise ValueError("Unsupported image shape.")

        self.rows, self.cols = self.image.shape[:2]
        self.name = self.image_path.stem

    def derive_context(self, image: np.ndarray, nonce: bytes):
        digest = image_digest(image)
        state_material = hkdf_sha256(
            self.root_key,
            salt=nonce,
            info=b"FilterBubble/initial-state/" + digest,
            length=32,
        )
        initials = bytes_to_initial_state(state_material)

        cs_material = hkdf_sha256(
            self.root_key,
            salt=nonce,
            info=b"FilterBubble/CS-seed/" + digest,
            length=8,
        )
        cs_seed = int.from_bytes(cs_material, "big") % (2**63 - 1)

        return digest, initials, cs_seed

    def optimize_parameters(self, image: np.ndarray, nonce: bytes):
        digest, initials, cs_seed = self.derive_context(image, nonce)

        cs = CuckooSearch(
            initials,
            seed=cs_seed,
            pop_size=CS_CONFIG["population"],
            pa=CS_CONFIG["pa"],
            max_iter=CS_CONFIG["generations"],
            levy_beta=CS_CONFIG["levy_beta"],
            quick_steps=CS_CONFIG["quick_le_steps"],
            quick_transient=CS_CONFIG["quick_transient"],
            dt=CS_CONFIG["dt"],
        )

        start = time.perf_counter()
        params, fitness = cs.optimize(verbose=False)
        elapsed = time.perf_counter() - start

        return digest, initials, cs_seed, params, fitness, elapsed, cs.history

    @staticmethod
    def generate_streams(size: int, initials: np.ndarray, params: np.ndarray, dt: float = 0.001):
        state = initials.copy()
        perm_stream = np.empty(size, dtype=float)
        diff_stream = np.empty(size, dtype=np.uint8)

        for i in range(size):
            state = HyperchaoticLorenz4D.rk4_state(state, params, dt)
            if not np.all(np.isfinite(state)):
                raise FloatingPointError("Chaotic trajectory became non-finite.")

            perm_stream[i] = state[0]
            value = int(np.floor(abs(state[0]) * 1e6) + np.floor(abs(state[2]) * 1e4))
            diff_stream[i] = value % 256

        return perm_stream, diff_stream

    def encrypt(
        self,
        image: np.ndarray,
        nonce: Optional[bytes] = None,
        params: Optional[np.ndarray] = None,
        initials: Optional[np.ndarray] = None,
        optimize: bool = True,
    ):
        if nonce is None:
            nonce = nonce_for_protocol()

        if optimize:
            digest, initials, cs_seed, params, fitness, cs_time, cs_history = \
                self.optimize_parameters(image, nonce)
        else:
            digest = image_digest(image)
            if initials is None or params is None:
                raise ValueError("Non-optimization mode requires initials and params.")
            cs_seed = None
            fitness = None
            cs_time = 0.0
            cs_history = []

        shape = image.shape
        flat = np.ascontiguousarray(image).reshape(-1)
        perm, diff = self.generate_streams(flat.size, initials, params)

        order = np.argsort(perm, kind="mergesort")
        permuted = flat[order]
        cipher_flat = np.bitwise_xor(permuted, diff)
        cipher = cipher_flat.reshape(shape)

        metadata = {
            "version": "Round2-4DHL-v1",
            "image": self.name,
            "shape": list(shape),
            "channels": int(self.channels),
            "sha256": digest.hex(),
            "params": [float(x) for x in params],
            "initials": [float(x) for x in initials],
            "cs_seed": None if cs_seed is None else int(cs_seed),
            "fitness": None if fitness is None else float(fitness),
        }

        metadata_bytes = json_bytes(metadata)

        # The chaotic parameters and initial state are not transmitted in
        # plaintext. They are protected with AES-256-GCM. The HMAC then
        # authenticates nonce + protected metadata + ciphertext as a single
        # protocol package. The plaintext SHA-256 digest remains inside the
        # protected metadata and is rechecked after decryption.
        if AESGCM is None:
            raise RuntimeError(
                "cryptography is required: AES-256-GCM must be available "
                "for the protected metadata protocol."
            )

        meta_nonce = sha256_bytes(nonce + b"Round2-4DHL metadata v1")[:12]
        protected_metadata = AESGCM(self.root_key).encrypt(
            meta_nonce,
            metadata_bytes,
            nonce,
        )

        tag = hmac.new(
            self.root_key,
            nonce + protected_metadata + image_bytes(cipher),
            hashlib.sha256,
        ).hexdigest()

        package = {
            "cipher": cipher,
            "nonce": nonce,
            "protected_metadata": protected_metadata,
            "tag": tag,
            # Internal/debug copy used only for figure generation and local
            # experiments. A transmitted package contains only the protected
            # metadata, not these plaintext fields.
            "metadata": metadata,
            "params": params,
            "initials": initials,
            "digest": digest,
            # Internal/debug fields used by the analysis tables. These values
            # are also stored inside protected metadata, but are never sent
            # as plaintext protocol metadata.
            "cs_seed": cs_seed,
            "fitness": fitness,
            "cs_time": cs_time,
            "cs_history": cs_history,
        }
        return package

    def decrypt(self, package: dict, key: Optional[bytes] = None) -> np.ndarray:
        key = self.root_key if key is None else key

        cipher = package["cipher"]
        nonce = package["nonce"]
        protected_metadata = package["protected_metadata"]
        tag = package["tag"]

        expected = hmac.new(
            key,
            nonce + protected_metadata + image_bytes(cipher),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(expected, tag):
            raise ValueError("AUTHENTICATION_FAILED")

        if AESGCM is None:
            raise RuntimeError(
                "cryptography is required: AES-256-GCM must be available "
                "to recover protected metadata."
            )

        meta_nonce = sha256_bytes(nonce + b"Round2-4DHL metadata v1")[:12]
        try:
            metadata_bytes = AESGCM(key).decrypt(
                meta_nonce,
                protected_metadata,
                nonce,
            )
            metadata = json.loads(metadata_bytes.decode("utf-8"))
        except Exception as exc:
            raise ValueError("METADATA_AUTHENTICATION_FAILED") from exc

        params = np.asarray(metadata["params"], dtype=float)
        initials = np.asarray(metadata["initials"], dtype=float)

        flat = np.ascontiguousarray(cipher).reshape(-1)
        perm, diff = self.generate_streams(flat.size, initials, params)

        diffused_back = np.bitwise_xor(flat, diff)
        order = np.argsort(perm, kind="mergesort")

        original = np.empty_like(diffused_back)
        original[order] = diffused_back

        recovered = original.reshape(tuple(metadata["shape"]))

        recovered_digest = image_digest(recovered).hex()
        if recovered_digest != metadata["sha256"]:
            raise ValueError("PLAINTEXT_DIGEST_MISMATCH")

        # Enforce package shape/version consistency after authenticated
        # metadata recovery. This prevents accidental reshaping of a valid
        # byte stream under inconsistent metadata.
        if list(recovered.shape) != list(metadata["shape"]):
            raise ValueError("METADATA_SHAPE_MISMATCH")
        if metadata.get("version") != "Round2-4DHL-v1":
            raise ValueError("UNSUPPORTED_PROTOCOL_VERSION")

        return recovered


# ============================================================
# Security/statistical metrics
# ============================================================

def entropy_channels(image: np.ndarray) -> Dict[str, float]:
    if image.ndim == 2:
        channels = [image]
    else:
        channels = [image[:, :, i] for i in range(image.shape[2])]

    result = {}
    for i, ch in enumerate(channels):
        hist = np.bincount(ch.reshape(-1).astype(np.int64), minlength=256)
        prob = hist[hist > 0].astype(np.float64) / ch.size
        result[f"channel_{i+1}"] = float(-np.sum(prob * np.log2(prob)))

    result["mean"] = float(np.mean(list(result.values())))
    return result


def local_entropy(image: np.ndarray, block: int = 16) -> Tuple[float, float]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    vals = []
    h, w = gray.shape
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            patch = gray[y:y+block, x:x+block]
            hist = np.bincount(patch.reshape(-1).astype(np.int64), minlength=256)
            prob = hist[hist > 0].astype(np.float64) / patch.size
            vals.append(-np.sum(prob * np.log2(prob)))

    if not vals:
        return float("nan"), float("nan")

    return float(np.mean(vals)), float(np.std(vals, ddof=0))


def correlation_values(image: np.ndarray) -> Dict[str, float]:
    if image.ndim == 2:
        image = image[:, :, None]

    h, w, c = image.shape
    out = {}

    for ch in range(c):
        a = image[:, :, ch].astype(np.float64)

        pairs = {
            "H": (a[:, :-1].ravel(), a[:, 1:].ravel()),
            "V": (a[:-1, :].ravel(), a[1:, :].ravel()),
            "D": (a[:-1, :-1].ravel(), a[1:, 1:].ravel()),
        }

        for direction, (x, y) in pairs.items():
            if np.std(x) == 0 or np.std(y) == 0:
                corr = 1.0
            else:
                corr = float(np.corrcoef(x, y)[0, 1])
            out[f"C{ch+1}_{direction}"] = corr

    out["max_abs"] = float(max(abs(v) for v in out.values()))
    out["mean_abs"] = float(np.mean([abs(v) for v in out.values()]))
    return out


def npcr_uaci(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    if a.shape != b.shape:
        raise ValueError("NPCR/UACI images must have identical shape.")

    aa = a.astype(np.int16)
    bb = b.astype(np.int16)

    diff = np.not_equal(aa, bb)
    npcr = float(np.mean(diff) * 100.0)
    uaci = float(np.mean(np.abs(aa - bb)) / 255.0 * 100.0)
    return npcr, uaci


def psnr_ssim(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    psnr = float("inf") if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse)))

    if a.ndim == 2:
        s = float(ssim(a, b, data_range=255))
    else:
        vals = [
            ssim(a[:, :, i], b[:, :, i], data_range=255)
            for i in range(a.shape[2])
        ]
        s = float(np.mean(vals))
    return psnr, s


def initial_state_perturbation(
    system: ResearchEncryptionSystem,
    package: dict,
    epsilon: float = 1e-12,
) -> Dict[str, float]:
    cipher = package["cipher"]
    meta = package["metadata"]

    params = np.asarray(meta["params"], dtype=float)
    initials = np.asarray(meta["initials"], dtype=float)

    flat = np.ascontiguousarray(cipher).reshape(-1)
    perm, diff = system.generate_streams(flat.size, initials, params)

    perturbed = initials.copy()
    perturbed[0] += epsilon
    perm2, diff2 = system.generate_streams(flat.size, perturbed, params)

    dec1 = np.empty_like(flat)
    dec1[np.argsort(perm, kind="mergesort")] = np.bitwise_xor(flat, diff)

    dec2 = np.empty_like(flat)
    dec2[np.argsort(perm2, kind="mergesort")] = np.bitwise_xor(flat, diff2)

    a = dec1.reshape(cipher.shape)
    b = dec2.reshape(cipher.shape)

    p, s = psnr_ssim(a, b)
    n, u = npcr_uaci(a, b)

    return {
        "epsilon": epsilon,
        "psnr_db": p,
        "ssim": s,
        "npcr_percent": n,
        "uaci_percent": u,
    }


# ============================================================
# Figures
# ============================================================

def save_histogram(image: np.ndarray, path: Path, title: str):
    plt.figure(figsize=(9, 5))
    if image.ndim == 2:
        hist = np.bincount(image.reshape(-1).astype(np.int64), minlength=256)
        plt.plot(hist, label="Gray")
    else:
        labels = ["B", "G", "R"]
        for i, label in enumerate(labels):
            hist = np.bincount(image[:, :, i].reshape(-1).astype(np.int64), minlength=256)
            plt.plot(hist, label=label)
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()


def save_attractor(
    system: ResearchEncryptionSystem,
    initials: np.ndarray,
    params: np.ndarray,
    path: Path,
):
    state = initials.copy()
    traj = np.empty((30000, 4), dtype=float)
    for i in range(len(traj)):
        state = HyperchaoticLorenz4D.rk4_state(state, params, 0.001)
        traj[i] = state

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(221)
    ax.plot(traj[:, 0], traj[:, 2], linewidth=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("x-z")

    ax = fig.add_subplot(222)
    ax.plot(traj[:, 1], traj[:, 3], linewidth=0.5)
    ax.set_xlabel("y")
    ax.set_ylabel("w")
    ax.set_title("y-w")

    ax = fig.add_subplot(223, projection="3d")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("x-y-z")

    ax = fig.add_subplot(224, projection="3d")
    ax.plot(traj[:, 1], traj[:, 2], traj[:, 3], linewidth=0.3)
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_zlabel("w")
    ax.set_title("y-z-w")

    plt.tight_layout()
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()


def save_le_convergence(
    history: List[Tuple[float, np.ndarray]],
    path: Path,
):
    if not history:
        return

    t = np.asarray([h[0] for h in history])
    le = np.asarray([h[1] for h in history])

    plt.figure(figsize=(9, 5))
    for i in range(4):
        plt.plot(t, le[:, i], label=f"$\\lambda_{i+1}$")
    plt.xlabel("Integration time")
    plt.ylabel("Lyapunov exponent")
    plt.title("Lyapunov-exponent convergence")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()


def _load_for_figure(path: Path, gray=False):
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(path)
    if not gray and img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _savefig(path: Path):
    plt.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()


def save_figure1_framework(path: Path):
    """Figure 1: revised, operational protocol architecture."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    boxes = [
        (0.03, 0.68, 0.16, 0.16, "Plaintext image I\n(bytes)"),
        (0.25, 0.68, 0.16, 0.16, "SHA-256\nH = SHA256(I)"),
        (0.47, 0.68, 0.18, 0.16, "HKDF-SHA256\nK + nonce + H\n→ initial state + CS seed"),
        (0.71, 0.68, 0.24, 0.16, "Constrained Cuckoo Search\nselect [a,b,c,r] in admissible\nhyperchaotic region"),
        (0.18, 0.38, 0.24, 0.16, "4-D hyperchaotic Lorenz\nRK4 trajectory generation\npermutation + diffusion streams"),
        (0.49, 0.38, 0.20, 0.16, "Image encryption\npermutation + XOR diffusion"),
        (0.76, 0.38, 0.19, 0.16, "AEAD/HMAC-protected\nmetadata + ciphertext\n+ nonce"),
        (0.37, 0.10, 0.26, 0.16, "Receiver: authenticate → recover\nprotected parameters/initials →\nregenerate streams → decrypt → verify H"),
    ]
    for x,y,w,h,text in boxes:
        ax.add_patch(plt.Rectangle((x,y),w,h,fill=False,linewidth=1.8))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=10)
    arrows=[((.19,.76),(.25,.76)),((.41,.76),(.47,.76)),((.65,.76),(.71,.76)),
            ((.83,.68),(.83,.54)),((.71,.46),(.69,.46)),((.49,.46),(.42,.46)),
            ((.76,.46),(.63,.18)),((.50,.38),(.50,.26))]
    for a,b in arrows:
        ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",lw=1.5))
    ax.text(.50,.94,"Revised Dynamic Content-Aware Image Encryption Framework",ha="center",fontsize=14,fontweight="bold")
    ax.text(.05,.04,"K: independent 256-bit secret; nonce: fresh 96-bit public value; H is content-binding and non-secret.",fontsize=9)
    _savefig(path)


def save_figure2_visual(results_by_name: Dict[str, Dict], path: Path):
    """Figure 2: original / encrypted / decrypted for all available benchmark images."""
    names = [n for n in ["Baboon","Peppers","Car","Brain","Chest","Lung","Two Macaws"] if n in results_by_name]
    if not names:
        return
    fig, axes = plt.subplots(len(names), 3, figsize=(9, 2.2*len(names)))
    if len(names)==1: axes=np.asarray([axes])
    for i,name in enumerate(names):
        r=results_by_name[name]
        for j,(key,title) in enumerate([("original","Original"),("cipher","Encrypted"),("recovered","Decrypted")]):
            img=r[key]
            if img.ndim==3: img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            axes[i,j].imshow(img,cmap="gray" if img.ndim==2 else None)
            axes[i,j].axis("off")
            if i==0: axes[i,j].set_title(title,fontsize=11)
        axes[i,0].set_ylabel(name,fontsize=10)
    fig.suptitle("Visual results: original, encrypted and decrypted images",fontsize=14)
    plt.tight_layout(rect=[0,0,1,.98]); _savefig(path)


def save_figure3_histograms(results_by_name: Dict[str, Dict], path: Path):
    names=[n for n in ["Baboon","Peppers"] if n in results_by_name]
    if not names:return
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
    for i,n in enumerate(names):
        for j,key in enumerate(["original","cipher"]):
            img=results_by_name[n][key]
            ax=axes[i,j]
            channels=[img] if img.ndim==2 else [img[:,:,k] for k in range(img.shape[2])]
            labels=["Gray"] if img.ndim==2 else ["B","G","R"]
            for ch,label in zip(channels,labels):
                hist=np.bincount(ch.reshape(-1).astype(np.int64),minlength=256)
                ax.plot(hist,label=label,linewidth=.9)
            ax.set_title(f"{n} — {'Original' if j==0 else 'Encrypted'}")
            ax.set_xlabel("Pixel value"); ax.set_ylabel("Frequency")
            ax.legend(fontsize=8)
    fig.suptitle("Figure 3. Histogram distribution analysis",fontsize=14)
    plt.tight_layout(rect=[0,0,1,.96]); _savefig(path)


def polar_resultant(image):
    chans=[image] if image.ndim==2 else [image[:,:,k] for k in range(image.shape[2])]
    out=[]
    for ch in chans:
        hist=np.bincount(ch.reshape(-1).astype(np.int64),minlength=256).astype(float)
        theta=2*np.pi*np.arange(256)/256
        p=hist/max(hist.sum(),1)
        C=float(np.sum(p*np.cos(theta))); S=float(np.sum(p*np.sin(theta)))
        out.append((hist,theta,float(np.hypot(C,S))))
    return out


def save_figure4_polar(results_by_name: Dict[str, Dict], path: Path):
    if "Baboon" not in results_by_name:return
    fig=plt.figure(figsize=(12,7))
    for row,key,title in [(0,"original","Original"),(1,"cipher","Encrypted")]:
        vals=polar_resultant(results_by_name["Baboon"][key])
        for col,(hist,theta,R) in enumerate(vals):
            ax=fig.add_subplot(2,3,row*3+col+1,projection="polar")
            ax.plot(theta,hist/max(hist.max(),1))
            ax.set_title(f"{title} — {'BGR'[col]}  (R={R:.4f})",fontsize=9)
            ax.set_yticklabels([])
    fig.suptitle("Figure 4. Polar histogram comparison for Baboon",fontsize=14)
    plt.tight_layout(rect=[0,0,1,.95]); _savefig(path)


def _neighbor_pairs(ch, direction):
    if direction=="H": return ch[:,:-1].ravel(),ch[:,1:].ravel()
    if direction=="V": return ch[:-1,:].ravel(),ch[1:,:].ravel()
    return ch[:-1,:-1].ravel(),ch[1:,1:].ravel()


def save_figure5_3d_corr(results_by_name: Dict[str, Dict], path: Path):
    if "Baboon" not in results_by_name:return
    fig=plt.figure(figsize=(14,8))
    r=results_by_name["Baboon"]
    for idx,(key,title) in enumerate([("original","Original"),("cipher","Encrypted")]):
        img=r[key]
        chans=[img] if img.ndim==2 else [img[:,:,k] for k in range(3)]
        for c,ch in enumerate(chans):
            ax=fig.add_subplot(2,3,idx*3+c+1,projection="3d")
            x,y=_neighbor_pairs(ch.astype(float),["D","H","V"][c])
            n=min(12000,x.size); sel=np.linspace(0,x.size-1,n,dtype=int)
            ax.scatter(x[sel],y[sel],np.arange(n),s=1,alpha=.25)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("sample")
            direction = ["Diagonal", "Horizontal", "Vertical"][c]
            ax.set_title(f"{title} — {'BGR'[c]} / {direction}", fontsize=8)
    fig.suptitle("Figure 5. 3D RGB correlation scatter plots for Baboon",fontsize=14)
    plt.tight_layout(rect=[0,0,1,.95]); _savefig(path)


def save_figure6_key_sensitivity(results_by_name: Dict[str, Dict], path: Path):
    if "Baboon" not in results_by_name:return
    r=results_by_name["Baboon"]
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    axes[0].imshow(cv2.cvtColor(r["original"],cv2.COLOR_BGR2RGB) if r["original"].ndim==3 else r["original"],cmap="gray")
    axes[0].set_title("(a) Authorized decryption\nCorrect 256-bit key")
    axes[0].axis("off")
    axes[1].text(.5,.60,"AUTHENTICATION FAILED",ha="center",va="center",fontsize=16,fontweight="bold")
    axes[1].text(.5,.38,"One-bit modified root key\n→ ciphertext is rejected before decryption",ha="center",va="center",fontsize=11)
    axes[1].set_title("(b) Unauthorized decryption")
    axes[1].axis("off")
    fig.suptitle("Figure 6. Key authentication and sensitivity",fontsize=14)
    plt.tight_layout(rect=[0,0,1,.95]); _savefig(path)


def _tamper_cipher(cipher, mode, level):
    out=cipher.copy()
    if mode=="noise":
        rng=np.random.default_rng(20260919+int(level*100))
        mask=rng.random(out.shape)<level
        out[mask]=rng.integers(0,256,size=int(mask.sum()),dtype=np.uint8)
    else:
        h,w=out.shape[:2]
        frac=level
        ch=int(w*frac); cv=int(h*frac)
        y0=(h-cv)//2; x0=(w-ch)//2
        out[y0:y0+cv,x0:x0+ch]=0
    return out


def _unauthenticated_recovery(system, package, tampered):
    meta=package["metadata"]; flat=np.ascontiguousarray(tampered).reshape(-1)
    perm,diff=system.generate_streams(flat.size,np.asarray(meta["initials"]),np.asarray(meta["params"]))
    diffused=np.bitwise_xor(flat,diff)
    order=np.argsort(perm,kind="mergesort")
    original=np.empty_like(diffused); original[order]=diffused
    return original.reshape(tuple(meta["shape"]))


def save_figure7_noise(results_by_name: Dict[str, Dict], path: Path):
    if "Baboon" not in results_by_name:return
    r=results_by_name["Baboon"]; system=r["system"]; package=r["package"]
    levels=[.01,.05,.10]
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,lv in zip(axes,levels):
        tam=_tamper_cipher(package["cipher"],"noise",lv)
        rec=_unauthenticated_recovery(system,package,tam)
        ax.imshow(cv2.cvtColor(rec,cv2.COLOR_BGR2RGB) if rec.ndim==3 else rec,cmap="gray")
        ax.set_title(f"Salt-and-pepper diagnostic {lv*100:.0f}%")
        ax.axis("off")
    fig.suptitle("Figure 7. Recovery after ciphertext noise (authentication bypassed for diagnostic visualization)",fontsize=12)
    plt.tight_layout(rect=[0,0,1,.92]); _savefig(path)


def save_figure8_crop(results_by_name: Dict[str, Dict], path: Path):
    if "Baboon" not in results_by_name:return
    r=results_by_name["Baboon"]; system=r["system"]; package=r["package"]
    levels=[.10,.25,.50]
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,lv in zip(axes,levels):
        tam=_tamper_cipher(package["cipher"],"crop",lv)
        rec=_unauthenticated_recovery(system,package,tam)
        ax.imshow(cv2.cvtColor(rec,cv2.COLOR_BGR2RGB) if rec.ndim==3 else rec,cmap="gray")
        ax.set_title(f"Ciphertext crop {lv*100:.0f}%")
        ax.axis("off")
    fig.suptitle("Figure 8. Recovery after ciphertext cropping (diagnostic; authenticated protocol rejects tampering)",fontsize=12)
    plt.tight_layout(rect=[0,0,1,.92]); _savefig(path)


def save_figure9_bifurcation(params: np.ndarray, initials: np.ndarray, path: Path):
    bs=np.linspace(max(2.5,params[1]-.5),min(3.0,params[1]+.5),180)
    xs=[]
    for b in bs:
        p=params.copy(); p[1]=b; x=initials.copy()
        for _ in range(1800): x=HyperchaoticLorenz4D.rk4_state(x,p,.002)
        maxima=[]
        prev2=prev=x[0]
        for _ in range(5000):
            x=HyperchaoticLorenz4D.rk4_state(x,p,.002)
            cur=x[0]
            if prev>prev2 and prev>cur: maxima.append(prev)
            prev2,prev=prev,cur
        xs.extend([(b,v) for v in maxima[-80:]])
    if not xs:return
    arr=np.asarray(xs)
    plt.figure(figsize=(10,5))
    plt.plot(arr[:,0],arr[:,1],",",alpha=.55)
    plt.axvline(params[1],linestyle="--",linewidth=1.2,label=f"optimized b={params[1]:.6f}")
    plt.xlabel("b"); plt.ylabel("Local maxima of x"); plt.title("Figure 9. Bifurcation diagram: parameter b vs. x"); plt.legend()
    plt.tight_layout(); _savefig(path)


def save_figure10_11_attractors(params: np.ndarray, initials: np.ndarray, path10: Path, path11: Path):
    def draw(p,path,title):
        x=initials.copy(); traj=np.empty((22000,4))
        for i in range(len(traj)):
            x=HyperchaoticLorenz4D.rk4_state(x,p,.001); traj[i]=x
        fig=plt.figure(figsize=(12,8))
        pairs=[(0,2,"x-z"),(1,3,"y-w")]
        for k,(u,v,t) in enumerate(pairs):
            ax=fig.add_subplot(2,2,k+1); ax.plot(traj[:,u],traj[:,v],linewidth=.35); ax.set_xlabel("xyzw"[u]); ax.set_ylabel("xyzw"[v]); ax.set_title(t)
        ax=fig.add_subplot(2,2,3,projection="3d"); ax.plot(traj[:,0],traj[:,1],traj[:,2],linewidth=.25); ax.set_xlabel("x");ax.set_ylabel("y");ax.set_zlabel("z")
        ax=fig.add_subplot(2,2,4,projection="3d"); ax.plot(traj[:,1],traj[:,2],traj[:,3],linewidth=.25); ax.set_xlabel("y");ax.set_ylabel("z");ax.set_zlabel("w")
        fig.suptitle(title,fontsize=14); plt.tight_layout(rect=[0,0,1,.95]); _savefig(path)
    draw(REFERENCE_PARAMS, path10, "Figure 10. Pre-optimization attractor (reference parameters)")
    draw(params, path11, "Figure 11. Post-optimization attractor (content/key-bound parameters)")


# ============================================================
# One-image experiment
# ============================================================

def analyze_image(
    image_path: Path,
    root_key: bytes,
    save_figures: bool = True,
) -> Dict:
    print(f"\n=== {image_path.name} ===")

    system = ResearchEncryptionSystem(image_path, root_key)

    # Normal protocol encryption with fresh nonce.
    start = time.perf_counter()
    package = system.encrypt(system.image, nonce=nonce_for_protocol(), optimize=True)
    enc_time = time.perf_counter() - start

    try:
        start = time.perf_counter()
        recovered = system.decrypt(package)
        dec_time = time.perf_counter() - start
        decrypt_status = np.array_equal(system.image, recovered)
    except Exception as exc:
        recovered = None
        dec_time = float("nan")
        decrypt_status = False
        print(f"    Decryption failure: {exc}")

    # Recompute final Lyapunov spectrum independently using the final
    # optimized parameters and the content/key-bound initial state.
    le_result = HyperchaoticLorenz4D.lyapunov(
        np.asarray(package["params"], dtype=float),
        np.asarray(package["initials"], dtype=float),
        steps=FINAL_LE_CONFIG["steps"],
        transient=FINAL_LE_CONFIG["transient"],
        dt=FINAL_LE_CONFIG["dt"],
        qr_interval=FINAL_LE_CONFIG["qr_interval"],
        return_convergence=True,
    )

    if le_result is None:
        raise RuntimeError(f"Lyapunov calculation failed for {image_path.name}")

    le, history = le_result
    divergence = HyperchaoticLorenz4D.divergence(np.asarray(package["params"]))
    le_sum = float(np.sum(le))
    div_error = float(abs(le_sum - divergence))
    ky = HyperchaoticLorenz4D.kaplan_yorke(le)
    hyper_status = HyperchaoticLorenz4D.hyperchaos_status(le)

    # Controlled differential experiment:
    # same key and same nonce, only one plaintext pixel changed.
    control_nonce = nonce_for_controlled_experiment(root_key, system.name + "-NPCR")
    package2 = system.encrypt(
        system.image.copy(),
        nonce=control_nonce,
        optimize=True,
    )
    modified = system.image.copy()
    if modified.ndim == 2:
        modified[0, 0] = (int(modified[0, 0]) + 1) % 256
    else:
        modified[0, 0, 0] = (int(modified[0, 0, 0]) + 1) % 256

    package2_mod = system.encrypt(
        modified,
        nonce=control_nonce,
        optimize=True,
    )
    npcr, uaci = npcr_uaci(package2["cipher"], package2_mod["cipher"])

    entropy = entropy_channels(package["cipher"])
    local_mean, local_sd = local_entropy(package["cipher"])
    corr = correlation_values(package["cipher"])

    # One-bit root-key authentication test.
    wrong_key = bytearray(root_key)
    wrong_key[0] ^= 0x01
    wrong_key = bytes(wrong_key)

    wrong_key_status = "REJECTED"
    try:
        system.decrypt(package, key=wrong_key)
        wrong_key_status = "UNEXPECTED_SUCCESS"
    except Exception:
        pass

    # Chosen-plaintext nonce-separation check.
    if system.channels == 1:
        zero = np.zeros_like(system.image)
    else:
        zero = np.zeros_like(system.image)
    cp1 = system.encrypt(zero, nonce=nonce_for_controlled_experiment(root_key, system.name + "-CP1"))
    cp2 = system.encrypt(zero, nonce=nonce_for_controlled_experiment(root_key, system.name + "-CP2"))
    cp_same_plaintext_different_cipher = not np.array_equal(cp1["cipher"], cp2["cipher"])

    # Initial-state perturbation is reported separately, not as secret-key
    # sensitivity.
    init_sens = initial_state_perturbation(system, package)

    # Figures.
    if save_figures:
        save_histogram(
            system.image,
            FIG_DIR / f"{system.name}_original_hist.png",
            f"{system.name} - Original histogram",
        )
        save_histogram(
            package["cipher"],
            FIG_DIR / f"{system.name}_encrypted_hist.png",
            f"{system.name} - Encrypted histogram",
        )
        save_attractor(
            system,
            np.asarray(package["initials"]),
            np.asarray(package["params"]),
            FIG_DIR / f"{system.name}_attractor.png",
        )
        save_le_convergence(
            history,
            FIG_DIR / f"{system.name}_LE_convergence.png",
        )

    shape = list(system.image.shape)

    result = {
        "image": system.name,
        "filename": image_path.name,
        "shape": shape,
        "channels": int(system.channels),
        "pixels_or_samples": int(system.image.size),
        "sha256": package["digest"].hex(),
        "nonce": package["nonce"].hex(),
        "parameters": {
            "a": float(package["params"][0]),
            "b": float(package["params"][1]),
            "c": float(package["params"][2]),
            "r": float(package["params"][3]),
        },
        "initials": [float(x) for x in package["initials"]],
        "cs": {
            "population": CS_CONFIG["population"],
            "pa": CS_CONFIG["pa"],
            "generations": CS_CONFIG["generations"],
            "levy_beta": CS_CONFIG["levy_beta"],
            "seed": int(package["cs_seed"]),
            "fitness": float(package["fitness"]),
            "time_sec": float(package["cs_time"]),
        },
        "timing": {
            "encryption_sec": float(enc_time),
            "decryption_sec": float(dec_time),
        },
        "correct_decryption": bool(decrypt_status),
        "entropy": entropy,
        "local_entropy_mean": local_mean,
        "local_entropy_sd": local_sd,
        "correlation": corr,
        "npcr_percent": npcr,
        "uaci_percent": uaci,
        "lyapunov": {
            "lambda1": float(le[0]),
            "lambda2": float(le[1]),
            "lambda3": float(le[2]),
            "lambda4": float(le[3]),
            "sum": le_sum,
            "divergence": divergence,
            "absolute_error": div_error,
            "kaplan_yorke": ky,
            "positive_count": int(np.sum(le > 0)),
            "hyperchaos_status": hyper_status,
        },
        "key_authentication": {
            "root_key_bits": 256,
            "one_bit_modified_key": wrong_key_status,
        },
        "initial_state_perturbation": init_sens,
        "chosen_plaintext": {
            "same_plaintext_fresh_nonce_produces_different_ciphertext":
                bool(cp_same_plaintext_different_cipher),
        },
        "nist": {
            "status": "NOT_RUN",
            "note": "Use a validated SP 800-22 implementation before reporting NIST pass counts."
        },
    }

    # Keep non-serializable image/package objects only in memory for figure generation.
    FIGURE_DATA[system.name] = {
        "original": system.image.copy(),
        "cipher": package["cipher"].copy(),
        "recovered": None if recovered is None else recovered.copy(),
        "system": system,
        "package": package,
    }

    # Save raw result.
    (RAW_DIR / f"{system.name}.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(f"    Params: {result['parameters']}")
    print(f"    LE: {[round(x, 6) for x in le]}")
    print(f"    Sum LE: {le_sum:.9f}")
    print(f"    Divergence: {divergence:.9f}")
    print(f"    |Sum-Div|: {div_error:.3e}")
    print(f"    KY dimension: {ky:.6f}")
    print(f"    NPCR: {npcr:.6f}%")
    print(f"    UACI: {uaci:.6f}%")
    print(f"    Mean entropy: {entropy['mean']:.6f}")
    print(f"    Max |correlation|: {corr['max_abs']:.8f}")
    print(f"    Decryption: {'PASS' if decrypt_status else 'FAIL'}")
    print(f"    Wrong-key authentication: {wrong_key_status}")

    return result


# ============================================================
# Tables and report
# ============================================================

def flatten_results(results: List[Dict]):
    rows = []

    for r in results:
        e = r["entropy"]
        p = r["parameters"]
        le = r["lyapunov"]
        c = r["correlation"]

        row = {
            "Image": r["image"],
            "Filename": r["filename"],
            "Shape": "x".join(map(str, r["shape"])),
            "Channels": r["channels"],
            "a": p["a"],
            "b": p["b"],
            "c": p["c"],
            "r": p["r"],
            "Lambda1": le["lambda1"],
            "Lambda2": le["lambda2"],
            "Lambda3": le["lambda3"],
            "Lambda4": le["lambda4"],
            "SumLE": le["sum"],
            "Divergence": le["divergence"],
            "AbsSumDivError": le["absolute_error"],
            "KaplanYorke": le["kaplan_yorke"],
            "PositiveLEs": le["positive_count"],
            "Hyperchaos": le["hyperchaos_status"],
            "EntropyMean": e["mean"],
            "LocalEntropyMean": r["local_entropy_mean"],
            "LocalEntropySD": r["local_entropy_sd"],
            "MaxAbsCorrelation": c["max_abs"],
            "MeanAbsCorrelation": c["mean_abs"],
            "NPCR_percent": r["npcr_percent"],
            "UACI_percent": r["uaci_percent"],
            "Encryption_sec": r["timing"]["encryption_sec"],
            "Decryption_sec": r["timing"]["decryption_sec"],
            "DecryptionCorrect": r["correct_decryption"],
            "WrongKeyAuth": r["key_authentication"]["one_bit_modified_key"],
            "CP_FreshNonce_Different": r["chosen_plaintext"]["same_plaintext_fresh_nonce_produces_different_ciphertext"],
            "NIST": r["nist"]["status"],
        }

        for k, v in e.items():
            if k != "mean":
                row[f"Entropy_{k}"] = v

        rows.append(row)

    return rows


def write_csv(path: Path, rows: List[Dict]):
    import csv
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_excel(results: List[Dict], rows: List[Dict]):
    if Workbook is None:
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"

    # Table 2
    ws2 = wb.create_sheet("T2_Entropy")
    headers = ["Image", "EntropyMean", "LocalEntropyMean", "LocalEntropySD"]
    ws2.append(headers)
    for r in rows:
        ws2.append([r["Image"], r["EntropyMean"], r["LocalEntropyMean"], r["LocalEntropySD"]])

    # Table 4
    ws4 = wb.create_sheet("T4_Correlation")
    corr_keys = sorted([k for k in rows[0] if k.startswith("MaxAbsCorrelation") or k.startswith("MeanAbsCorrelation")])
    ws4.append(["Image"] + corr_keys)
    for r in rows:
        ws4.append([r["Image"]] + [r[k] for k in corr_keys])

    # Table 5
    ws5 = wb.create_sheet("T5_NPCR_UACI")
    ws5.append(["Image", "NPCR (%)", "UACI (%)"])
    for r in rows:
        ws5.append([r["Image"], r["NPCR_percent"], r["UACI_percent"]])
    npcrs = [r["NPCR_percent"] for r in rows]
    uacis = [r["UACI_percent"] for r in rows]
    ws5.append(["Mean", float(np.mean(npcrs)), float(np.mean(uacis))])
    ws5.append(["SD", float(np.std(npcrs)), float(np.std(uacis))])

    # Table 6
    ws6 = wb.create_sheet("T6_Key")
    ws6.append(["Image", "Wrong-key authentication", "Perturbation epsilon",
                "Perturbation PSNR", "Perturbation SSIM",
                "Perturbation NPCR", "Perturbation UACI"])
    for r in rows:
        s = r["initial_state_perturbation"]
        ws6.append([
            r["Image"],
            r["WrongKeyAuth"],
            s["epsilon"],
            s["psnr_db"],
            s["ssim"],
            s["npcr_percent"],
            s["uaci_percent"],
        ])

    # Table 8
    ws8 = wb.create_sheet("T8_Parameters")
    ws8.append(["Image", "a", "b", "c", "r"])
    for r in rows:
        ws8.append([r["Image"], r["a"], r["b"], r["c"], r["r"]])

    # Table 10
    ws10 = wb.create_sheet("T10_Lyapunov")
    ws10.append([
        "Image", "Lambda1", "Lambda2", "Lambda3", "Lambda4",
        "SumLE", "Divergence", "Absolute Error", "Kaplan-Yorke",
        "Positive LEs", "Hyperchaos"
    ])
    for r in rows:
        ws10.append([
            r["Image"], r["Lambda1"], r["Lambda2"], r["Lambda3"], r["Lambda4"],
            r["SumLE"], r["Divergence"], r["AbsSumDivError"],
            r["KaplanYorke"], r["PositiveLEs"], r["Hyperchaos"]
        ])

    # Table 11
    ws11 = wb.create_sheet("T11_Config")
    config_rows = [
        ("Secret key", "256-bit independent root key"),
        ("Nonce", "96-bit; fresh for protocol encryption"),
        ("Hash", "SHA-256"),
        ("KDF", "HKDF-SHA256"),
        ("Chaotic system", "4-D hyperchaotic Lorenz"),
        ("Solver", "RK4"),
        ("Lyapunov method", "Variational equations + QR/Benettin"),
        ("CS population", CS_CONFIG["population"]),
        ("CS generations", CS_CONFIG["generations"]),
        ("CS discovery probability Pa", CS_CONFIG["pa"]),
        ("Levy beta", CS_CONFIG["levy_beta"]),
        ("Parameter ranges", str({"a": LB[0:1].tolist()+UB[0:1].tolist(),
                                  "b": [LB[1], UB[1]],
                                  "c": [LB[2], UB[2]],
                                  "r": [LB[3], UB[3]]})),
        ("Secret key space", "2^256"),
    ]
    for item in config_rows:
        ws11.append(list(item))

    # Raw complete summary
    ws.append(["Number of images", len(rows)])
    ws.append(["Mean NPCR (%)", float(np.mean([r["NPCR_percent"] for r in rows]))])
    ws.append(["SD NPCR (%)", float(np.std([r["NPCR_percent"] for r in rows]))])
    ws.append(["Mean UACI (%)", float(np.mean([r["UACI_percent"] for r in rows]))])
    ws.append(["SD UACI (%)", float(np.std([r["UACI_percent"] for r in rows]))])
    ws.append(["Mean entropy", float(np.mean([r["EntropyMean"] for r in rows]))])
    ws.append(["Max absolute correlation", float(max(r["MaxAbsCorrelation"] for r in rows))])
    ws.append(["Max LE/divergence absolute error", float(max(r["AbsSumDivError"] for r in rows))])

    wb.save(ROOT / "PLOS_Round2_Results.xlsx")


def make_final_report(results: List[Dict]):
    rows = flatten_results(results)

    report = []
    report.append("PLOS ONE ROUND-2 EXPERIMENTAL REPORT")
    report.append("=" * 70)
    report.append(f"Images analyzed: {len(results)}")
    report.append("")

    report.append("TABLE 2 — ENTROPY AND LOCAL ENTROPY")
    report.append("-" * 70)
    for r in rows:
        report.append(
            f"{r['Image']}: mean entropy={r['EntropyMean']:.6f}, "
            f"local={r['LocalEntropyMean']:.6f} ± {r['LocalEntropySD']:.6f}"
        )

    report.append("")
    report.append("TABLE 5 — NPCR AND UACI")
    report.append("-" * 70)
    for r in rows:
        report.append(
            f"{r['Image']}: NPCR={r['NPCR_percent']:.6f}%, "
            f"UACI={r['UACI_percent']:.6f}%"
        )

    report.append(
        f"Mean NPCR={np.mean([r['NPCR_percent'] for r in rows]):.6f}%"
    )
    report.append(
        f"SD NPCR={np.std([r['NPCR_percent'] for r in rows]):.6f}%"
    )
    report.append(
        f"Mean UACI={np.mean([r['UACI_percent'] for r in rows]):.6f}%"
    )
    report.append(
        f"SD UACI={np.std([r['UACI_percent'] for r in rows]):.6f}%"
    )

    report.append("")
    report.append("TABLE 10 — LYAPUNOV / DIVERGENCE / KAPLAN-YORKE")
    report.append("-" * 70)
    for r in rows:
        report.append(
            f"{r['Image']}: "
            f"LE=[{r['Lambda1']:.9f}, {r['Lambda2']:.9f}, "
            f"{r['Lambda3']:.9f}, {r['Lambda4']:.9f}], "
            f"sum={r['SumLE']:.9f}, div={r['Divergence']:.9f}, "
            f"error={r['AbsSumDivError']:.3e}, "
            f"D_KY={r['KaplanYorke']:.6f}, "
            f"hyperchaos={r['Hyperchaos']}"
        )

    report.append("")
    report.append("SECURITY VALIDATION")
    report.append("-" * 70)
    for r in rows:
        report.append(
            f"{r['Image']}: decryption={'PASS' if r['DecryptionCorrect'] else 'FAIL'}, "
            f"wrong-key authentication={r['WrongKeyAuth']}, "
            f"fresh-nonce CPA separation={r['CP_FreshNonce_Different']}"
        )

    report.append("")
    report.append("NIST")
    report.append("-" * 70)
    report.append(
        "NIST results are NOT RUN unless a validated SP 800-22 implementation "
        "is explicitly connected. No pass count is fabricated."
    )

    report.append("")
    report.append("IMPORTANT INTERPRETATION")
    report.append("-" * 70)
    report.append(
        "The SHA-256 digest is a non-secret content-binding value. "
        "Confidentiality relies on the independent 256-bit root key."
    )
    report.append(
        "Initial-state perturbation is reported separately from root-key "
        "authentication and is not presented as proof of secret-key sensitivity."
    )
    report.append(
        "Cuckoo Search parameters are optimized within a predefined feasible "
        "hyperchaotic region; global optimality is not claimed."
    )

    (ROOT / "FINAL_REPORT.txt").write_text("\n".join(report), encoding="utf-8")


def write_table_csvs(results: List[Dict]):
    rows = flatten_results(results)

    # T2
    write_csv(TABLE_DIR / "Table_02_Entropy.csv", [
        {
            "Image": r["Image"],
            "EntropyMean": r["EntropyMean"],
            "LocalEntropyMean": r["LocalEntropyMean"],
            "LocalEntropySD": r["LocalEntropySD"],
        } for r in rows
    ])

    # T3
    write_csv(TABLE_DIR / "Table_03_NIST.csv", [
        {
            "Image": r["Image"],
            "Status": r["NIST"],
            "Note": "Validated SP 800-22 implementation required before reporting pass counts."
        } for r in rows
    ])

    # T4
    corr_rows = []
    for r in results:
        row = {"Image": r["image"]}
        row.update(r["correlation"])
        corr_rows.append(row)
    write_csv(TABLE_DIR / "Table_04_Correlation.csv", corr_rows)

    # T5
    write_csv(TABLE_DIR / "Table_05_NPCR_UACI.csv", [
        {"Image": r["Image"], "NPCR_percent": r["NPCR_percent"], "UACI_percent": r["UACI_percent"]}
        for r in rows
    ])

    # T6
    write_csv(TABLE_DIR / "Table_06_Key_and_Initial_State.csv", [
        {
            "Image": r["Image"],
            "WrongKeyAuthentication": r["WrongKeyAuth"],
            "Epsilon": r["initial_state_perturbation"]["epsilon"],
            "PerturbationPSNR": r["initial_state_perturbation"]["psnr_db"],
            "PerturbationSSIM": r["initial_state_perturbation"]["ssim"],
            "PerturbationNPCR": r["initial_state_perturbation"]["npcr_percent"],
            "PerturbationUACI": r["initial_state_perturbation"]["uaci_percent"],
        } for r in results
    ])

    # T7
    write_csv(TABLE_DIR / "Table_07_Chosen_Plaintext.csv", [
        {
            "Image": r["image"],
            "SamePlaintextFreshNonceProducesDifferentCiphertext":
                r["chosen_plaintext"]["same_plaintext_fresh_nonce_produces_different_ciphertext"]
        } for r in results
    ])

    # T8
    write_csv(TABLE_DIR / "Table_08_Optimized_Parameters.csv", [
        {"Image": r["image"], **r["parameters"]} for r in results
    ])

    # T9
    write_csv(TABLE_DIR / "Table_09_CS_Configuration.csv", [
        {"Parameter": "Population", "Value": CS_CONFIG["population"]},
        {"Parameter": "Pa", "Value": CS_CONFIG["pa"]},
        {"Parameter": "Generations", "Value": CS_CONFIG["generations"]},
        {"Parameter": "Levy beta", "Value": CS_CONFIG["levy_beta"]},
        {"Parameter": "a range", "Value": f"[{LB[0]}, {UB[0]}]"},
        {"Parameter": "b range", "Value": f"[{LB[1]}, {UB[1]}]"},
        {"Parameter": "c range", "Value": f"[{LB[2]}, {UB[2]}]"},
        {"Parameter": "r range", "Value": f"[{LB[3]}, {UB[3]}]"},
        {"Parameter": "Fitness", "Value": "lambda1 + lambda2 + 0.05*entropy - 0.5*autocorrelation"},
    ])

    # T10
    write_csv(TABLE_DIR / "Table_10_Lyapunov_KY.csv", [
        {
            "Image": r["image"],
            "lambda1": r["lyapunov"]["lambda1"],
            "lambda2": r["lyapunov"]["lambda2"],
            "lambda3": r["lyapunov"]["lambda3"],
            "lambda4": r["lyapunov"]["lambda4"],
            "SumLE": r["lyapunov"]["sum"],
            "Divergence": r["lyapunov"]["divergence"],
            "AbsError": r["lyapunov"]["absolute_error"],
            "KaplanYorke": r["lyapunov"]["kaplan_yorke"],
            "PositiveLEs": r["lyapunov"]["positive_count"],
            "Hyperchaos": r["lyapunov"]["hyperchaos_status"],
        } for r in results
    ])

    # T11
    write_csv(TABLE_DIR / "Table_11_Implementation.csv", [
        {"Parameter": "Secret key", "Value": "256-bit independent root key"},
        {"Parameter": "Nonce", "Value": "96-bit fresh nonce"},
        {"Parameter": "Hash", "Value": "SHA-256"},
        {"Parameter": "KDF", "Value": "HKDF-SHA256"},
        {"Parameter": "Chaotic system", "Value": "4-D hyperchaotic Lorenz"},
        {"Parameter": "Numerical solver", "Value": "RK4"},
        {"Parameter": "Lyapunov method", "Value": "Variational equations + QR/Benettin"},
        {"Parameter": "Secret key space", "Value": "2^256"},
    ])


# ============================================================
# Main
# ============================================================

def discover_images(folder: Path) -> List[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions])


def main():
    ensure_dirs()

    # Put the seven experimental images in an "images" folder.
    image_dir = Path("images")
    image_dir.mkdir(exist_ok=True)

    images = discover_images(image_dir)

    if not images:
        # Backward-compatible single-image fallback.
        fallback = Path("11256.png")
        if fallback.exists():
            images = [fallback]
        else:
            raise FileNotFoundError(
                "No images found. Put the seven experimental images in ./images "
                "or place 11256.png beside this script."
            )

    root_key = load_root_key()
    print("=" * 70)
    print("ROUND-2 RESEARCH PIPELINE")
    print("=" * 70)
    print(f"Images found: {len(images)}")
    print("Root key: 256-bit independent secret")
    print("System: 4-D hyperchaotic Lorenz")
    print("CS: population=10, Pa=0.25, generations=5")
    print("NIST: NOT RUN unless validated implementation is supplied")
    print("=" * 70)

    results = []
    for image_path in images:
        try:
            results.append(analyze_image(image_path, root_key))
        except Exception as exc:
            print(f"[ERROR] {image_path.name}: {exc}")
            raise

    # Generate every figure corresponding to the current manuscript, using the
    # same experimental objects that produced the numerical tables.
    save_figure1_framework(FIG_DIR / "Fig01_Protocol_Framework.png")
    save_figure2_visual(FIGURE_DATA, FIG_DIR / "Fig02_Visual_Results_Seven_Images.png")
    save_figure3_histograms(FIGURE_DATA, FIG_DIR / "Fig03_Histograms_Baboon_Peppers.png")
    save_figure4_polar(FIGURE_DATA, FIG_DIR / "Fig04_Polar_Histogram_Baboon.png")
    save_figure5_3d_corr(FIGURE_DATA, FIG_DIR / "Fig05_3D_RGB_Correlation_Baboon.png")
    save_figure6_key_sensitivity(FIGURE_DATA, FIG_DIR / "Fig06_Key_Authentication_Sensitivity.png")
    save_figure7_noise(FIGURE_DATA, FIG_DIR / "Fig07_Noise_Diagnostic_Baboon.png")
    save_figure8_crop(FIGURE_DATA, FIG_DIR / "Fig08_Cropping_Diagnostic_Baboon.png")
    if "Baboon" in FIGURE_DATA:
        baboon = FIGURE_DATA["Baboon"]
        bp = np.asarray(baboon["package"]["params"], dtype=float)
        bi = np.asarray(baboon["package"]["initials"], dtype=float)
        save_figure9_bifurcation(bp, bi, FIG_DIR / "Fig09_Bifurcation_b_vs_x.png")
        save_figure10_11_attractors(bp, bi,
                                     FIG_DIR / "Fig10_PreOptimization_Attractor.png",
                                     FIG_DIR / "Fig11_PostOptimization_Attractor.png")

    write_table_csvs(results)
    write_excel(results, flatten_results(results))
    make_final_report(results)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    rows = flatten_results(results)
    print(f"Images analyzed: {len(rows)}")
    print(f"Mean NPCR: {np.mean([r['NPCR_percent'] for r in rows]):.6f}%")
    print(f"SD NPCR:   {np.std([r['NPCR_percent'] for r in rows]):.6f}%")
    print(f"Mean UACI: {np.mean([r['UACI_percent'] for r in rows]):.6f}%")
    print(f"SD UACI:   {np.std([r['UACI_percent'] for r in rows]):.6f}%")
    print(f"Mean entropy: {np.mean([r['EntropyMean'] for r in rows]):.6f}")
    print(f"Max |correlation|: {max(r['MaxAbsCorrelation'] for r in rows):.8f}")
    print(f"Max |SumLE-Divergence|: {max(r['AbsSumDivError'] for r in rows):.3e}")
    print(f"Output: {ROOT.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
