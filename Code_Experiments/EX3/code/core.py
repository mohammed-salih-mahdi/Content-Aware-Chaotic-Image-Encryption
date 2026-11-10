

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import time
from typing import Optional, Tuple

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

PROTOCOL_VERSION = "Round2-4DHL-v1"

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
}

STREAM_DT = 0.001

LB = np.array([9.0, 2.5, 27.0, -1.52], dtype=float)
UB = np.array([11.0, 3.0, 29.0, -0.06], dtype=float)
REFERENCE_PARAMS = np.array([10.0, 8.0 / 3.0, 28.0, -1.0], dtype=float)


def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def image_bytes(image: np.ndarray) -> bytes:
    return np.ascontiguousarray(image).tobytes()


def image_digest(image: np.ndarray) -> bytes:
    return sha256_bytes(image_bytes(image))


def json_bytes(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
   
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    out, t, counter = b"", b"", 1
    while len(out) < length:
        t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha256).digest()
        out += t
        counter += 1
    return out[:length]


def bytes_to_initial_state(material: bytes) -> np.ndarray:
    
    vals = []
    for i in range(4):
        u = int.from_bytes(material[i * 8:(i + 1) * 8], "big")
        vals.append(0.1 + (u / 2 ** 64) * 0.8)
    return np.asarray(vals, dtype=float)


def random_nonce() -> bytes:
    return os.urandom(12)


def derived_nonce(root_key: bytes, label: str) -> bytes:experiments.
   
    return sha256_bytes(root_key + label.encode("utf-8"))[:12]



class HyperchaoticLorenz4D:
    @staticmethod
    def f(state: np.ndarray, p: np.ndarray) -> np.ndarray:
        x, y, z, w = state
        a, b, c, r = p
        return np.array([a * (y - x) + w,
                         c * x - y - x * z,
                         x * y - b * z,
                         -y * z + r * w], dtype=float)

    @staticmethod
    def jacobian(state: np.ndarray, p: np.ndarray) -> np.ndarray:
        x, y, z, w = state
        a, b, c, r = p
        return np.array([[-a, a, 0.0, 1.0],
                         [c - z, -1.0, -x, 0.0],
                         [y, x, -b, 0.0],
                         [0.0, -z, -y, r]], dtype=float)

    @staticmethod
    def divergence(p: np.ndarray) -> float:
        a, b, c, r = p
        return float(-a - 1.0 - b + r)

    @classmethod
    def rk4_state(cls, state, p, dt):
        k1 = cls.f(state, p)
        k2 = cls.f(state + 0.5 * dt * k1, p)
        k3 = cls.f(state + 0.5 * dt * k2, p)
        k4 = cls.f(state + dt * k3, p)
        return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    @classmethod
    def rk4_variational(cls, state, V, p, dt):
        def rhs(x, Y):
            return cls.f(x, p), cls.jacobian(x, p) @ Y
        k1x, k1v = rhs(state, V)
        k2x, k2v = rhs(state + 0.5 * dt * k1x, V + 0.5 * dt * k1v)
        k3x, k3v = rhs(state + 0.5 * dt * k2x, V + 0.5 * dt * k2v)
        k4x, k4v = rhs(state + dt * k3x, V + dt * k3v)
        return (state + dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0,
                V + dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0)

    @classmethod
    def lyapunov(cls, p, initials, steps, transient, dt, qr_interval=10,
                 return_convergence=False):
       
        x = np.asarray(initials, dtype=float).copy()
        p = np.asarray(p, dtype=float)
        for _ in range(transient):
            x = cls.rk4_state(x, p, dt)
            if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e6:
                return None
        V = np.eye(4)
        accum = np.zeros(4)
        total_time = 0.0
        history = []
        for step in range(1, steps + 1):
            x, V = cls.rk4_variational(x, V, p, dt)
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(V)) or np.linalg.norm(x) > 1e6:
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

    @staticmethod
    def kaplan_yorke(le) -> float:
        le = np.sort(np.asarray(le, dtype=float))[::-1]
        cumulative = np.cumsum(le)
        j = int(np.sum(cumulative >= 0))
        if j == 0:
            return 0.0
        if j >= len(le):
            return float(len(le))
        return float(j + cumulative[j - 1] / abs(le[j]))

    @staticmethod
    def hyperchaos_status(le) -> str:
        return "PASS" if int(np.sum(np.asarray(le) > 0.0)) >= 2 else "FAIL"


def trajectory(initials, params, n, dt=STREAM_DT) -> np.ndarray:
    
    x, y, z, w = (float(v) for v in initials)
    a, b, c, r = (float(v) for v in params)
    h2 = 0.5 * dt
    out = np.empty((n, 4))
    for i in range(n):
        x, y, z, w = _rk4_scalar(x, y, z, w, a, b, c, r, dt, h2)
        out[i] = (x, y, z, w)
    return out


def _rk4_scalar(x, y, z, w, a, b, c, r, dt, h2):
    k1x = a * (y - x) + w; k1y = c * x - y - x * z; k1z = x * y - b * z; k1w = -y * z + r * w
    xs = x + h2 * k1x; ys = y + h2 * k1y; zs = z + h2 * k1z; ws = w + h2 * k1w
    k2x = a * (ys - xs) + ws; k2y = c * xs - ys - xs * zs; k2z = xs * ys - b * zs; k2w = -ys * zs + r * ws
    xs = x + h2 * k2x; ys = y + h2 * k2y; zs = z + h2 * k2z; ws = w + h2 * k2w
    k3x = a * (ys - xs) + ws; k3y = c * xs - ys - xs * zs; k3z = xs * ys - b * zs; k3w = -ys * zs + r * ws
    xs = x + dt * k3x; ys = y + dt * k3y; zs = z + dt * k3z; ws = w + dt * k3w
    k4x = a * (ys - xs) + ws; k4y = c * xs - ys - xs * zs; k4z = xs * ys - b * zs; k4w = -ys * zs + r * ws
    return (x + dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0,
            y + dt * (k1y + 2 * k2y + 2 * k3y + k4y) / 6.0,
            z + dt * (k1z + 2 * k2z + 2 * k3z + k4z) / 6.0,
            w + dt * (k1w + 2 * k2w + 2 * k3w + k4w) / 6.0)


def generate_streams(size: int, initials, params, dt: float = STREAM_DT):
 
    x, y, z, w = (float(v) for v in initials)
    a, b, c, r = (float(v) for v in params)
    h2 = 0.5 * dt
    perm = np.empty(size, dtype=float)
    diff = np.empty(size, dtype=np.uint8)
    floor, isfinite = math.floor, math.isfinite
    for i in range(size):
        x, y, z, w = _rk4_scalar(x, y, z, w, a, b, c, r, dt, h2)
        if not (isfinite(x) and isfinite(y) and isfinite(z) and isfinite(w)):
            raise FloatingPointError("Chaotic trajectory became non-finite.")
        perm[i] = x
        diff[i] = (floor(abs(x) * 1e6) + floor(abs(z) * 1e4)) % 256
    return perm, diff


def generate_streams_reference(size, initials, params, dt=STREAM_DT):
     
    state = np.asarray(initials, dtype=float).copy()
    params = np.asarray(params, dtype=float)
    perm = np.empty(size, dtype=float)
    diff = np.empty(size, dtype=np.uint8)
    for i in range(size):
        state = HyperchaoticLorenz4D.rk4_state(state, params, dt)
        perm[i] = state[0]
        diff[i] = int(np.floor(abs(state[0]) * 1e6) + np.floor(abs(state[2]) * 1e4)) % 256
    return perm, diff



class CuckooSearch:
    def __init__(self, initials, seed, pop_size=10, pa=0.25, max_iter=5,
                 levy_beta=1.5, quick_steps=2500, quick_transient=1000, dt=0.001):
        self.initials = np.asarray(initials, dtype=float)
        self.rng = np.random.default_rng(int(seed))
        self.pop_size, self.pa, self.max_iter = int(pop_size), float(pa), int(max_iter)
        self.levy_beta = float(levy_beta)
        self.quick_steps, self.quick_transient, self.dt = int(quick_steps), int(quick_transient), float(dt)
        self.lb, self.ub = LB.copy(), UB.copy()
        self.history = []
        self.evaluations = 0

    @staticmethod
    def stream_stats(p, initials, n=1000):
        state = np.asarray(initials, dtype=float).copy()
        vals = np.empty(n)
        for i in range(n):
            state = HyperchaoticLorenz4D.rk4_state(state, p, 0.001)
            if not np.all(np.isfinite(state)):
                return 0.0, 1.0
            vals[i] = state[0]
        q = np.mod(np.floor(np.abs(vals) * 1e6).astype(np.int64), 256)
        counts = np.bincount(q, minlength=256).astype(np.float64)
        prob = counts[counts > 0] / float(n)
        entropy = -np.sum(prob * np.log2(prob))
        if np.std(q[:-1]) > 0 and np.std(q[1:]) > 0:
            ac = abs(float(np.corrcoef(q[:-1], q[1:])[0, 1]))
        else:
            ac = 1.0
        return float(entropy), float(ac)

    def levy(self, size):
        beta = self.levy_beta
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
                 / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = self.rng.normal(0, sigma, size)
        v = self.rng.normal(0, 1, size)
        return u / np.maximum(np.abs(v), 1e-12) ** (1 / beta)

    def fitness(self, p):

        self.evaluations += 1
        p = np.asarray(p, dtype=float)
        if HyperchaoticLorenz4D.divergence(p) >= 0:
            return -1e9
        le = HyperchaoticLorenz4D.lyapunov(p, self.initials, steps=self.quick_steps,
                                           transient=self.quick_transient, dt=self.dt, qr_interval=10)
        if le is None:
            return -1e9
        if int(np.sum(le > 0)) < 2:
            return -1e6 + float(np.sum(le))
        entropy, autocorr = self.stream_stats(p, self.initials)
        return float(le[0] + le[1] + 0.05 * entropy - 0.5 * autocorr)

    def optimize(self):
        nests, fitness = [], []
        attempts, max_attempts = 0, max(100, self.pop_size * 30)
        while len(nests) < self.pop_size and attempts < max_attempts:
            attempts += 1
            p = self.rng.uniform(self.lb, self.ub)
            f = self.fitness(p)
            if f > -1e6:
                nests.append(p)
                fitness.append(f)
        if len(nests) < self.pop_size:
            raise RuntimeError("Unable to obtain enough feasible hyperchaotic CS candidates.")
        nests, fitness = np.asarray(nests), np.asarray(fitness)
        best_idx = int(np.argmax(fitness))
        best, best_f = nests[best_idx].copy(), float(fitness[best_idx])
        span = self.ub - self.lb
        for gen in range(1, self.max_iter + 1):
            for i in range(self.pop_size):
                candidate = np.clip(nests[i] + 0.01 * self.levy(4) * span, self.lb, self.ub)
                f = self.fitness(candidate)
                if f > fitness[i]:
                    nests[i], fitness[i] = candidate, f
            for i in np.argsort(fitness)[:max(1, int(self.pa * self.pop_size))]:
                for _ in range(20):
                    candidate = self.rng.uniform(self.lb, self.ub)
                    f = self.fitness(candidate)
                    if f > -1e6:
                        nests[i], fitness[i] = candidate, f
                        break
            idx = int(np.argmax(fitness))
            if fitness[idx] > best_f:
                best, best_f = nests[idx].copy(), float(fitness[idx])
            self.history.append({"generation": gen, "best_fitness": best_f})
        return best, best_f


class FilterBubbleCipher:
 
    def __init__(self, root_key: bytes):
        if len(root_key) != 32:
            raise ValueError("root_key must be 32 bytes (256 bits).")
        self.root_key = bytes(root_key)

 
    def derive_context(self, image: np.ndarray, nonce: bytes):
        digest = image_digest(image)
        initials = bytes_to_initial_state(
            hkdf_sha256(self.root_key, salt=nonce, info=b"FilterBubble/initial-state/" + digest, length=32))
        cs_seed = int.from_bytes(
            hkdf_sha256(self.root_key, salt=nonce, info=b"FilterBubble/CS-seed/" + digest, length=8),
            "big") % (2 ** 63 - 1)
        return digest, initials, cs_seed

   
    def optimize_parameters(self, image: np.ndarray, nonce: bytes):
        digest, initials, cs_seed = self.derive_context(image, nonce)
        cs = CuckooSearch(initials, seed=cs_seed, pop_size=CS_CONFIG["population"], pa=CS_CONFIG["pa"],
                          max_iter=CS_CONFIG["generations"], levy_beta=CS_CONFIG["levy_beta"],
                          quick_steps=CS_CONFIG["quick_le_steps"], quick_transient=CS_CONFIG["quick_transient"],
                          dt=CS_CONFIG["dt"])
        t0 = time.perf_counter()
        params, fitness = cs.optimize()
        cs_time = time.perf_counter() - t0
        return {"digest": digest, "initials": initials, "cs_seed": cs_seed, "params": params,
                "fitness": fitness, "cs_time": cs_time, "cs_history": cs.history,
                "cs_evaluations": cs.evaluations}

    @staticmethod
    def _meta_nonce(nonce: bytes) -> bytes:
        return sha256_bytes(nonce + b"Round2-4DHL metadata v1")[:12]

  
    def encrypt_with_session(self, image: np.ndarray, nonce: bytes, session: dict) -> dict:
     
        shape = image.shape
        flat = np.ascontiguousarray(image).reshape(-1)
        perm, diff = generate_streams(flat.size, session["initials"], session["params"])
        order = np.argsort(perm, kind="mergesort")
        cipher = np.bitwise_xor(flat[order], diff).reshape(shape)
        metadata = {
            "version": PROTOCOL_VERSION,
            "shape": list(shape),
            "channels": 1 if image.ndim == 2 else int(shape[2]),
            "sha256": session["digest"].hex(),
            "params": [float(v) for v in session["params"]],
            "initials": [float(v) for v in session["initials"]],
            "cs_seed": int(session["cs_seed"]),
            "fitness": float(session["fitness"]),
        }
        protected = AESGCM(self.root_key).encrypt(self._meta_nonce(nonce), json_bytes(metadata), nonce)
        tag = hmac.new(self.root_key, nonce + protected + image_bytes(cipher), hashlib.sha256).hexdigest()
        return {"cipher": cipher, "nonce": nonce, "protected_metadata": protected, "tag": tag,
                "_session": session, "_metadata": metadata}

    def encrypt(self, image: np.ndarray, nonce: Optional[bytes] = None) -> dict:
        nonce = random_nonce() if nonce is None else nonce
        session = self.optimize_parameters(image, nonce)
        return self.encrypt_with_session(image, nonce, session)

    def open_metadata(self, package: dict, key: Optional[bytes] = None) -> dict:
        key = self.root_key if key is None else key
        expected = hmac.new(key, package["nonce"] + package["protected_metadata"] + image_bytes(package["cipher"]),
                            hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, package["tag"]):
            raise ValueError("AUTHENTICATION_FAILED")
        try:
            meta = json.loads(AESGCM(key).decrypt(self._meta_nonce(package["nonce"]),
                                                  package["protected_metadata"], package["nonce"]))
        except Exception as exc:
            raise ValueError("METADATA_AUTHENTICATION_FAILED") from exc
        if meta.get("version") != PROTOCOL_VERSION:
            raise ValueError("UNSUPPORTED_PROTOCOL_VERSION")
        return meta

    def decrypt(self, package: dict, key: Optional[bytes] = None) -> np.ndarray:
        meta = self.open_metadata(package, key)
        recovered = invert_cipher_core(package["cipher"], meta["initials"], meta["params"], meta["shape"])
        if image_digest(recovered).hex() != meta["sha256"]:
            raise ValueError("PLAINTEXT_DIGEST_MISMATCH")
        return recovered


def invert_cipher_core(cipher, initials, params, shape) -> np.ndarray:
   
    flat = np.ascontiguousarray(cipher).reshape(-1)
    perm, diff = generate_streams(flat.size, initials, params)
    order = np.argsort(perm, kind="mergesort")
    out = np.empty_like(flat)
    out[order] = np.bitwise_xor(flat, diff)
    return out.reshape(tuple(shape))
