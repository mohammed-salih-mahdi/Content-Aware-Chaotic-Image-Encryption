

import hashlib
import hmac
import json
import os
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.dirname(__file__))
from cuckoo_search import CuckooSearch  # noqa: E402  (v1's own cuckoo_search.py, unchanged)

FBCRYPT_DIR = "/home/claude/unify/final/Content-Aware-Chaotic-Image-Encryption"
sys.path.insert(0, FBCRYPT_DIR)
try:
    from fbcrypt import nist_sts as _official_sts  # official STS 2.1.2 wrapper (unmodified)
    HAVE_OFFICIAL_STS = True
except Exception as _e:  # pragma: no cover
    HAVE_OFFICIAL_STS = False
    _STS_IMPORT_ERROR = _e

NONCE_LEN = 12  # bytes (96-bit), standard for AES-GCM


class FullResearchEncryptionSystem:
    """v1 cipher core, verbatim, with a real AES-GCM/HMAC authentication
    layer wrapped around it (new), and official-STS NIST (new)."""

    def __init__(self, image_path, output_folder="Final_Research_Data", cs_seed=None,
                 secret_key=None):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        self.image_path = image_path
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        self.img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.img is None:
            raise ValueError(f"OpenCV could not read image: {image_path}")
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        self.rows, self.cols, self.channels = self.img.shape
        self.total_pixels = self.rows * self.cols * self.channels

        # NEW: independent 256-bit root secret key K (v1 had none).
        self.secret_key = bytes(secret_key) if secret_key is not None else os.urandom(32)
        if len(self.secret_key) != 32:
            raise ValueError("secret_key must be 32 bytes (256 bits)")

        print("[*] Optimizing control parameters (image-independent search)...")
        t0 = time.time()
        cs = CuckooSearch(pop_size=10, max_iter=5, seed=cs_seed)
        best_params, best_fit = cs.optimize()
        self.cs_time = time.time() - t0
        self.a, self.b, self.c, self.d, self.r = best_params
        print(f"[*] Optimized Parameters: a={self.a:.4f}, b={self.b:.4f}, "
              f"c={self.c:.4f}, d={self.d:.4f}, r={self.r:.4f} "
              f"(fitness={best_fit:.4f}, CS time={self.cs_time:.4f}s)")
        self.dt = 0.002



    def _derivatives(self, x, y, z, w):
        dx = self.a * (y - x) + w
        dy = self.c * x - y - x * z
        dz = x * y - self.b * z
        dw = -self.d * y + self.r * w
        return dx, dy, dz, dw

    def _jacobian(self, x, y, z, w):
        a, b, c, d, r = self.a, self.b, self.c, self.d, self.r
        return np.array([
            [-a,     a,  0.0, 1.0],
            [c - z, -1.0, -x, 0.0],
            [y,      x,  -b, 0.0],
            [0.0,   -d, 0.0,   r],
        ])

    def analyze_dynamics(self, le_list):
        a, b, c, d, r = self.a, self.b, self.c, self.d, self.r
        divergence = -a - 1 - b + r
        dissipative = divergence < 0
        sorted_les = np.sort(le_list)[::-1]
        cumsum = 0.0
        k = 0
        for i, le in enumerate(sorted_les):
            if cumsum + le < 0:
                k = i
                break
            cumsum += le
            k = i + 1
        if k < len(sorted_les) and k > 0:
            ky_dim = k + cumsum / abs(sorted_les[k]) if abs(sorted_les[k]) > 1e-12 else float(k)
        else:
            ky_dim = float(k)
        report = [
            f"Divergence (trace J) = -a - 1 - b + r = {divergence:.4f} "
            f"=> system is {'DISSIPATIVE' if dissipative else 'NOT DISSIPATIVE'}",
            f"Sorted Lyapunov spectrum: {np.round(sorted_les, 4).tolist()}",
            f"Kaplan-Yorke dimension (D_KY): {ky_dim:.4f}",
            f"Positive exponents: {int(np.sum(le_list > 0))} (hyperchaos requires >= 2)",
        ]
        for line in report:
            print(f"   {line}")
        return report

    def get_fingerprint(self, data):
        hash_val = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
        states = []
        for i in range(4):
            seg_a = int(hash_val[i * 8: i * 8 + 8], 16)
            seg_b = int(hash_val[(i + 4) * 8: (i + 4) * 8 + 8], 16)
            folded = seg_a ^ seg_b
            val = (folded % 1000) / 4000.0
            states.append(val + 0.1)
        return states

    def chaotic_engine(self, size, initials):
        x, y, z, w = initials
        ks_p = np.zeros(size)
        ks_d = np.zeros(size, dtype=np.uint8)
        traj_x, traj_y, traj_z, traj_w = [], [], [], []
        capture_limit = 3000
        for i in range(size):
            dx, dy, dz, dw = self._derivatives(x, y, z, w)
            x += dx * self.dt; y += dy * self.dt; z += dz * self.dt; w += dw * self.dt
            if abs(x) > 500: x %= 50
            if abs(y) > 500: y %= 50
            if abs(z) > 500: z %= 50
            if abs(w) > 500: w %= 50
            ks_p[i] = x
            ks_d[i] = int(abs(x) * 1e6 + abs(y) * 1e5 + abs(z) * 1e4 + abs(w) * 1e3) % 256
            if i < capture_limit:
                traj_x.append(x); traj_y.append(y); traj_z.append(z); traj_w.append(w)
        return ks_p, ks_d, (traj_x, traj_y, traj_z, traj_w)

    def calculate_lyapunov(self, transient_steps=5000, steps=50000, dt_lyap=0.0005):
        x, y, z, w = -0.1, 0.5, -0.6, 0.2
        for _ in range(transient_steps):
            dx, dy, dz, dw = self._derivatives(x, y, z, w)
            x += dx * dt_lyap; y += dy * dt_lyap; z += dz * dt_lyap; w += dw * dt_lyap
            if np.isnan(x) or np.abs(x) > 1e5:
                x, y, z, w = -0.1, 0.5, -0.6, 0.2
        W = np.eye(4, dtype=np.float64)
        LEs = np.zeros(4, dtype=np.float64)
        for _ in range(steps):
            dx, dy, dz, dw = self._derivatives(x, y, z, w)
            x += dx * dt_lyap; y += dy * dt_lyap; z += dz * dt_lyap; w += dw * dt_lyap
            if np.isnan(x) or np.abs(x) > 1e5:
                x, y, z, w = -0.1, 0.5, -0.6, 0.2
                W = np.eye(4)
                continue
            J = self._jacobian(x, y, z, w)
            try:
                W = W + np.dot(J, W) * dt_lyap
                Q, Rm = np.linalg.qr(W)
                LEs += np.log(np.abs(np.diag(Rm))) / dt_lyap
                W = Q
            except np.linalg.LinAlgError:
                W = np.eye(4)
        return LEs / steps

    def _cipher_core_encrypt(self, image, initials):
        """v1's encrypt_image body, unchanged math."""
        rows, cols, _ = image.shape
        total_pixels = rows * cols * 3
        ks_p, ks_d, _ = self.chaotic_engine(total_pixels, initials)
        flat_img = image.flatten()
        idx = np.argsort(ks_p)
        permuted = flat_img[idx]
        encrypted_flat = np.bitwise_xor(permuted, ks_d)
        return encrypted_flat.reshape((rows, cols, 3))

    def _cipher_core_decrypt(self, encrypted_image, initials):
        """v1's decrypt_image body, unchanged math."""
        rows, cols, _ = encrypted_image.shape
        total_pixels = rows * cols * 3
        ks_p, ks_d, _ = self.chaotic_engine(total_pixels, initials)
        flat_enc = encrypted_image.flatten()
        diffused_back = np.bitwise_xor(flat_enc, ks_d)
        idx = np.argsort(ks_p)
        original_flat = np.zeros(total_pixels, dtype=np.uint8)
        original_flat[idx] = diffused_back
        return original_flat.reshape((rows, cols, 3))

    def add_noise(self, image, percent):
        noisy = image.copy()
        num_salt = int(np.ceil(percent * image.size * 0.5))
        coords = tuple(np.random.randint(0, max(1, i - 1), num_salt) for i in image.shape[:2])
        noisy[coords[0], coords[1], :] = 255
        num_pepper = int(np.ceil(percent * image.size * 0.5))
        coords = tuple(np.random.randint(0, max(1, i - 1), num_pepper) for i in image.shape[:2])
        noisy[coords[0], coords[1], :] = 0
        return noisy

    def crop_image(self, image, percent):
        cropped = image.copy()
        h, w, _ = image.shape
        ratio = np.sqrt(percent / 100.0)
        crop_h = int(h * ratio); crop_w = int(w * ratio)
        y1 = (h - crop_h) // 2; y2 = y1 + crop_h
        x1 = (w - crop_w) // 2; x2 = x1 + crop_w
        cropped[y1:y2, x1:x2, :] = 0
        return cropped

    def calculate_metrics(self, img1, img2):
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        ssim_val = 0
        for i in range(3):
            ssim_val += ssim(img1[:, :, i], img2[:, :, i], data_range=255)
        return psnr, ssim_val / 3.0

    def calculate_npcr_uaci(self, img1, img2):
        img1 = img1.astype(np.int32); img2 = img2.astype(np.int32)
        d_matrix = np.not_equal(img1, img2).astype(np.int32)
        total_pixels = img1.size
        npc_r = (np.sum(d_matrix) / total_pixels) * 100
        uaci = (np.sum(np.abs(img1 - img2)) / (total_pixels * 255)) * 100
        return npc_r, uaci

    def full_correlation_table(self, image):
        channel_names = {0: "Blue", 1: "Green", 2: "Red"}
        offsets = {"Horizontal": (0, 1), "Vertical": (1, 0), "Diagonal": (1, 1)}
        table = {}
        for ch_idx, ch_name in channel_names.items():
            table[ch_name] = {}
            plane = image[:, :, ch_idx].astype(np.float64)
            h, w = plane.shape
            for dname, (dy, dx) in offsets.items():
                p1 = plane[: h - dy if dy else h, : w - dx if dx else w]
                p2 = plane[dy:, dx:]
                r = np.corrcoef(p1.flatten(), p2.flatten())[0, 1]
                table[ch_name][dname] = r
        return table

    def inter_channel_correlation(self, image):
        b = image[:, :, 0].flatten().astype(np.float64)
        g = image[:, :, 1].flatten().astype(np.float64)
        r = image[:, :, 2].flatten().astype(np.float64)
        return {"R-G": np.corrcoef(r, g)[0, 1], "R-B": np.corrcoef(r, b)[0, 1],
                "G-B": np.corrcoef(g, b)[0, 1]}

    def local_entropy(self, image, window=32, stride=32):
        """IDENTICAL to v1: non-overlapping 32x32 blocks, RGB channels
        pooled jointly (not separated), full non-overlapping sweep."""
        h, w, _ = image.shape
        entropies = []
        for y0 in range(0, h - window + 1, stride):
            for x0 in range(0, w - window + 1, stride):
                block = image[y0:y0 + window, x0:x0 + window, :]
                hist, _ = np.histogram(block, bins=256, range=(0, 256), density=True)
                prob = hist[hist > 0]
                if len(prob) == 0:
                    continue
                entropies.append(-np.sum(prob * np.log2(prob)))
        if not entropies:
            return 0.0, 0.0
        return float(np.mean(entropies)), float(np.std(entropies))

    def plot_bifurcation(self):
        """Sweeps parameter `b` and plots the resulting attractor
        maxima of x, marking the value of b actually used. [v1, verbatim]"""
        print("[*] Generating Bifurcation Diagram (b vs x)...")
        b_values = np.linspace(2.0, 5.0, 400)
        iterations = 1000
        last = 100
        x_init = -0.1
        plt.figure(figsize=(10, 6))
        for b_val in b_values:
            x, y, z, w = x_init, 0.5, -0.6, 0.2
            for i in range(iterations):
                dx = self.a * (y - x) + w
                dy = self.c * x - y - x * z
                dz = x * y - b_val * z
                dw = -self.d * y + self.r * w
                x += dx * self.dt; y += dy * self.dt; z += dz * self.dt; w += dw * self.dt
                if i >= (iterations - last):
                    plt.plot(b_val, x, marker=',', markersize=2, linestyle='None',
                              color='blue', alpha=0.6)
        plt.axvline(self.b, color='red', linestyle='--', linewidth=1,
                    label=f'Optimized b = {self.b:.2f}')
        plt.title(f'Bifurcation Diagram (Parameter b vs x)\n'
                  f'Optimized Params: a={self.a:.2f}, c={self.c:.2f}, '
                  f'd={self.d:.2f}, r={self.r:.2f}', fontsize=14)
        plt.xlabel('Parameter b', fontsize=12)
        plt.ylabel('Variable x (Maxima)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_folder, f'{self._tag}_Bifurcation_Diagram.png'), dpi=300)
        plt.close()
        print("[*] Saved Bifurcation Diagram")

    def plot_chaotic_attractors(self, trajectory, suffix=""):
        """[v1, verbatim]"""
        tx, ty, tz, tw = trajectory
        fig = plt.figure(figsize=(14, 10))
        plt.suptitle(f'Chaotic Attractor {suffix}', fontsize=16)
        ax1 = fig.add_subplot(2, 2, 1); ax1.plot(tx, tz, 'b-'); ax1.set_title('(a) x - z')
        ax2 = fig.add_subplot(2, 2, 2); ax2.plot(ty, tw, 'r-'); ax2.set_title('(b) y - w')
        ax3 = fig.add_subplot(2, 2, 3, projection='3d'); ax3.plot(tx, ty, tz, 'g-'); ax3.set_title('(c) x - y - z')
        ax4 = fig.add_subplot(2, 2, 4, projection='3d'); ax4.plot(ty, tz, tw, 'm-'); ax4.set_title('(d) y - z - w')
        plt.tight_layout()
        fname = f'Chaotic_Attractors{suffix}.png'
        plt.savefig(os.path.join(self.output_folder, fname), dpi=300)
        plt.close()
        print(f"[*] Saved Attractors ({fname})")

    def save_histogram(self, image, filename, title):
        """[v1, verbatim]"""
        plt.figure(figsize=(10, 6))
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col, label=f'Channel {col.upper()}')
        plt.title(title); plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_folder, filename))
        plt.close()

    def polar_histogram(self, image, filename, title, bins=64):
        """Table 1 source. [v1, verbatim]"""
        colors = {'Blue': (image[:, :, 0], 'b'), 'Green': (image[:, :, 1], 'g'), 'Red': (image[:, :, 2], 'r')}
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': 'polar'})
        fig.suptitle(title, fontsize=14)
        edges = np.linspace(0, 256, bins + 1)
        theta = np.linspace(0, 2 * np.pi, bins, endpoint=False)
        width = (2 * np.pi) / bins
        for ax, (name, (channel, col)) in zip(axes, colors.items()):
            counts, _ = np.histogram(channel.flatten(), bins=edges)
            ax.bar(theta, counts, width=width, color=col, alpha=0.75, edgecolor='none')
            ax.set_title(name, pad=15)
            ax.set_yticklabels([])
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, filename), dpi=200)
        plt.close()
        results = {}
        for name, (channel, _) in colors.items():
            counts, _ = np.histogram(channel.flatten(), bins=edges)
            n = counts.sum()
            if n == 0:
                results[name] = 0.0
                continue
            bin_centers = (edges[:-1] + edges[1:]) / 2 * (2 * np.pi / 256)
            C = np.sum(counts * np.cos(bin_centers)) / n
            S = np.sum(counts * np.sin(bin_centers)) / n
            results[name] = float(np.sqrt(C ** 2 + S ** 2))
        return results

    def plot_3d_rgb_correlation(self, image_rgb, prefix_name):
        """[v1, verbatim]"""
        total_pixels = self.rows * self.cols
        stride = max(1, total_pixels // 5000)
        offsets = [(0, 1, "Vertical"), (1, 0, "Horizontal"), (1, 1, "Diagonal")]
        for dx, dy, label in offsets:
            h, w, c = image_rgb.shape
            if h <= dy or w <= dx:
                continue
            p1 = image_rgb[dy:, dx:, :].reshape(-1, 3)
            p2 = image_rgb[:h - dy, :w - dx, :].reshape(-1, 3)
            if len(p1) == 0 or len(p2) == 0:
                continue
            p1 = p1[::stride]; p2 = p2[::stride]
            if len(p1) == 0:
                continue
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(p1[:, 0], p2[:, 0], np.zeros_like(p1[:, 0]), c='r', marker='.', s=1, alpha=0.6)
            ax.scatter(p1[:, 1], p2[:, 1], np.ones_like(p1[:, 1]), c='g', marker='.', s=1, alpha=0.6)
            ax.scatter(p1[:, 2], p2[:, 2], np.full_like(p1[:, 2], 2), c='b', marker='.', s=1, alpha=0.6)
            ax.set_xlabel('Pixel (x,y)'); ax.set_ylabel(f'Neighbor (x+{dx}, y+{dy})')
            ax.set_zlabel('Channel'); ax.set_zticks([0, 1, 2])
            ax.set_title(f'{label} - {prefix_name}')
            plt.savefig(os.path.join(self.output_folder, f'3D_{label}_{prefix_name}.png'), dpi=150)
            plt.close()

    def analyze_key_space(self):
        precision = 1e-15
        min_val = 0.1
        max_val = (999 / 4000.0) + 0.1
        span = max_val - min_val
        keys_per_state = span / precision
        total_keyspace = keys_per_state ** 4
        log2_keyspace = np.log2(total_keyspace)
        report = [
            f"Chaotic initial-state search space: 2^{log2_keyspace:.2f} "
            f"(image-content-derived, not secret on its own).",
            f"Independent root secret key K: 2^256 (new in this build; see "
            f"'AUTHENTICATION LAYER' section).",
        ]
        for line in report:
            print(f"   {line}")
        return report



    def _subkeys(self, nonce):
        """HKDF-SHA256(K, nonce) -> (aes_key, hmac_key), 32 bytes each."""
        okm = HKDF(algorithm=hashes.SHA256(), length=64, salt=nonce,
                   info=b"FBCRYPT-v4-subkeys").derive(self.secret_key)
        return okm[:32], okm[32:]

    def encrypt_image(self, image, nonce=None):
        """New authenticated encryption: cipher core (unchanged) + AES-GCM
        metadata protection + HMAC-SHA256 over (nonce||enc_meta||cipher).
        Returns (cipher_image, package_dict). The caller must keep
        `package_dict` (analogous to v1's `initials`) to decrypt later."""
        initials = self.get_fingerprint(image)
        cipher_img = self._cipher_core_encrypt(image, initials)

        nonce = os.urandom(NONCE_LEN) if nonce is None else bytes(nonce)
        aes_key, hmac_key = self._subkeys(nonce)
        meta = {
            "a": self.a, "b": self.b, "c": self.c, "d": self.d, "r": self.r,
            "initials": list(initials), "shape": list(image.shape),
        }
        raw_meta = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
        enc_meta = AESGCM(aes_key).encrypt(nonce, raw_meta, nonce)
        tag = hmac.new(hmac_key, nonce + enc_meta + cipher_img.tobytes(),
                        hashlib.sha256).digest()

        package = {"nonce": nonce.hex(), "encrypted_metadata": enc_meta.hex(), "hmac": tag.hex()}
        return cipher_img, package

    def decrypt_image(self, cipher_img, package):
        """Authenticated decryption: HMAC verified BEFORE decryption
        (fail-closed). Raises on any tampering. Then AES-GCM-decrypts
        metadata, extracts the initial states, and runs the unchanged
        cipher core to recover the plaintext."""
        nonce = bytes.fromhex(package["nonce"])
        enc_meta = bytes.fromhex(package["encrypted_metadata"])
        given_tag = bytes.fromhex(package["hmac"])

        _, hmac_key = self._subkeys(nonce)
        expected_tag = hmac.new(hmac_key, nonce + enc_meta + cipher_img.tobytes(),
                                 hashlib.sha256).digest()
        if not hmac.compare_digest(given_tag, expected_tag):
            raise ValueError("AUTHENTICATION_FAILED: HMAC verification failed; "
                              "package rejected before decryption.")

        aes_key, _ = self._subkeys(nonce)
        try:
            raw_meta = AESGCM(aes_key).decrypt(nonce, enc_meta, nonce)
        except Exception as e:
            raise ValueError(f"METADATA_AUTHENTICATION_FAILED: {e}")
        meta = json.loads(raw_meta)
        initials = meta["initials"]
        return self._cipher_core_decrypt(cipher_img, initials)

    def decrypt_core_unauthenticated(self, cipher_img, package):
        """DIAGNOSTIC ONLY (noise/cropping robustness figures). Bypasses
        the HMAC check on purpose to show how the cipher core alone
        degrades under corruption; the authenticated protocol
        (`decrypt_image`) rejects every one of these packages. Matches
        the manuscript's disclosed robustness-testing methodology."""
        nonce = bytes.fromhex(package["nonce"])
        enc_meta = bytes.fromhex(package["encrypted_metadata"])
        aes_key, _ = self._subkeys(nonce)
        raw_meta = AESGCM(aes_key).decrypt(nonce, enc_meta, nonce)
        meta = json.loads(raw_meta)
        return self._cipher_core_decrypt(cipher_img, meta["initials"])

    def export_key_material(self, package, path):
        with open(path, "w") as f:
            json.dump(package, f, indent=2)
        return path

    @staticmethod
    def import_key_material(path):
        with open(path) as f:
            return json.load(f)

    def key_sensitivity_test(self, cipher_img, correct_package):
        """1-ULP permutation-key shift + fully randomized diffusion key,
        compared against the correct authenticated decryption. Also
        confirms a single bit flip in K is rejected by HMAC (new)."""
        print("[*] Key-Sensitivity Test...")
        correct_decrypted = self.decrypt_image(cipher_img, correct_package)

        nonce = bytes.fromhex(correct_package["nonce"])
        enc_meta = bytes.fromhex(correct_package["encrypted_metadata"])
        aes_key, _ = self._subkeys(nonce)
        meta = json.loads(AESGCM(aes_key).decrypt(nonce, enc_meta, nonce))
        initials = meta["initials"]

        total_pixels = self.rows * self.cols * 3
        ks_p, ks_d, _ = self.chaotic_engine(total_pixels, initials)
        ks_p_wrong = ks_p.copy()
        for i in range(total_pixels):
            ks_p_wrong[i] = np.nextafter(ks_p_wrong[i], np.inf)
        np.random.seed(42)
        ks_d_wrong = np.random.randint(0, 256, total_pixels, dtype=np.uint8)
        flat_enc = cipher_img.flatten()
        diffused_back = np.bitwise_xor(flat_enc, ks_d_wrong)
        idx_wrong = np.argsort(ks_p_wrong)
        wrong_flat = np.zeros(total_pixels, dtype=np.uint8)
        wrong_flat[idx_wrong] = diffused_back
        wrong_decrypted = wrong_flat.reshape((self.rows, self.cols, 3))
        cv2.imwrite(os.path.join(self.output_folder, 'Sensitivity_Wrong_Key_Decryption.png'), wrong_decrypted)

        psnr_sens, ssim_sens = self.calculate_metrics(correct_decrypted, wrong_decrypted)
        npcr_sens, uaci_sens = self.calculate_npcr_uaci(correct_decrypted, wrong_decrypted)
        is_sensitive = "PASS" if npcr_sens > 99.6 else "FAIL"

        # NEW: one-bit change of K must be rejected by HMAC.
        bad_key_system = FullResearchEncryptionSystem.__new__(FullResearchEncryptionSystem)
        bad_key = bytearray(self.secret_key); bad_key[0] ^= 0x01
        bad_key_system.secret_key = bytes(bad_key)
        try:
            bad_key_system.decrypt_image(cipher_img, correct_package)
            one_bit_k_result = "ACCEPTED (BUG)"
        except ValueError as e:
            one_bit_k_result = f"REJECTED ({str(e).split(':')[0]})"

        report = [
            "Deviation Applied: np.nextafter (+1 ULP) on 100% of permutation key stream",
            "Deviation Applied: randomized diffusion key on 100% of diffusion key stream",
            f"PSNR (Correct vs Wrong Decrypt): {psnr_sens:.2f} dB (Ideal: < 10 dB)",
            f"SSIM (Correct vs Wrong Decrypt): {ssim_sens:.4f} (Ideal: ~ 0.0)",
            f"NPCR: {npcr_sens:.4f}% (Ideal > 99.6%) | Status: {is_sensitive}",
            f"UACI: {uaci_sens:.4f}% (Ideal ~ 33.46%)",
            f"One-bit change of K: {one_bit_k_result}",
        ]
        for line in report:
            print(f"   {line}")
        return report

    def chosen_plaintext_diffusion_probe(self):
        print("[*] Chosen-Plaintext Diffusion-Leakage Probe...")
        zero_img = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        impulse_img = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        impulse_img[self.rows // 2, self.cols // 2, :] = 255
        enc_zero, _ = self.encrypt_image(zero_img)
        enc_impulse, _ = self.encrypt_image(impulse_img)
        flat_cipher = enc_zero.flatten()
        hist, _ = np.histogram(flat_cipher, bins=256, range=(0, 256), density=True)
        prob = hist[hist > 0]
        leaked_entropy = -np.sum(prob * np.log2(prob))
        report = [
            "NOTE: ks_p is unknown to a chosen-plaintext attacker; the all-zero "
            "ciphertext cannot be trivially un-permuted.",
            f"All-zero ciphertext entropy: {leaked_entropy:.6f} (ideal 8.0)",
        ]
        cv2.imwrite(os.path.join(self.output_folder, 'CPA_Probe_AllZero.png'), enc_zero)
        cv2.imwrite(os.path.join(self.output_folder, 'CPA_Probe_Impulse.png'), enc_impulse)
        for line in report:
            print(f"   {line}")
        return report


    def nist_sp800_22_official(self, byte_stream, alpha=0.01):
        if not HAVE_OFFICIAL_STS:
            raise RuntimeError(f"Official STS wrapper unavailable: {_STS_IMPORT_ERROR}")
        bits = np.unpackbits(np.frombuffer(byte_stream, dtype=np.uint8))
        return _official_sts.run_sts(bits)


    def run_full_analysis(self):
        tag = os.path.splitext(os.path.basename(self.image_path))[0]
        self._tag = tag
        print(f"\n[*] Processing Image: {self.image_path} "
              f"({self.rows}x{self.cols}, {self.channels} channels)")

        self.plot_bifurcation()
        le_list = self.calculate_lyapunov()
        dynamics_report = self.analyze_dynamics(le_list)

        print("[*] Generating Chaotic Attractors...")
        dummy_initials_for_plot = self.get_fingerprint(self.img)
        _, _, trajectory = self.chaotic_engine(3000, dummy_initials_for_plot)
        self.plot_chaotic_attractors(trajectory, suffix=f"_{tag}")

        print("[*] Measuring Encryption Speed (10 runs)...")
        times_enc = []
        cipher_img = None
        package = None
        for _ in range(10):
            start = time.time()
            cipher_img, package = self.encrypt_image(self.img)
            times_enc.append(time.time() - start)
        avg_enc_time = float(np.mean(times_enc))

        key_path = os.path.join(self.output_folder, f'key_material_{tag}.json')
        self.export_key_material(package, key_path)

        print("[*] Measuring Decryption Speed (10 runs, authenticated)...")
        loaded_package = self.import_key_material(key_path)
        times_dec = []
        for _ in range(10):
            start = time.time()
            _ = self.decrypt_image(cipher_img, loaded_package)
            times_dec.append(time.time() - start)
        avg_dec_time = float(np.mean(times_dec))

        print("[*] Robustness Analysis (cipher core, HMAC verification bypassed)...")
        robustness_report = []
        robustness_rows = []
        for p in [0.01, 0.05, 0.10]:
            noisy_enc = self.add_noise(cipher_img, p)
            tampered_pkg = dict(loaded_package)  # metadata/HMAC now mismatch cipher
            recovered_noisy = self.decrypt_core_unauthenticated(noisy_enc, tampered_pkg)
            psnr, ssim_val = self.calculate_metrics(self.img, recovered_noisy)
            robustness_report.append(f"NOISE ({int(p*100)}%): PSNR={psnr:.2f}dB, SSIM={ssim_val:.4f}")
            robustness_rows.append(("Noise", f"{int(p*100)}%", psnr, ssim_val))
            cv2.imwrite(os.path.join(self.output_folder, f'{tag}_Recovered_Noise_{int(p*100)}pct.png'), recovered_noisy)
            try:
                self.decrypt_image(noisy_enc, tampered_pkg)
                auth_verdict = "ACCEPTED (BUG)"
            except ValueError as e:
                auth_verdict = f"REJECTED ({str(e).split(':')[0]})"
            robustness_rows[-1] = robustness_rows[-1] + (auth_verdict,)
        for p in [10, 25, 50]:
            cropped_enc = self.crop_image(cipher_img, p)
            tampered_pkg = dict(loaded_package)
            recovered_cropped = self.decrypt_core_unauthenticated(cropped_enc, tampered_pkg)
            psnr, ssim_val = self.calculate_metrics(self.img, recovered_cropped)
            robustness_report.append(f"CROP ({p}%): PSNR={psnr:.2f}dB, SSIM={ssim_val:.4f}")
            robustness_rows.append(("Cropping", f"{p}%", psnr, ssim_val))
            cv2.imwrite(os.path.join(self.output_folder, f'{tag}_Recovered_Crop_{p}pct.png'), recovered_cropped)
            try:
                self.decrypt_image(cropped_enc, tampered_pkg)
                auth_verdict = "ACCEPTED (BUG)"
            except ValueError as e:
                auth_verdict = f"REJECTED ({str(e).split(':')[0]})"
            robustness_rows[-1] = robustness_rows[-1] + (auth_verdict,)

        sensitivity_report = self.key_sensitivity_test(cipher_img, loaded_package)
        cpa_probe_report = self.chosen_plaintext_diffusion_probe()

        print("[*] Calculating NPCR and UACI (single-pixel differential test)...")
        modified_img = self.img.copy()
        modified_img[0, 0, 0] = np.uint8((int(modified_img[0, 0, 0]) + 1) % 256)
        encrypted_modified_img, _ = self.encrypt_image(modified_img, nonce=bytes.fromhex(package["nonce"]))
        npcr_val, uaci_val = self.calculate_npcr_uaci(cipher_img, encrypted_modified_img)

        entropy_vals = []
        for i in range(3):
            hist, _ = np.histogram(cipher_img[:, :, i], bins=256, range=(0, 256), density=True)
            prob = hist[hist > 0]
            entropy_vals.append(-np.sum(prob * np.log2(prob)))
        avg_entropy = float(np.mean(entropy_vals))

        corr_table = self.full_correlation_table(cipher_img)
        inter_ch_corr = self.inter_channel_correlation(cipher_img)

        print("[*] Generating Histograms, Polar Histograms (Table 1), and 3D Correlations...")
        self.save_histogram(self.img, f'{tag}_Hist_Original.png', 'Original Image Histogram')
        self.save_histogram(cipher_img, f'{tag}_Hist_Encrypted.png', 'Encrypted Image Histogram')
        polar_original = self.polar_histogram(self.img, f'{tag}_PolarHist_Original.png',
                                               'Polar Histogram - Original Image')
        polar_encrypted = self.polar_histogram(cipher_img, f'{tag}_PolarHist_Encrypted.png',
                                                'Polar Histogram - Encrypted Image')
        self.plot_3d_rgb_correlation(self.img_rgb, f"{tag}_Original")
        self.plot_3d_rgb_correlation(cv2.cvtColor(cipher_img, cv2.COLOR_BGR2RGB), f"{tag}_Encrypted")

        cv2.imwrite(os.path.join(self.output_folder, f'{tag}_Original.png'), self.img)
        cv2.imwrite(os.path.join(self.output_folder, f'{tag}_Encrypted.png'), cipher_img)
        decrypted_img = self.decrypt_image(cipher_img, loaded_package)
        cv2.imwrite(os.path.join(self.output_folder, f'{tag}_Decrypted.png'), decrypted_img)
        assert np.array_equal(self.img, decrypted_img), "Lossless round-trip check FAILED"
        print("[*] Lossless round-trip check (authenticated): PASSED")

        key_space_report = self.analyze_key_space()

        print("[*] Computing Local (sliding-window) Entropy [v1 method, unchanged]...")
        local_ent_mean, local_ent_std = self.local_entropy(cipher_img)
        print(f"   Local Entropy (32x32 blocks, pooled RGB): mean={local_ent_mean:.6f}, "
              f"std={local_ent_std:.6f}  (Global Entropy: {avg_entropy:.6f})")

        print("[*] Running the OFFICIAL NIST STS 2.1.2 on the diffusion key stream...")
        total_pixels = self.rows * self.cols * 3
        dummy_initials = self.get_fingerprint(self.img)
        ks_p, ks_d, _ = self.chaotic_engine(total_pixels, dummy_initials)
        nist_result = self.nist_sp800_22_official(ks_d.tobytes())

        out = {
            "image": tag, "shape": [self.rows, self.cols, self.channels],
            "secret_key_hex": self.secret_key.hex(),
            "parameters": {"a": self.a, "b": self.b, "c": self.c, "d": self.d, "r": self.r},
            "cs_time_sec": self.cs_time,
            "timing": {"encryption_sec": avg_enc_time, "decryption_sec": avg_dec_time},
            "lyapunov": le_list.tolist(), "dynamics": dynamics_report,
            "global_entropy": avg_entropy,
            "local_entropy_mean": local_ent_mean, "local_entropy_std": local_ent_std,
            "correlation": corr_table, "inter_channel_correlation": inter_ch_corr,
            "polar_R_original": polar_original, "polar_R_encrypted": polar_encrypted,
            "npcr_percent": npcr_val, "uaci_percent": uaci_val,
            "robustness": robustness_rows,
            "key_sensitivity": sensitivity_report,
            "chosen_plaintext_probe": cpa_probe_report,
            "key_space": key_space_report,
            "nist_official_sts": nist_result,
        }
        report_path = os.path.join(self.output_folder, f'Research_Report_{tag}.json')
        with open(report_path, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\n[OK] Success! '{tag}' saved in '{self.output_folder}'\n")
        return report_path
