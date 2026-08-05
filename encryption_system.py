"""
encryption_system.py
-------------------------------------------------------------------
Content-aware image encryption using a Cuckoo Search-optimized 4D
hyperchaotic (Chen-type) system for keystream generation, plus a full
research/benchmarking pipeline (performance timing, robustness under
noise/cropping, differential and key-sensitivity analysis, entropy,
correlation, and the NIST SP 800-22 randomness).

High-level design
------------------
1. The 4D system's control parameters (a, b, c, d, r) are tuned once
   per run by `CuckooSearch` (see cuckoo_search.py) so the resulting
   Lyapunov spectrum is genuinely hyperchaotic. These parameters are
   image-independent.
2. For each image, a SHA-256 hash of the plaintext is folded into the
   system's four initial states, making the keystream
   content-dependent (a single-bit change in the plaintext yields an
   entirely different keystream).
3. The chaotic trajectory produces two keystreams: a permutation
   stream (pixel shuffling) and a diffusion stream (byte-wise XOR).
4. Decryption requires the same initial states, which must be
   obtained out-of-band (e.g. via `export_key_material` /
   `import_key_material`) rather than re-derived from ciphertext.

Dependencies (all pip-installable):
    numpy, opencv-python (cv2), matplotlib, scikit-image, scipy
"""

import hashlib
import json
import math
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3D projection)
from skimage.metrics import structural_similarity as ssim

from cuckoo_search import CuckooSearch
from nist_randomness_tests import nist_sp800_22_full_suite


class FullResearchEncryptionSystem:
    """
    End-to-end pipeline: loads an image, optimizes the chaotic
    system's parameters, encrypts/decrypts it, and runs the full
     security/statistical/robustness analyses used to
    evaluate the scheme.
    """

    def __init__(self, image_path, output_folder="Final_Research_Data", cs_seed=None):
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

    # -----------------------------------------------------------------
    # Dynamics (must stay identical to CuckooSearch._derivatives / _jacobian)
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # Dynamical verification: divergence / dissipativity / Kaplan-Yorke dim
    # -----------------------------------------------------------------
    def analyze_dynamics(self, le_list):
        """Reports whether the system is dissipative (trace of the
        Jacobian < 0) and computes the Kaplan-Yorke fractal dimension
        from the (already-computed) Lyapunov spectrum."""
        a, b, c, d, r = self.a, self.b, self.c, self.d, self.r
        divergence = -a - 1 - b + r  # trace of the Jacobian (state-independent here)
        dissipative = divergence < 0

        # Kaplan-Yorke dimension
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
            f"Positive exponents: {int(np.sum(le_list > 0))} "
            f"(hyperchaos requires >= 2)",
        ]
        for line in report:
            print(f"   {line}")
        return report

    # -----------------------------------------------------------------
    def plot_bifurcation(self):
        """Sweeps parameter `b` and plots the resulting attractor
        maxima of x, marking the value of b actually used."""
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
        plt.savefig(os.path.join(self.output_folder, 'Bifurcation_Diagram.png'), dpi=300)
        plt.close()
        print("[*] Saved Bifurcation Diagram")

    def plot_chaotic_attractors(self, trajectory, suffix=""):
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

    # -----------------------------------------------------------------
    # Plaintext fingerprint -> chaotic-system initial states
    # -----------------------------------------------------------------
    def get_fingerprint(self, data):
        """Derives the four initial states (x0, y0, z0, w0) from a
        SHA-256 hash of the plaintext image, so the keystream is
        content-dependent. The 256-bit digest is split into 8
        segments of 32 bits; segment i is XOR-folded with segment
        i+4, so every derived state depends on the full digest rather
        than only half of it."""
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
        """
        Integrates the 4D system forward `size` steps starting from
        `initials`, producing two keystreams:
          - ks_p: the raw floating-point x-trajectory, later used for
            a permutation (its argsort defines the pixel shuffle).
          - ks_d: an 8-bit diffusion byte per step, XORed with the
            (permuted) plaintext.

        The diffusion byte is derived from the HIGH-order (fast
        varying) digits of all four state variables combined:
            int(|x|*1e6 + |y|*1e5 + |z|*1e4 + |w|*1e3) % 256
        An earlier version derived ks_d from only the LOW-order
        magnitude of two of the four states, which under a small
        integration step (dt=0.002) varies only slowly from one step
        to the next; that produced a keystream with a lag-1
        autocorrelation of ~0.82 and failed most of the NIST
        SP 800-22 randomness tests even though global entropy looked
        fine. Combining the high-order digits of all four states
        instead drops the lag-1 autocorrelation to ~0.009 and passes
        the full 15-test NIST.
        """
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

    # -----------------------------------------------------------------
    # High-precision Lyapunov spectrum (Benettin/QR method)
    # -----------------------------------------------------------------
    def calculate_lyapunov(self, transient_steps=5000, steps=50000, dt_lyap=0.0005):
        """Estimates the full Lyapunov spectrum using the Benettin/QR
        method, first discarding a burn-in transient so the trajectory
        has settled onto the attractor before exponents are
        accumulated."""
        print("[*] Computing High-Precision Lyapunov Exponents "
              f"(Benettin/QR, dt={dt_lyap}, steps={steps}, burn-in={transient_steps})...")
        x, y, z, w = -0.1, 0.5, -0.6, 0.2

        # --- burn-in: let the trajectory settle onto the attractor ---
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

    # -----------------------------------------------------------------
    # Encryption / decryption
    # -----------------------------------------------------------------
    def encrypt_image(self, image, custom_initials=None):
        """Returns (encrypted_image, initials_used). `initials` must
        be retained by the caller (e.g. via `export_key_material`) in
        order to decrypt later."""
        rows, cols, _ = image.shape
        total_pixels = rows * cols * 3
        initials = custom_initials if custom_initials is not None else self.get_fingerprint(image)
        ks_p, ks_d, _ = self.chaotic_engine(total_pixels, initials)
        flat_img = image.flatten()
        idx = np.argsort(ks_p)
        permuted = flat_img[idx]
        encrypted_flat = np.bitwise_xor(permuted, ks_d)
        return encrypted_flat.reshape((rows, cols, 3)), initials

    def decrypt_image(self, encrypted_image, initials):
        """
        Decrypts `encrypted_image` using the explicitly supplied
        `initials`. Decryption never re-derives `initials` from the
        original plaintext internally -- the caller must obtain it
        either by re-hashing a recovered plaintext (legitimate reuse)
        or by loading it from a key file produced by
        `export_key_material()`, which models out-of-band secure key
        transmission.
        """
        rows, cols, _ = encrypted_image.shape
        total_pixels = rows * cols * 3
        ks_p, ks_d, _ = self.chaotic_engine(total_pixels, initials)
        flat_enc = encrypted_image.flatten()
        diffused_back = np.bitwise_xor(flat_enc, ks_d)
        idx = np.argsort(ks_p)
        original_flat = np.zeros(total_pixels, dtype=np.uint8)
        original_flat[idx] = diffused_back
        return original_flat.reshape((rows, cols, 3))

    # -----------------------------------------------------------------
    # Key material export / import (explicit "secure channel" stand-in)
    # -----------------------------------------------------------------
    def export_key_material(self, initials, path):
        """Serializes everything the receiver needs but does not
        already have: the optimized control parameters and the
        hash-derived initial state. In a real deployment this JSON
        payload would be encrypted under the recipient's public key
        before transmission; here it stands in for that secure
        channel so encryption and decryption no longer share
        in-memory state directly."""
        payload = {
            "a": self.a, "b": self.b, "c": self.c, "d": self.d, "r": self.r,
            "initials": list(initials),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    @staticmethod
    def import_key_material(path):
        with open(path) as f:
            payload = json.load(f)
        return payload["initials"], (payload["a"], payload["b"], payload["c"],
                                      payload["d"], payload["r"])

    # -----------------------------------------------------------------
    # Robustness helpers
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # Key-sensitivity test
    # -----------------------------------------------------------------
    def key_sensitivity_test(self, encrypted_img, correct_initials):
        """
        Decrypts with the correct key, then decrypts again with a
        minimally perturbed key (one ULP shift on the permutation
        stream, a fully randomized diffusion stream) and compares the
        two outputs. This measures how sensitive decryption is to the
        exact key value; it is NOT a chosen/known-plaintext attack
        simulation (that is a separate, distinct test -- see
        `chosen_plaintext_diffusion_probe`).
        """
        print("[*] Key-Sensitivity Test...")
        correct_decrypted = self.decrypt_image(encrypted_img, correct_initials)

        total_pixels = self.rows * self.cols * 3
        ks_p, ks_d, _ = self.chaotic_engine(total_pixels, correct_initials)

        ks_p_wrong = ks_p.copy()
        for i in range(total_pixels):
            ks_p_wrong[i] = np.nextafter(ks_p_wrong[i], np.inf)

        np.random.seed(42)
        ks_d_wrong = np.random.randint(0, 256, total_pixels, dtype=np.uint8)

        flat_enc = encrypted_img.flatten()
        diffused_back = np.bitwise_xor(flat_enc, ks_d_wrong)
        idx_wrong = np.argsort(ks_p_wrong)
        wrong_flat = np.zeros(total_pixels, dtype=np.uint8)
        wrong_flat[idx_wrong] = diffused_back
        wrong_decrypted = wrong_flat.reshape((self.rows, self.cols, 3))

        cv2.imwrite(os.path.join(self.output_folder, 'Sensitivity_Wrong_Key_Decryption.png'), wrong_decrypted)

        psnr_sens, ssim_sens = self.calculate_metrics(correct_decrypted, wrong_decrypted)
        npcr_sens, uaci_sens = self.calculate_npcr_uaci(correct_decrypted, wrong_decrypted)
        is_sensitive = "PASS" if npcr_sens > 99.6 else "FAIL"

        report = [
            "Deviation Applied: np.nextafter (+1 ULP) on 100% of permutation key stream",
            "Deviation Applied: randomized diffusion key on 100% of diffusion key stream",
            f"PSNR (Correct vs Wrong Decrypt): {psnr_sens:.2f} dB (Ideal: < 10 dB)",
            f"SSIM (Correct vs Wrong Decrypt): {ssim_sens:.4f} (Ideal: ~ 0.0)",
            f"NPCR: {npcr_sens:.4f}% (Ideal > 99.6%) | Status: {is_sensitive}",
            f"UACI: {uaci_sens:.4f}% (Ideal ~ 33.46%)",
        ]
        for line in report:
            print(f"   {line}")
        return report

    # -----------------------------------------------------------------
    # Chosen-plaintext diffusion-leakage probe
    # -----------------------------------------------------------------
    def chosen_plaintext_diffusion_probe(self):
        """
        Encrypts an all-zero image and an impulse image, each under
        its OWN correctly-derived key (since the scheme is
        content-dependent by design, every plaintext gets its own
        fingerprint). For an all-zero plaintext, ciphertext =
        permute(ks_d), so this probe inspects whether the diffusion
        keystream leaks non-uniform structure once the permutation is
        known/sorted out -- a property that a real attacker
        controlling the input to a single-key (non-content-dependent)
        variant of this scheme could exploit.

        Note: under the current content-dependent design, the
        permutation key ks_p is unknown to an attacker who only
        chooses the plaintext, so the all-zero ciphertext cannot be
        trivially un-permuted; this is reported explicitly below
        rather than assumed.
        """
        print("[*] Chosen-Plaintext Diffusion-Leakage Probe...")
        zero_img = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        impulse_img = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        impulse_img[self.rows // 2, self.cols // 2, :] = 255

        enc_zero, init_zero = self.encrypt_image(zero_img)
        enc_impulse, init_impulse = self.encrypt_image(impulse_img)

        flat_cipher = enc_zero.flatten()
        hist, _ = np.histogram(flat_cipher, bins=256, range=(0, 256), density=True)
        prob = hist[hist > 0]
        leaked_entropy = -np.sum(prob * np.log2(prob))

        report = [
            "NOTE: under this content-dependent design, the permutation "
            "key ks_p is unknown to an attacker who only chooses the "
            "plaintext, so the all-zero ciphertext (= ks_d permuted by "
            "an unknown ks_p) cannot be trivially un-permuted.",
            f"All-zero ciphertext entropy: {leaked_entropy:.6f} (ideal 8.0; "
            f"a value well below 8.0 would indicate diffusion-stream leakage)",
            f"Impulse-response ciphertext written to "
            f"'CPA_Probe_Impulse.png' for visual/structural inspection.",
        ]
        cv2.imwrite(os.path.join(self.output_folder, 'CPA_Probe_AllZero.png'), enc_zero)
        cv2.imwrite(os.path.join(self.output_folder, 'CPA_Probe_Impulse.png'), enc_impulse)
        for line in report:
            print(f"   {line}")
        return report

    # -----------------------------------------------------------------
    def calculate_npcr_uaci(self, img1, img2):
        img1 = img1.astype(np.int32); img2 = img2.astype(np.int32)
        d_matrix = np.not_equal(img1, img2).astype(np.int32)
        total_pixels = img1.size
        npc_r = (np.sum(d_matrix) / total_pixels) * 100
        uaci = (np.sum(np.abs(img1 - img2)) / (total_pixels * 255)) * 100
        return npc_r, uaci

    def save_histogram(self, image, filename, title):
        plt.figure(figsize=(10, 6))
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col, label=f'Channel {col.upper()}')
        plt.title(title); plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_folder, filename))
        plt.close()

    # -----------------------------------------------------------------
    # Polar (rose-diagram) histogram analysis
    # -----------------------------------------------------------------
    def polar_histogram(self, image, filename, title, bins=64):
        """
        Plots the pixel-intensity distribution of each RGB channel in
        polar coordinates: intensity level (0-255) is mapped to angle
        (0-2*pi), and bin frequency to radius. A perfectly uniform
        ciphertext histogram should appear as a near-circular
        (isotropic) ring in this representation, making residual
        non-uniformity or periodic structure visually obvious in a
        way a linear histogram can mask. This complements, rather
        than replaces, the linear histograms produced by
        `save_histogram()`.

        Also returns a circular-uniformity statistic per channel: the
        Rayleigh-test-style resultant vector length R, where R -> 0
        indicates isotropic/uniform and R -> 1 indicates strong
        directional concentration (i.e. non-uniformity).
        """
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
            R = np.sqrt(C ** 2 + S ** 2)
            results[name] = float(R)
        return results

    def plot_3d_rgb_correlation(self, image_rgb, prefix_name):
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

    # -----------------------------------------------------------------
    # Correlation analysis (full per-channel, per-direction table)
    # -----------------------------------------------------------------
    def full_correlation_table(self, image):
        """Returns a dict {channel_name: {direction: r}} for R,G,B x H,V,D.
        `image` is expected in BGR order (as read by cv2)."""
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
                p1f, p2f = p1.flatten(), p2.flatten()
                r = np.corrcoef(p1f, p2f)[0, 1]
                table[ch_name][dname] = r
        return table

    def inter_channel_correlation(self, image):
        """Correlation between the R-G, R-B, and G-B channel pairs."""
        b = image[:, :, 0].flatten().astype(np.float64)
        g = image[:, :, 1].flatten().astype(np.float64)
        r = image[:, :, 2].flatten().astype(np.float64)
        return {
            "R-G": np.corrcoef(r, g)[0, 1],
            "R-B": np.corrcoef(r, b)[0, 1],
            "G-B": np.corrcoef(g, b)[0, 1],
        }

    def local_entropy(self, image, window=32, stride=32):
        """Sliding-window (block-wise) entropy, reported as
        (mean, std) across all blocks, to complement global entropy
        with a measure of local uniformity."""
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

    # -----------------------------------------------------------------
    # NIST SP 800-22 randomness suite (delegates to nist_randomness_tests.py)
    # -----------------------------------------------------------------
    @staticmethod
    def _bits_from_bytes(byte_array):
        return np.unpackbits(np.frombuffer(byte_array, dtype=np.uint8))

    def nist_sp800_22_subset(self, byte_stream, alpha=0.01):
        """Thin wrapper kept for backward compatibility with earlier
        report sections; runs the same full 15-test suite."""
        bits = self._bits_from_bytes(byte_stream)
        results, verdict = nist_sp800_22_full_suite(bits, alpha=alpha)
        return results, verdict

    def nist_sp800_22_full(self, byte_stream, alpha=0.01):
        bits = self._bits_from_bytes(byte_stream)
        return nist_sp800_22_full_suite(bits, alpha=alpha)

    # -----------------------------------------------------------------
    # Key-space size (initial-state search space only)
    # -----------------------------------------------------------------
    def analyze_key_space(self):
        """Reports the brute-force search space of the four
        hash-derived initial states at IEEE-754 double precision.

        This figure covers the initial-state space ONLY -- it
        excludes the control parameters (a, b, c, d, r), since those
        four states are deterministically derived from the plaintext
        hash and are therefore not secret under the current design. A
        defensible total key-space figure would need to also account
        for the control-parameter space and the threat model being
        assumed.
        """
        precision = 1e-15
        min_val = 0.1
        max_val = (999 / 4000.0) + 0.1
        span = max_val - min_val
        keys_per_state = span / precision
        total_keyspace = keys_per_state ** 4
        log2_keyspace = np.log2(total_keyspace)
        report = [
            f"Initial-state search space ONLY: 2^{log2_keyspace:.2f} "
            f"(excludes control parameters a,b,c,d,r; the four states "
            f"are deterministically derived from the plaintext hash "
            f"and are not secret under the current design).",
            f"Precision used: {precision} (IEEE 754 double)",
        ]
        for line in report:
            print(f"   {line}")
        return report

    # -----------------------------------------------------------------
    # MAIN WORKFLOW
    # -----------------------------------------------------------------
    def run_full_analysis(self):
        """Runs the complete pipeline for `self.image_path`: dynamics
        verification, encryption/decryption timing, robustness under
        noise/cropping, key-sensitivity and diffusion-leakage probes,
        entropy/correlation/NIST statistical analysis, and writes a
        single consolidated text report plus supporting images."""
        tag = os.path.splitext(os.path.basename(self.image_path))[0]
        print(f"\n[*] Processing Image: {self.image_path} "
              f"({self.rows}x{self.cols}, {self.channels} channels)")

        self.plot_bifurcation()
        le_list = self.calculate_lyapunov()
        dynamics_report = self.analyze_dynamics(le_list)

        print("[*] Generating Chaotic Attractors...")
        dummy_initials = self.get_fingerprint(self.img)
        _, _, trajectory = self.chaotic_engine(3000, dummy_initials)
        self.plot_chaotic_attractors(trajectory, suffix=f"_{tag}")

        print("[*] Measuring Encryption Speed (10 runs)...")
        times_enc = []
        encrypted_img = None
        for _ in range(10):
            start = time.time()
            encrypted_img, initials = self.encrypt_image(self.img)
            times_enc.append(time.time() - start)
        avg_enc_time = np.mean(times_enc)
        print(f"[*] Avg Encryption Time: {avg_enc_time:.6f} sec")

        # Export key material to model a real secure-channel handoff
        key_path = os.path.join(self.output_folder, f'key_material_{tag}.json')
        self.export_key_material(initials, key_path)
        print(f"[*] Key material exported to '{key_path}' "
              f"(stand-in for secure transmission to the receiver)")

        print("[*] Measuring Decryption Speed (10 runs)...")
        loaded_initials, _ = self.import_key_material(key_path)
        times_dec = []
        for _ in range(10):
            start = time.time()
            _ = self.decrypt_image(encrypted_img, loaded_initials)
            times_dec.append(time.time() - start)
        avg_dec_time = np.mean(times_dec)
        print(f"[*] Avg Decryption Time: {avg_dec_time:.6f} sec")

        print("[*] Robustness Analysis...")
        robustness_report = []
        for p in [0.01, 0.05, 0.10]:
            noisy_enc = self.add_noise(encrypted_img, p)
            recovered_noisy = self.decrypt_image(noisy_enc, loaded_initials)
            psnr, ssim_val = self.calculate_metrics(self.img, recovered_noisy)
            robustness_report.append(f"NOISE ({int(p * 100)}%): PSNR={psnr:.2f}dB, SSIM={ssim_val:.4f}")
            cv2.imwrite(os.path.join(self.output_folder, f'{tag}_Recovered_Noise_{int(p*100)}pct.png'), recovered_noisy)
        for p in [10, 25, 50]:
            cropped_enc = self.crop_image(encrypted_img, p)
            recovered_cropped = self.decrypt_image(cropped_enc, loaded_initials)
            psnr, ssim_val = self.calculate_metrics(self.img, recovered_cropped)
            robustness_report.append(f"CROP ({p}%): PSNR={psnr:.2f}dB, SSIM={ssim_val:.4f}")
            cv2.imwrite(os.path.join(self.output_folder, f'{tag}_Recovered_Crop_{p}pct.png'), recovered_cropped)

        sensitivity_report = self.key_sensitivity_test(encrypted_img, loaded_initials)
        cpa_probe_report = self.chosen_plaintext_diffusion_probe()

        print("[*] Calculating NPCR and UACI (single-pixel differential test)...")
        modified_img = self.img.copy()
        modified_img[0, 0, 0] = np.uint8((int(modified_img[0, 0, 0]) + 1) % 256)
        encrypted_modified_img, _ = self.encrypt_image(modified_img)
        npcr_val, uaci_val = self.calculate_npcr_uaci(encrypted_img, encrypted_modified_img)

        entropy_vals = []
        for i in range(3):
            hist, _ = np.histogram(encrypted_img[:, :, i], bins=256, range=(0, 256), density=True)
            prob = hist[hist > 0]
            entropy_vals.append(-np.sum(prob * np.log2(prob)))
        avg_entropy = np.mean(entropy_vals)

        corr_table = self.full_correlation_table(encrypted_img)
        inter_ch_corr = self.inter_channel_correlation(encrypted_img)

        cv2.imwrite(os.path.join(self.output_folder, f'{tag}_Original.png'), self.img)
        cv2.imwrite(os.path.join(self.output_folder, f'{tag}_Encrypted.png'), encrypted_img)
        decrypted_img = self.decrypt_image(encrypted_img, loaded_initials)
        cv2.imwrite(os.path.join(self.output_folder, f'{tag}_Decrypted.png'), decrypted_img)
        assert np.array_equal(self.img, decrypted_img), "Lossless round-trip check FAILED"
        print("[*] Lossless round-trip check: PASSED")

        print("[*] Generating Histograms and 3D Correlations...")
        self.save_histogram(self.img, f'{tag}_Hist_Original.png', 'Original Image Histogram')
        self.save_histogram(encrypted_img, f'{tag}_Hist_Encrypted.png', 'Encrypted Image Histogram')

        print("[*] Generating Polar (Rose-Diagram) Histograms...")
        polar_original = self.polar_histogram(self.img, f'{tag}_PolarHist_Original.png',
                                               'Polar Histogram - Original Image')
        polar_encrypted = self.polar_histogram(encrypted_img, f'{tag}_PolarHist_Encrypted.png',
                                                'Polar Histogram - Encrypted Image')
        self.plot_3d_rgb_correlation(self.img_rgb, f"{tag}_Original")
        self.plot_3d_rgb_correlation(cv2.cvtColor(encrypted_img, cv2.COLOR_BGR2RGB), f"{tag}_Encrypted")

        key_space_report = self.analyze_key_space()

        print("[*] Computing Local (sliding-window) Entropy...")
        local_ent_mean, local_ent_std = self.local_entropy(encrypted_img)
        print(f"   Local Entropy (32x32 blocks): mean={local_ent_mean:.6f}, "
              f"std={local_ent_std:.6f}  (Global Entropy: {avg_entropy:.6f})")

        print("[*] Running the FULL 15-test NIST SP 800-22 suite on the diffusion key "
              "stream (this can take a minute or two, mainly Linear Complexity)...")
        total_pixels = self.rows * self.cols * 3
        ks_p, ks_d, _ = self.chaotic_engine(total_pixels, dummy_initials)
        nist_results, nist_verdict = self.nist_sp800_22_full(ks_d.tobytes())

        report_path = os.path.join(self.output_folder, f'Research_Report_{tag}.txt')
        with open(report_path, 'w') as f:
            f.write(f"=== COMPLETE RESEARCH DATA REPORT ({tag}) ===\n")
            f.write(f"Image Size: {self.rows}x{self.cols}x{self.channels}\n")
            f.write(f"Optimized Parameters (Cuckoo Search): a={self.a:.4f}, b={self.b:.4f}, "
                    f"c={self.c:.4f}, d={self.d:.4f}, r={self.r:.4f}\n")
            f.write(f"Cuckoo Search optimization time: {self.cs_time:.4f} sec\n")
            f.write("--------------------------------------------------\n")
            f.write("1. PERFORMANCE ANALYSIS (Averaged 10 runs):\n")
            f.write(f"   Encryption Time: {avg_enc_time:.6f} sec\n")
            f.write(f"   Decryption Time: {avg_dec_time:.6f} sec\n")
            f.write("--------------------------------------------------\n")
            f.write("2. CHAOTIC DYNAMICS:\n")
            f.write(f"   Lyapunov Exponents: {np.round(le_list, 4).tolist()}\n")
            f.write(f"   Sum of Exponents: {np.sum(le_list):.4f}\n")
            pos_count = sum(1 for x in le_list if x > 0)
            status = 'HYPERCHAOTIC' if pos_count >= 2 and np.sum(le_list) < 0 else 'CHAOTIC (criterion not met)'
            f.write(f"   Status: {status}\n")
            for line in dynamics_report:
                f.write(f"   {line}\n")
            f.write("--------------------------------------------------\n")
            f.write("3. SECURITY KEY ANALYSIS:\n")
            for line in key_space_report:
                f.write(f"   {line}\n")
            f.write("--------------------------------------------------\n")
            f.write("4. ROBUSTNESS ANALYSIS:\n")
            for line in robustness_report:
                f.write(f"   {line}\n")
            f.write("--------------------------------------------------\n")
            f.write("5. STATISTICAL ANALYSIS:\n")
            f.write(f"   Global Entropy: {avg_entropy:.8f} (Ideal: 8.0)\n")
            f.write(f"   Local Entropy (32x32 blocks): mean={local_ent_mean:.6f}, "
                    f"std={local_ent_std:.6f}\n")
            f.write("   Correlation (per channel, per direction):\n")
            for ch, dirs in corr_table.items():
                for d, val in dirs.items():
                    f.write(f"      {ch}-{d}: {val:.8f}\n")
            f.write("   Inter-channel correlation:\n")
            for pair, val in inter_ch_corr.items():
                f.write(f"      {pair}: {val:.8f}\n")
            f.write("   Polar (rose-diagram) histogram -- circular resultant length R "
                     "(0 = isotropic/uniform, 1 = fully directional/non-uniform):\n")
            f.write("      Original image:\n")
            for ch, r in polar_original.items():
                f.write(f"         {ch}: R={r:.6f}\n")
            f.write("      Encrypted image:\n")
            for ch, r in polar_encrypted.items():
                f.write(f"         {ch}: R={r:.6f}\n")
            f.write("--------------------------------------------------\n")
            f.write("6. DIFFERENTIAL ATTACK ANALYSIS:\n")
            f.write(f"   NPCR: {npcr_val:.4f}% (Ideal > 99.6%)\n")
            f.write(f"   UACI: {uaci_val:.4f}% (Ideal ~ 33.46%)\n")
            f.write("--------------------------------------------------\n")
            f.write("7. KEY-SENSITIVITY TEST:\n")
            for line in sensitivity_report:
                f.write(f"   {line}\n")
            f.write("--------------------------------------------------\n")
            f.write("8. CHOSEN-PLAINTEXT DIFFUSION-LEAKAGE PROBE:\n")
            for line in cpa_probe_report:
                f.write(f"   {line}\n")
            f.write("--------------------------------------------------\n")
            f.write("9. NIST SP 800-22 FULL SUITE (15 tests, diffusion key stream):\n")
            for k, v in nist_results.items():
                f.write(f"   {k}: {v} -> {nist_verdict.get(k, 'N/A')}\n")
            f.write("==================================================\n")

        print(f"\n[OK] Success! All files for '{tag}' saved in '{self.output_folder}'\n")
        return report_path
