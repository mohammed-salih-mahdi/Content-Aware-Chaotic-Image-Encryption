"""
cuckoo_search.py
-------------------------------------------------------------------
Cuckoo Search metaheuristic used to tune the control parameters
(a, b, c, d, r) of the 4D hyperchaotic system defined in
`encryption_system.py`, so that the resulting Lyapunov spectrum
satisfies the hyperchaos criterion: the sum of the four exponents
must be negative (dissipative system) while at least two of the
exponents must be positive.

Candidate generation uses a genuine Levy-flight step (Mantegna's
algorithm) rather than a plain Gaussian perturbation, which gives the
search occasional long jumps in parameter space and helps it escape
local optima.

This module has no dependency on the image-encryption code -- it can
be imported and tested on its own.
"""

import math

import numpy as np


class CuckooSearch:
    """
    Searches the 5-D parameter space (a, b, c, d, r) for a
    configuration of the 4D hyperchaotic system that is both
    dissipative and genuinely hyperchaotic (>= 2 positive Lyapunov
    exponents, negative exponent sum).
    """

    def __init__(self, pop_size=10, pa=0.25, max_iter=5, levy_beta=1.5, seed=None):
        self.pop_size = pop_size
        self.pa = pa                    # fraction of nests abandoned each generation
        self.max_iter = max_iter
        self.levy_beta = levy_beta
        # Search bounds for (a, b, c, d, r)
        self.lb = np.array([30, 2, 20, 10, 0.1])
        self.ub = np.array([40, 5, 35, 35, 1.5])
        self.rng = np.random.default_rng(seed)

    # ---- 4D hyperchaotic system, using all five control parameters ----
    @staticmethod
    def _derivatives(x, y, z, w, a, b, c, d, r):
        dx = a * (y - x) + w
        dy = c * x - y - x * z
        dz = x * y - b * z
        dw = -d * y + r * w
        return dx, dy, dz, dw

    @staticmethod
    def _jacobian(x, y, z, w, a, b, c, d, r):
        return np.array([
            [-a,     a,  0.0, 1.0],
            [c - z, -1.0, -x, 0.0],
            [y,      x,  -b, 0.0],
            [0.0,   -d, 0.0,   r],
        ])

    def get_lyapunov_spectrum(self, params, quick_steps=2000, dt_lyap=0.002):
        """Fast (low-precision) Lyapunov spectrum estimate used only
        inside the search loop, where speed matters more than the
        high-precision estimate computed later for the final report."""
        a, b, c, d, r = params
        x, y, z, w = -0.1, 0.5, -0.6, 0.2
        W = np.eye(4, dtype=np.float64)
        LEs = np.zeros(4, dtype=np.float64)

        for _ in range(quick_steps):
            dx, dy, dz, dw = self._derivatives(x, y, z, w, a, b, c, d, r)
            x += dx * dt_lyap; y += dy * dt_lyap; z += dz * dt_lyap; w += dw * dt_lyap
            if np.isnan(x) or np.abs(x) > 100:
                return np.full(4, -np.inf)
            J = self._jacobian(x, y, z, w, a, b, c, d, r)
            try:
                W = W + np.dot(J, W) * dt_lyap
                Q, Rm = np.linalg.qr(W)
                LEs += np.log(np.abs(np.diag(Rm))) / dt_lyap
                W = Q
            except np.linalg.LinAlgError:
                return np.full(4, -np.inf)
        return LEs / quick_steps

    def fitness(self, params):
        """
        Hyperchaos-constrained fitness function.

        Rewards configurations that (a) have a negative exponent sum
        (dissipative) and (b) have at least two positive exponents --
        the hyperchaos criterion -- instead of a naive "maximize the
        raw sum of exponents" objective, which can drift into an
        unstable, volume-expanding regime that is not dissipative.
        """
        LEs = self.get_lyapunov_spectrum(params)
        if np.any(np.isneginf(LEs)) or np.any(np.isnan(LEs)):
            return -1e9
        total = np.sum(LEs)
        pos_count = int(np.sum(LEs > 0))

        if total >= 0:
            # Hard penalty: violates the required negative-sum (dissipative) condition.
            return -1e6 + total  # still ranks "less positive" sums as less bad
        if pos_count < 2:
            # Soft penalty: not genuinely hyperchaotic; encourage more separation.
            return total - 50.0 * (2 - pos_count)

        # Valid hyperchaotic candidate: reward larger Kaplan-Yorke-style
        # separation (sum of the two largest exponents) while remaining
        # dissipative overall.
        sorted_les = np.sort(LEs)[::-1]
        return float(sorted_les[0] + sorted_les[1])

    # ---- Mantegna's algorithm for Levy-distributed steps -----------
    def _levy_step(self, size):
        beta = self.levy_beta
        sigma_u = (
            math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
            (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = self.rng.normal(0, sigma_u, size)
        v = self.rng.normal(0, 1, size)
        return u / (np.abs(v) ** (1 / beta))

    def optimize(self, verbose=True):
        """Runs the Cuckoo Search loop and returns (best_params, best_fitness)."""
        if verbose:
            print("[*] Initializing Cuckoo Search (Levy-flight, constrained objective)...")
        nests = self.rng.uniform(self.lb, self.ub, (self.pop_size, 5))
        fitness_vals = np.array([self.fitness(n) for n in nests])
        best_idx = np.argmax(fitness_vals)
        best_nest, best_fitness = nests[best_idx].copy(), fitness_vals[best_idx]

        bound_span = self.ub - self.lb
        step_scale = 0.01  # standard CS step-size scaling factor

        for t in range(self.max_iter):
            # Generate new solutions via Levy flights
            for i in range(self.pop_size):
                levy = self._levy_step(5)
                step = step_scale * levy * bound_span
                new_nest = np.clip(nests[i] + step, self.lb, self.ub)
                new_fit = self.fitness(new_nest)
                if new_fit > fitness_vals[i]:
                    nests[i], fitness_vals[i] = new_nest, new_fit

            # Abandon a fraction of the worst nests and replace them randomly
            num_abandon = int(self.pa * self.pop_size)
            if num_abandon > 0:
                worst = np.argsort(fitness_vals)[:num_abandon]
                for i in worst:
                    nests[i] = self.rng.uniform(self.lb, self.ub, 5)
                    fitness_vals[i] = self.fitness(nests[i])

            gen_best_idx = np.argmax(fitness_vals)
            if fitness_vals[gen_best_idx] > best_fitness:
                best_fitness = fitness_vals[gen_best_idx]
                best_nest = nests[gen_best_idx].copy()
            if verbose:
                print(f"[*] Gen {t + 1}/{self.max_iter} | Best fitness: {best_fitness:.4f}")

        return best_nest, best_fitness
