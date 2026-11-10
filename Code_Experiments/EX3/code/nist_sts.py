

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np

STS_DIR = Path(__file__).resolve().parent.parent / "third_party" / "sts-2.1.2"
TESTS = ["Frequency", "BlockFrequency", "Runs", "LongestRun", "Rank", "FFT", "NonOverlappingTemplate",
         "OverlappingTemplate", "Universal", "LinearComplexity", "Serial", "ApproximateEntropy",
         "CumulativeSums", "RandomExcursions", "RandomExcursionsVariant"]
ALPHA = 0.01


def ensure_built() -> Path:
    exe = STS_DIR / "assess"
    if not exe.exists():
        (STS_DIR / "obj").mkdir(exist_ok=True)
        subprocess.run(["make", "-s"], cwd=STS_DIR, check=True, capture_output=True)
    return exe


def _read(path: Path):
    return [float(v) for v in path.read_text().split()] if path.exists() else []


def run_sts(bits: np.ndarray) -> Dict:
    """Run all 15 STS tests on one bit sequence; return p-values."""
    exe = ensure_built()
    bits = np.asarray(bits, dtype=np.uint8)
    n = int(bits.size)
    with tempfile.TemporaryDirectory(prefix="sts_") as tmp:
        tmp = Path(tmp)
        os.symlink(STS_DIR / "templates", tmp / "templates")
        for t in TESTS:
            (tmp / "experiments" / "AlgorithmTesting" / t).mkdir(parents=True)
        (tmp / "seq.txt").write_bytes((bits + ord("0")).tobytes())
        # 0: input file | path | 1: apply all tests | 0: keep default parameters
        # | 1 bitstream | 0: ASCII
        stdin = "0\nseq.txt\n1\n0\n1\n0\n"
        # NB: STS 2.1.2 main() returns 1 on success, so the exit code is not used.
        proc = subprocess.run([str(exe), str(n)], cwd=tmp, input=stdin, text=True, capture_output=True)
        base = tmp / "experiments" / "AlgorithmTesting"
        p = {t: _read(base / t / "results.txt") for t in TESTS}
        if "Statistical Testing Complete" not in proc.stdout or not p["Frequency"]:
            raise RuntimeError("NIST STS failed: " + (proc.stdout + proc.stderr)[-800:])
        stats = {t: (base / t / "stats.txt").read_text() if (base / t / "stats.txt").exists() else ""
                 for t in ("RandomExcursions", "RandomExcursionsVariant")}
    cycles = re.search(r"Number Of Cycles \(J\)\s*=\s*(\d+)", stats["RandomExcursions"])
    re_ok = "NOT APPLICABLE" not in stats["RandomExcursions"]
    rev_ok = "NOT APPLICABLE" not in stats["RandomExcursionsVariant"]

    def g(t, i=0):
        return p[t][i] if len(p[t]) > i else None

    core = {
        "01 Frequency": g("Frequency"),
        "02 Block Frequency": g("BlockFrequency", 0),
        "03 Runs": g("Runs", 0),
        "04 Longest Run of Ones": g("LongestRun", 0),
        "05 Binary Matrix Rank": g("Rank", 0),
        "06 DFT (Spectral)": g("FFT", 0),
        "07 Non-overlapping Template (000000001)": g("NonOverlappingTemplate", 0),
        "08 Overlapping Template": g("OverlappingTemplate", 0),
        "09 Maurer Universal": g("Universal", 0),
        "10 Linear Complexity": g("LinearComplexity", 0),
        "11a Serial (p1)": g("Serial", 0),
        "11b Serial (p2)": g("Serial", 1),
        "12 Approximate Entropy": g("ApproximateEntropy", 0),
        "13a Cumulative Sums (forward)": g("CumulativeSums", 0),
        "13b Cumulative Sums (reverse)": g("CumulativeSums", 1),
    }
    excursion = {}
    if re_ok:
        for s, v in zip([-4, -3, -2, -1, 1, 2, 3, 4], p["RandomExcursions"]):
            excursion[f"14 Random Excursions (x={s:+d})"] = v
    if rev_ok:
        for s, v in zip([s for s in range(-9, 10) if s != 0], p["RandomExcursionsVariant"]):
            excursion[f"15 Random Excursions Variant (x={s:+d})"] = v
    missing = [k for k, v in core.items() if v is None]
    applicable = {k: v for k, v in {**core, **excursion}.items() if v is not None}
    passed = sum(v >= ALPHA for v in applicable.values())
    nont = p["NonOverlappingTemplate"]
    return {
        "n_bits": n,
        "cycles_J": int(cycles.group(1)) if cycles else None,
        "random_excursions_applicable": re_ok and rev_ok,
        "p_values": applicable,
        "applicable": len(applicable),
        "passed": int(passed),
        "not_applicable": len(missing) + (0 if (re_ok and rev_ok) else 2),  # tests 14 and 15 count as 2 N/A
        "not_applicable_core": missing,
        "nonoverlapping_all_templates": {"templates": len(nont),
                                         "passed": int(sum(v >= ALPHA for v in nont))},
    }


