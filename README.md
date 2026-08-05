# Content-Aware Chaotic Image Encryption

Image encryption using a Cuckoo Search-optimized 4D hyperchaotic
(Chen-type) system for keystream generation. The scheme is
content-aware: a SHA-256 hash of the plaintext seeds the chaotic
system's initial states, so a single-bit change in the image produces
a completely different keystream.

The repository also includes a full research/benchmarking pipeline:
performance timing, robustness under noise and cropping, key
sensitivity, a chosen-plaintext diffusion-leakage probe, entropy and
correlation analysis, and the complete 15-test NIST SP 800-22
randomness  .

## Project structure

```
.
├── nist_randomness_tests.py   # Standalone NIST SP 800-22 test suite (15 tests)
├── cuckoo_search.py           # Cuckoo Search optimizer for the chaotic system's parameters
├── encryption_system.py       # Main encryption/decryption + analysis pipeline
├── main.py                    # Entry point / example usage
├── requirements.txt
└── README.md
```

### `nist_randomness_tests.py`
A self-contained implementation of the NIST SP 800-22 Rev. 1a
statistical test suite. Operates purely on a numpy bit array, so it
can be reused or unit-tested independently of the encryption code.

### `cuckoo_search.py`
Implements the `CuckooSearch` class, which tunes the 4D hyperchaotic
system's control parameters `(a, b, c, d, r)` so the resulting
Lyapunov spectrum satisfies the hyperchaos criterion (negative
exponent sum, at least two positive exponents). Uses a genuine
Levy-flight step (Mantegna's algorithm) for candidate generation.

### `encryption_system.py`
Implements `FullResearchEncryptionSystem`, which:
- Loads an image and optimizes the chaotic system's parameters once
  (image-independent).
- Derives per-image initial states from a SHA-256 hash of the
  plaintext (content-aware keying).
- Encrypts/decrypts via permutation (pixel shuffle) + diffusion
  (byte-wise XOR) using the chaotic keystream.
- Exports/imports key material as a JSON file, standing in for secure
  out-of-band key transmission.
- Runs the full analysis suite: Lyapunov/Kaplan-Yorke dynamics,
  bifurcation diagrams, encryption/decryption timing, noise/crop
  robustness, key-sensitivity test, chosen-plaintext diffusion probe,
  NPCR/UACI, global and local entropy, full correlation tables
  (per-channel, per-direction, and inter-channel), polar histograms,
  and the NIST SP 800-22   — then writes it all to a single
  text report.

### `main.py`
Example entry point. Point `IMAGE_FILES` at one or more images and
run.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from encryption_system import FullResearchEncryptionSystem

system = FullResearchEncryptionSystem("path/to/image.png")
report_path = system.run_full_analysis()
```

Or from the command line:

```bash
python main.py
```

This produces, inside `Final_Research_Data/` (or your chosen output
folder):
- Original / encrypted / decrypted images
- Linear and polar histograms
- Bifurcation diagram and chaotic attractor plots
- 3D RGB correlation plots
- Robustness test outputs (noisy/cropped recoveries)
- Key-sensitivity and chosen-plaintext probe outputs
- Exported key material (`key_material_<tag>.json`)
- A consolidated `Research_Report_<tag>.txt` with every metric

## Notes on the security model

- The four chaotic initial states are derived deterministically from
  the plaintext hash and are **not** part of the secret key space —
  `analyze_key_space()` reports this explicitly. A defensible total
  key-space figure needs to also account for the control-parameter
  space and the assumed threat model.
- `key_sensitivity_test()` measures how much decryption changes under
  a minimally perturbed key; `chosen_plaintext_diffusion_probe()` is
  a separate, distinct test of whether the diffusion keystream leaks
  structure under chosen plaintexts. They are not interchangeable.

## Dependencies

- `numpy`
- `opencv-python`
- `matplotlib`
- `scikit-image`
- `scipy`
