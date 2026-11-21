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
randomness .

This repository contains every version of the code used across the review and revision process for the associated manuscript (PONE-D-26-20680), from the original Round 1 implementation through the current revision. Each version is self-contained — its own code and its own genuine, reproducible run output — so the evolution of the implementation is fully transparent rather than collapsed into a single unlabeled script.

## Installation
```
pip install -r requirements.txt
```

## Usage

This produces:
- Original / encrypted / decrypted images
- Linear and polar histograms
- Bifurcation diagram and chaotic attractor plots
- 3D RGB correlation plots
- Robustness test outputs (noisy/cropped recoveries)
- Key-sensitivity and chosen-plaintext probe outputs
- Exported key material (`key_material_<tag>.json`)
- A consolidated `Research_Report_<tag>.txt` with every metric

## Dependencies
- `numpy`
- `opencv-python`
- `matplotlib`
- `scikit-image`
- `scipy`
