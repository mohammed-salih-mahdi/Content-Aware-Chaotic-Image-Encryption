"""
main.py
-------------------------------------------------------------------
Entry point. Point IMAGE_FILES at one or more images (any size,
grayscale or color) and run the full research/benchmarking pipeline
on each of them.

Example:
    IMAGE_FILES = ["Baboon.png", "Peppers.png", "Car.png",etc]
"""

import traceback

from encryption_system import FullResearchEncryptionSystem


def run(image_files, output_folder="Final_Research_Data", cs_seed=None):
    """Runs the full analysis pipeline on each image in `image_files`.
    Skips missing files and continues past per-image errors so a
    batch run isn't aborted by a single bad input."""
    report_paths = []
    for path in image_files:
        try:
            research = FullResearchEncryptionSystem(path, output_folder=output_folder, cs_seed=cs_seed)
            report_paths.append(research.run_full_analysis())
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")
        except Exception as e:
            print(f"[ERROR] processing '{path}': {e}")
            traceback.print_exc()
    return report_paths


if __name__ == "__main__":
    IMAGE_FILES = ["gradient.png"]  # demo run
    run(IMAGE_FILES)
