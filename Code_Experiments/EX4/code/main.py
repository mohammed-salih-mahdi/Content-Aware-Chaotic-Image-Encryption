
import json
import time
import traceback
from pathlib import Path

from encryption_system_v4 import FullResearchEncryptionSystem

RESEARCH_KEY_HEX = "70777f6c8316f14bc70dab110718774aa7b105c560e41f561ba12365865c78af"

BENCHMARK_IMAGES = ["Baboon.png", "Peppers.png", "Car.png", "Brain.png",
                     "Chest.png", "Lung.png", "Two Macaws.png"]
WORST_CASE_IMAGES = ["black.png", "white.png", "gradient.png"]


def run(image_files, output_folder="Final_Research_Data", secret_key_hex=RESEARCH_KEY_HEX):
    key = bytes.fromhex(secret_key_hex) if secret_key_hex else None
    reports = []
    for f in image_files:
        if not Path(f).exists():
            print("[SKIP]", f); continue
        try:
            system = FullResearchEncryptionSystem(f, output_folder, secret_key=key)
            report = system.run_full_analysis()
            reports.append(report)
            print("[OK]", report)
        except Exception as e:
            print(f"[ERROR] processing '{f}': {e}")
            traceback.print_exc()
    return reports


if __name__ == "__main__":
    t0 = time.time()
    all_reports = run(BENCHMARK_IMAGES + WORST_CASE_IMAGES)
    print("\nTOTAL", round(time.time() - t0, 1), "s")
    print("REPORTS:", all_reports)
