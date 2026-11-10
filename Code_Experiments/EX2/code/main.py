from pathlib import Path
from encryption_system import FullResearchEncryptionSystem

IMAGE_FILES = [
    'Baboon.png', 'Peppers.png', 'Car.png', 'Brain.png',
    'Chest.png', 'Lung.png', 'Two Macaws.png'
]

def run(files=IMAGE_FILES, output_folder='Final_Research_Data', secret_key_hex=None):
    key = bytes.fromhex(secret_key_hex) if secret_key_hex else None
    reports = []
    for f in files:
        if not Path(f).exists():
            print('[SKIP]', f); continue
        system = FullResearchEncryptionSystem(f, output_folder, secret_key=key)
        report = system.run_full_analysis()
        reports.append(report)
        print('[OK]', report)
    return reports


run()
