import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUTPUTS = ROOT / "outputs"


def main():
    os.chdir(str(ROOT))

    separator = ";" if sys.platform == "win32" else ":"

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        "--name", "ALNet_Screening_Tool",
        "--add-data", f"{OUTPUTS / 'alnet_best.pt'}{separator}outputs",
        "--add-data", f"{OUTPUTS / 'alnet_final.pt'}{separator}outputs",
        "--add-data", f"{SRC / 'session_logger.py'}{separator}.",
        "--add-data", f"{SRC / 'alnet_model.py'}{separator}.",
        "--hidden-import", "torch",
        "--hidden-import", "torchvision",
        "--hidden-import", "torchvision.transforms",
        "--hidden-import", "PIL",
        "--hidden-import", "PIL.Image",
        "--hidden-import", "customtkinter",
        "--hidden-import", "numpy",
        "--hidden-import", "sqlite3",
        "--collect-all", "torchvision",
        "--clean",
        "--noconfirm",
        str(SRC / "desktop_app.py"),
    ]

    print("Building ALNet Screening Tool executable ...")
    subprocess.run(cmd, check=True)

    dist = ROOT / "dist" / "ALNet_Screening_Tool.exe"
    if dist.exists():
        size_mb = dist.stat().st_size / (1024 * 1024)
        print(f"\nBuild successful: {dist} ({size_mb:.1f} MB)")
    else:
        print("\nBuild may have failed — check dist/ directory")


if __name__ == "__main__":
    main()
