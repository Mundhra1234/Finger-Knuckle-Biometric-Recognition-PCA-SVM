# 1_setup.py - Run this first to install packages
import sys
import subprocess
import importlib

def install_package(package):
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f" {package} installed!")
        return True
    except:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])
            print(f" {package} installed with --user!")
            return True
        except:
            print(f" Could not install {package}")
            return False

def test_import(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        importlib.import_module(import_name)
        print(f" {package_name} import successful")
        return True
    except ImportError as e:
        print(f" {package_name} import failed: {e}")
        return False

print("  KNUCKLE BIOMETRIC SYSTEM SETUP")
print("=" * 50)

packages = [
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("joblib", "joblib")
]

print("Installing packages...")
for package, import_name in packages:
    install_package(package)

print("Testing imports...")
for package, import_name in packages:
    test_import(package, import_name)

print(" Setup completed! Run: python main_knuckle.py")