# install_packages.py - Run this first!
import subprocess
import sys

def install_packages():
    packages = [
        "opencv-python",
        "scikit-learn", 
        "scikit-image",
        "matplotlib",
        "numpy",
        "joblib"
    ]
    
    print("🔧 Installing required packages...")
    print("This may take a few minutes...")
    
    for package in packages:
        try:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully!")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            print("Trying with --user flag...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])
                print(f"✅ {package} installed successfully with --user!")
            except:
                print(f"❌ Could not install {package}")

if __name__ == "__main__":
    install_packages()
    print("\n🎉 Installation completed! Now run your knuckle system.")