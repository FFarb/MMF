"""
Reliable environment installation script for QFC Trading System.
Designed for Vast.AI and cloud environments with dependency validation.
"""
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Ensure Python 3.10+ is being used."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ ERROR: Python 3.10+ required, found {version.major}.{version.minor}")
        sys.exit(1)
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")

def run_pip_install(packages, description="packages"):
    """Run pip install with error handling."""
    print(f"\n[INSTALLING] {description}...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade"] + packages,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"✓ {description} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {description}")
        print(f"   Error: {e}")
        return False

def install_from_requirements():
    """Install from requirements.txt with special handling for pandas-ta."""
    req_file = Path("requirements.txt")
    
    if not req_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    # Read requirements
    with open(req_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Separate pandas-ta from other requirements
    pandas_ta_line = None
    other_reqs = []
    
    for line in lines:
        if 'pandas-ta' in line:
            pandas_ta_line = line
        else:
            other_reqs.append(line)
    
    # Install core packages first
    print("\n" + "="*70)
    print("INSTALLING QFC DEPENDENCIES")
    print("="*70)
    
    if other_reqs:
        success = run_pip_install(other_reqs, "core dependencies")
        if not success:
            print("\n⚠️  Some core packages failed. Continuing anyway...")
    
    # Install pandas-ta separately (may fail in some environments)
    if pandas_ta_line:
        print("\n[INSTALLING] pandas-ta (technical analysis library)...")
        print("   Note: This uses git installation and may take longer...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pandas_ta_line],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("✓ pandas-ta installed successfully")
        except subprocess.CalledProcessError:
            print("⚠️  pandas-ta git installation failed")
            print("   Trying PyPI fallback version...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "pandas-ta==0.3.14b"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                print("✓ pandas-ta (PyPI version) installed")
            except subprocess.CalledProcessError:
                print("❌ pandas-ta installation failed completely")
                print("   You may need to install manually: pip install pandas-ta")
    
    return True

def verify_imports():
    """Verify critical imports work."""
    print("\n" + "="*70)
    print("VERIFYING INSTALLATIONS")
    print("="*70)
    
    critical_imports = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "scikit-learn"),
        ("torch", "PyTorch"),
        ("xgboost", "XGBoost"),
        ("pybit", "Bybit API"),
    ]
    
    optional_imports = [
        ("pandas_ta", "pandas-ta"),
    ]
    
    all_ok = True
    
    for module, name in critical_imports:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name} - CRITICAL")
            all_ok = False
    
    for module, name in optional_imports:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"⚠️  {name} - OPTIONAL (some features may not work)")
    
    return all_ok

def main():
    """Main installation workflow."""
    print("="*70)
    print("QFC TRADING SYSTEM - ENVIRONMENT INSTALLER")
    print("="*70)
    
    # Step 1: Check Python version
    check_python_version()
    
    # Step 2: Upgrade pip
    print("\n[UPGRADING] pip...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("✓ pip upgraded")
    except subprocess.CalledProcessError:
        print("⚠️  pip upgrade failed, continuing anyway...")
    
    # Step 3: Install dependencies
    success = install_from_requirements()
    
    if not success:
        print("\n❌ Installation encountered errors")
        sys.exit(1)
    
    # Step 4: Verify installations
    print()
    if verify_imports():
        print("\n" + "="*70)
        print("✓ INSTALLATION COMPLETE - ALL CRITICAL PACKAGES VERIFIED")
        print("="*70)
        print("\nYou can now run:")
        print("  python run_deep_research.py")
        print("  python test_multi_asset.py")
    else:
        print("\n" + "="*70)
        print("⚠️  INSTALLATION COMPLETE WITH WARNINGS")
        print("="*70)
        print("\nSome critical packages failed to install.")
        print("Please review errors above and install manually if needed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
