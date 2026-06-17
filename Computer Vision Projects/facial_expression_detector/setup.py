#!/usr/bin/env python3
"""
Setup script to install dependencies and verify the environment.
Run this before first use: python3.10 setup.py
"""

import subprocess
import sys
import os


def check_python_version():
    """Verify that the Python version is 3.10.x."""
    version = sys.version_info
    if version.major != 3 or version.minor != 10:
        print(f"ERROR: Python 3.10.x required. You have {version.major}.{version.minor}")
        return False
    print(f"\u2713 Python {version.major}.{version.minor} detected")
    return True


def create_and_activate_venv():
    """Create a Python 3.10 virtual environment if it does not exist."""
    venv_dir = "venv3.10"
    if not os.path.isdir(venv_dir):
        print(f"\nCreating virtual environment in '{venv_dir}'...")
        try:
            subprocess.check_call(["python3.10", "-m", "venv", venv_dir])
        except Exception as e:
            print(f"ERROR: Could not create venv: {e}")
            print("Ensure 'python3.10' is available in your PATH.")
            return False
        print(f"\u2713 Virtual environment created at {venv_dir}")
    else:
        print(f"\u2713 Virtual environment '{venv_dir}' already exists.")
    return True


def install_requirements():
    """Install required packages inside the virtual environment."""
    venv_dir = "venv3.10"
    pip_path = os.path.join(venv_dir, "bin", "pip")
    if not os.path.isfile(pip_path):
        print(f"ERROR: pip not found in {venv_dir}. Recreate the virtual environment.")
        return False
    print("\nInstalling dependencies from requirements.txt...")
    try:
        subprocess.check_call([pip_path, "install", "--upgrade", "pip"])
        subprocess.check_call([
            pip_path, "install", "-r", "requirements.txt"
        ])
        print("\u2713 Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies: {e}")
        return False


def verify_imports():
    """Verify that all required packages can be imported."""
    print("\nVerifying imports...")
    required = ["cv2", "mediapipe", "numpy", "PIL"]
    failed = []
    for module in required:
        try:
            __import__(module)
            print(f"\u2713 {module}")
        except ImportError as e:
            print(f"\u2717 {module}: {e}")
            failed.append(module)

    if "mediapipe" not in failed:
        try:
            import mediapipe as mp
            _ = mp.solutions.face_mesh
        except AttributeError as e:
            print("\u2717 mediapipe.solutions not found. Attempting to reinstall mediapipe...")
            venv_dir = "venv3.10"
            pip_path = os.path.join(venv_dir, "bin", "pip")
            try:
                subprocess.check_call([pip_path, "install", "--force-reinstall", "mediapipe"])
                import mediapipe as mp
                _ = mp.solutions.face_mesh
                print("\u2713 mediapipe.solutions available after reinstall")
            except Exception as reinstall_e:
                print(f"\u2717 Failed to reinstall mediapipe: {reinstall_e}")
                failed.append("mediapipe.solutions")
    return len(failed) == 0


def check_camera():
    """Check whether a camera is accessible."""
    print("\nChecking camera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("\u2713 Camera detected")
            cap.release()
            return True
        print("\u2717 Camera not accessible (may still work at runtime)")
        return False
    except Exception as e:
        print(f"\u2717 Camera check failed: {e}")
        return False


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 50)
    print("Expression Mimic \u2014 Setup")
    print("=" * 50)

    venv_dir = os.path.abspath("venv3.10")
    in_venv = (
        hasattr(sys, "base_prefix")
        and sys.prefix != sys.base_prefix
        and venv_dir in os.path.abspath(sys.prefix)
    )

    if not in_venv:
        if not os.path.isdir(venv_dir):
            if not create_and_activate_venv():
                print("\n\u2717 Could not create virtual environment.")
                sys.exit(1)
        venv_python = os.path.join(venv_dir, "bin", "python")
        if not os.path.isfile(venv_python):
            print(f"ERROR: Python not found in {venv_dir}.")
            sys.exit(1)
        print("\nRe-running setup inside the virtual environment...")
        subprocess.check_call([venv_python, os.path.abspath(__file__)])
        print("\nSetup completed.")
        print("\nTo run the application:")
        print("  source venv3.10/bin/activate")
        print("  python main.py")
        sys.exit(0)

    checks = [
        ("Python 3.10.x", check_python_version()),
        ("Dependencies", install_requirements()),
        ("Imports", verify_imports()),
        ("Camera", check_camera()),
    ]

    print("\n" + "=" * 50)
    print("Setup Summary:")
    print("=" * 50)
    for check_name, result in checks:
        status = "\u2713 PASS" if result else "\u2717 FAIL"
        print(f"  {check_name}: {status}")

    critical_passed = all(result for name, result in checks[:2])
    if critical_passed:
        print("\n\u2713 Setup complete! Ready to run.")
        print("\n  source venv3.10/bin/activate")
        print("  python main.py")
    else:
        print("\n\u2717 Setup incomplete. Fix the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
