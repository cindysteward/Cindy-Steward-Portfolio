"""
Setup script for the Facial Motor and Speech Behavior Analysis Pipeline.

Creates a Python 3.11 virtual environment, installs all dependencies from
requirements.txt, verifies that MediaPipe and OpenCV are functional, creates
the required data directory structure, and prints usage instructions.

Usage:
    python3 setup.py
"""

import sys
import subprocess
import platform
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
VENV_NAME = "venv"
VENV_PATH = PROJECT_ROOT / VENV_NAME


def print_header(message: str) -> None:
    """Print a prominent section header."""
    print("\n" + "=" * 60)
    print(message)
    print("=" * 60)


def print_step(step: int, message: str) -> None:
    """Print a numbered step label."""
    print(f"\n[Step {step}] {message}")


def run_command(cmd: list, description: str, check: bool = True) -> subprocess.CompletedProcess:
    """Execute a shell command, print its first few output lines, and return the result."""
    print(f"  Running: {' '.join(str(c) for c in cmd)}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=check, encoding="utf-8",
        )
        if result.stdout:
            for line in result.stdout.strip().split("\n")[:5]:
                print(f"    {line}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"  Error: {e.stderr}")
        if check:
            raise
        return e


def get_python_executable() -> list:
    """Return a list of command args to invoke Python 3.11.

    Returns a list so callers can unpack it into subprocess args, e.g.:
        subprocess.run([*get_python_executable(), "-m", "venv", ...])

    On Windows, tries the Python Launcher (py -3.11) first, then plain
    'python' or 'python3' if they resolve to 3.11.
    On macOS/Linux, tries 'python3.11' and 'python3.11.9' first.
    """
    import shutil

    for candidate in ("python3.11.9", "python3.11"):
        py = shutil.which(candidate)
        if py:
            return [py]

    py_launcher = shutil.which("py")
    if py_launcher:
        try:
            result = subprocess.run(
                [py_launcher, "-3.11", "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if result.returncode == 0 and "3.11" in (result.stdout + result.stderr):
                return [py_launcher, "-3.11"]
        except Exception:
            pass

    for candidate in ("python3", "python"):
        py = shutil.which(candidate)
        if not py:
            continue
        try:
            result = subprocess.run(
                [py, "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split()
                if len(parts) >= 2 and parts[0] == "3" and parts[1] == "11":
                    return [py]
        except Exception:
            pass

    print(
        "ERROR: Python 3.11 is required for this project.\n"
        "  macOS/Linux:  install via pyenv or your package manager, then ensure\n"
        "                'python3.11' is on your PATH.\n"
        "  Windows:      download Python 3.11 from https://www.python.org/downloads/\n"
        "                and install the Python Launcher (py.exe) so 'py -3.11' works,\n"
        "                or ensure 'python' on PATH resolves to Python 3.11."
    )
    sys.exit(1)


def get_venv_python() -> Path:
    """Return the path to the Python executable inside the virtual environment."""
    if platform.system() == "Windows":
        return VENV_PATH / "Scripts" / "python.exe"
    return VENV_PATH / "bin" / "python"


def get_venv_pip() -> Path:
    """Return the path to pip inside the virtual environment."""
    if platform.system() == "Windows":
        return VENV_PATH / "Scripts" / "pip.exe"
    return VENV_PATH / "bin" / "pip"


def create_virtual_environment() -> bool:
    """Create a Python 3.11 virtual environment, optionally recreating an existing one."""
    print_step(1, "Creating virtual environment")

    if VENV_PATH.exists():
        print(f"  Virtual environment already exists at {VENV_PATH}")
        response = input("  Recreate? (y/N): ").strip().lower()
        if response != "y":
            print("  Using existing virtual environment")
            return True

        import shutil

        print("  Removing existing virtual environment...")
        shutil.rmtree(VENV_PATH)

    print(f"  Creating virtual environment at {VENV_PATH}")

    try:
        py_args = get_python_executable()
        subprocess.run(
            [*py_args, "-m", "venv", str(VENV_PATH)], check=True
        )
        print("  Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Failed to create virtual environment: {e}")
        return False


def install_dependencies() -> bool:
    """Install all packages from requirements.txt into the virtual environment."""
    print_step(2, "Installing dependencies")

    pip_path = get_venv_pip()

    print("  Upgrading pip...")
    run_command([str(pip_path), "install", "--upgrade", "pip"], "Upgrade pip")

    requirements_file = PROJECT_ROOT / "requirements.txt"
    print("  Installing packages from requirements.txt (this may take a few minutes)...")
    result = run_command(
        [str(pip_path), "install", "-r", str(requirements_file)],
        "Install requirements",
        check=True,
    )
    return result.returncode == 0


def verify_installation() -> bool:
    """Import each required package inside the venv and verify MediaPipe Face Mesh initialises."""
    print_step(3, "Verifying installation")

    python_path = get_venv_python()

    packages_to_test = [
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("yaml", "PyYAML"),
        ("PIL", "Pillow"),
        ("nibabel", "nibabel"),
        ("nilearn", "nilearn"),
    ]

    all_ok = True

    for module, name in packages_to_test:
        result = subprocess.run(
            [
                str(python_path),
                "-c",
                f"import {module}; print({module}.__version__ if hasattr({module}, '__version__') else 'OK')",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"  ✓ {name}: {version}")
        else:
            print(f"  ✗ {name}: NOT INSTALLED")
            all_ok = False

    print("\n  Testing MediaPipe FaceLandmarker (Tasks API)...")
    test_code = (
        "from mediapipe.tasks import python as mp_python; "
        "from mediapipe.tasks.python import vision; "
        "assert hasattr(vision, 'FaceLandmarker'), 'FaceLandmarker class not found'; "
        "assert hasattr(vision, 'FaceLandmarkerOptions'), 'FaceLandmarkerOptions not found'; "
        "print('MediaPipe FaceLandmarker (Tasks API): Available')"
    )

    result = subprocess.run(
        [str(python_path), "-c", test_code],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    if result.returncode == 0:
        print(f"  ✓ {result.stdout.strip()}")
    else:
        print(f"  ✗ MediaPipe FaceLandmarker Tasks API not available: {result.stderr.strip()}")
        all_ok = False

    return all_ok


def create_directories() -> None:
    """Create the required data and log directories if they do not exist."""
    print_step(4, "Creating project directories")

    directories = [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "results",
        PROJECT_ROOT / "logs",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory.relative_to(PROJECT_ROOT)}")


def print_usage_instructions() -> None:
    """Display post-setup usage instructions for the pipeline."""
    print_step(5, "Setup complete!")

    if platform.system() == "Windows":
        activate_cmd = (
            f"{VENV_NAME}\\Scripts\\activate.bat  "
            f"(or: {VENV_NAME}\\Scripts\\Activate.ps1  for PowerShell)"
        )
    else:
        activate_cmd = f"source {VENV_NAME}/bin/activate"

    print(f"""
  To use the pipeline:

  1. Activate the virtual environment:
     cd {PROJECT_ROOT}
     {activate_cmd}

  2. Run the pipeline (live capture):
     python src/run_pipeline.py --mode pilot --subject P001 --session baseline --input live

  3. Run the pipeline (video file):
     python src/run_pipeline.py --mode pilot --subject P001 --session test --input /path/to/video.mp4

  4. Run with reference comparison:
     python src/run_pipeline.py --mode patient --subject PAT001 --session intra_op --input live --reference P001_baseline_20260101_120000

  5. Launch the Flask web interface:
     python launch_ui.py
     Then open http://localhost:5050 in your browser

  Live Capture Controls:
    'n' - Mark neutral baseline segment
    'm' - Mark measurement segment
    'r' - End current segment
    'q' or 'ESC' - Stop capture and process

  Output locations:
    - Raw data: data/raw/
    - Processed data: data/processed/
    - Results & visualizations: data/results/
    - Logs: logs/
""")


def main() -> int:
    """Run all setup steps: venv creation, dependency install, verification, and directory setup."""
    print_header("FACIAL MOTOR AND SPEECH BEHAVIOR ANALYSIS PIPELINE - SETUP")

    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Project root: {PROJECT_ROOT}")

    if sys.version_info < (3, 8):
        print("\nError: Python 3.8 or higher is required")
        return 1

    if not create_virtual_environment():
        print("\nSetup failed: Could not create virtual environment")
        return 1

    if not install_dependencies():
        print("\nSetup failed: Could not install dependencies")
        return 1

    verification_ok = verify_installation()

    create_directories()

    if verification_ok:
        print_usage_instructions()
        return 0

    print("\n  Warning: Some packages may not be properly installed.")
    print("  The pipeline may still work, but some features might be limited.")
    print_usage_instructions()
    return 0


if __name__ == "__main__":
    sys.exit(main())
