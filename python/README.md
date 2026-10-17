# AI-Funn-game Python Package

Bootstrap instructions for a fresh Windows 11 installation.

## Prerequisites

- Python 3.11 or 3.12 (verify with `python --version`)
- pip 23.0 or newer (verify with `python -m pip --version`)

## Setup

1. Open PowerShell in this repository root.
2. Create a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Upgrade pip and install the package in editable mode with the test
   extra:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -e ./python[test]
   ```

## Running the smoke tests

```powershell
python -m pytest ./python/tests
```

A passing run confirms that the package imports, the declared
subpackages are importable, and the Python version floor is met.