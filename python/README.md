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
## Quality gates

Run the following from the repository root in PowerShell:

`powershell
# Install lint, test and type-check dependencies
python -m pip install -e './python[dev,test]'

# Lint
python -m ruff check ./python/src ./python/tests

# Format check
python -m ruff format --check ./python/src ./python/tests

# Tests
python -m pytest ./python/tests

# Type check
python -m mypy ./python/src
`

CI runs the same gates on Windows via .github/workflows/ci.yml (lint, test, type-check) on push to main and pull requests. Training artifacts (checkpoints/, uns/, *.pt, *.pth, *.zip) are excluded from CI and are not uploaded as artifacts.