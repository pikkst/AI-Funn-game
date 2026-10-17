# Local development and contributor workflow

**Current state:** docs-only repository. All setup and commands below describe the target M1 implementation, not already available scripts.

## Target runtime

Windows 11 Pro, Python 3.11 (or 3.12 when dependencies support it), 32 GB system memory, NVIDIA RTX 2060 SUPER 8 GB. Python environment should use pinned dependencies and a lock strategy selected in the first implementation task.

## Preflight (PowerShell)

```powershell
nvidia-smi
py -3.11 --version
```

Create a virtual environment when the Python project scaffold exists:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch using the **current official Windows/PyTorch CUDA selector** at https://pytorch.org/get-started/locally/ ; do not assume a DX12 device automatically provides a working CUDA runtime. After installation:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

CPU is supported for M1. An 8 GB VRAM limit precludes assumptions about running large language models simultaneously, but small MLP policies are feasible.

## Planned quality gates

- `pytest` for simulation invariants, deterministic traces, Gymnasium API and policy smoke tests.
- Python lint/format (Ruff), type checks where practical, dependency audit and reproducibility smoke.
- No full 500k-step job required on every PR; use a short training smoke and isolated deterministic tests in CI.
- Full benchmark/training runs executed locally and attached as separate evidence.

## Development policy

Branch changes, run targeted tests, verify relevant quality gates, open PR, review, then merge only with explicit approval. Do not commit experiments as successful without actual run artifacts.

## Ignore and retention rules

Ignore `.venv/`, `__pycache__/`, `*.pyc`, `checkpoints/`, `runs/`, `*.pt`, `*.pth`, `*.zip` model artifacts, `*.db`, local secrets and dataset caches as appropriate. Preserve small human-readable experiment manifests and documentation where useful. Record retention limits in trainer config.

## Future UI

M1 replay can be CLI or static snapshot output. Live WebSocket visualization, HTTP endpoints and PixiJS belong to later milestones. Avoid coupling backend API contracts into the core simulation until they are needed.

## Troubleshooting guide (once implemented)

- CUDA unavailable: check `nvidia-smi`, PyTorch build and Python environment, then test CPU.
- Non-reproducible traces: inspect RNG ownership, dictionary iteration order, floating-point path, versioned config and replay state.
- Reward high without eating: run always-rest baseline and inspect metabolism/infinite-rest loopholes.
- Train good/eval poor: investigate seed leakage, overfitting, observation normalization, incorrect termination/truncation and evaluation determinism.
- CI resource issues: reduce smoke steps, never weaken behavioral acceptance tests.
