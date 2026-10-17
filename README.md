# AI-Funn-game — Digital Life Evolution Lab

**Status:** design / pre-implementation (2026-10-17). No working simulator, trained model, UI, or CI is claimed.

A local-first artificial-life research playground. Train neural agents **from randomly initialized weights** to survive in a procedurally generated environment; later investigate competition, cooperation, signaling, reproduction, and emergent social behavior.

The initial experiment is deliberately modest: **one organism learns to acquire food and survive better than a random policy in held-out worlds**. Human-like language, open-ended technological discovery, and civilization formation are research aspirations, not MVP deliverables.

## Documentation

- [Project vision and scope](docs/VISION.md)
- [Architecture and interfaces](docs/ARCHITECTURE.md)
- [World and environment specification](docs/SIMULATION.md)
- [Training and evaluation protocol](docs/TRAINING.md)
- [Development environment](docs/DEVELOPMENT.md)
- [Roadmap and acceptance criteria](TASKS.md)
- [Rules for coding agents](AGENTS.md)

## Planned stack

- Python 3.11 or 3.12, NumPy, Gymnasium, PyTorch, Stable-Baselines3 PPO, pytest.
- Subsequent stages: FastAPI + WebSockets, React/TypeScript + PixiJS, SQLite/TensorBoard.
- Windows 11 local-first; reference development hardware: i7-8700K, RTX 2060 SUPER 8 GB VRAM, 32 GB RAM.
- CUDA is optional for the first small policy; environment simulation remains CPU-side.

## Project principles

1. The **simulation is authoritative**: policies choose actions, never modify world state.
2. Train from scratch means **random neural-network weights**, not a pretrained LLM.
3. A policy's observation is local; avoid global-map information leaks.
4. Keep simulation rules pure or isolated from ML/UI infrastructure.
5. Distinguish training performance from held-out evaluation.
6. Treat deterministic replays and reproducible experiment artifacts as first-class.
7. Keep training headless; rendering must not be on the critical simulation path.
8. Never claim emergent intelligence based on reward alone; measure behavior.

## Planned repository layout (not yet implemented)

```text
apps/observatory/                # future React/PixiJS UI
packages/protocol/schemas/       # future UI/service contracts
python/pyproject.toml
python/src/ai_funn/
  simulation/
  environments/survival_v1.py
  agents/
  training/
  experiments/
  api/                           # future FastAPI interface
python/tests/
docs/
checkpoints/                     # ignored
runs/                            # ignored
```

## Starting point

Work through [TASKS.md](TASKS.md) milestone **M1**, beginning with the Python environment and deterministic simulation core. The commands in [DEVELOPMENT.md](docs/DEVELOPMENT.md) are **planned** until their supporting files and CLIs exist.

## Non-goals for M1

No multi-agent civilization, language emergence, genetic reproduction, autonomous technology invention, cloud inference, distributed training, 3D rendering, or web deployment.

## Safety and resource controls

Local-only by default; no secrets in repo or logs. Bound training duration, disk usage, episode length, and checkpoint retention. Store configuration and code revision with each experiment; never commit model weights or large run data.
