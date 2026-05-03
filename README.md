# Sonic DQN Generalization

Measuring how DQN agents trained on specific Sonic the Hedgehog zones
generalize to unseen zones.

---

## Quick Start

```bash
# 1. Install dependencies (auto-detects NVIDIA GPU on Linux)
make install

# 2. Import your Sonic ROM (see "ROM Setup" below)
make import-rom

# 3. Train
make train

# 4. Evaluate
make evaluate
```

For detailed, OS-specific setup see **[SETUP.md](SETUP.md)**.

---

## ROM Setup

Sonic ROMs cannot be distributed with this project.

**How to get the ROM legally:**
Purchase *Sega Mega Drive & Genesis Classics* on Steam. The ROM file
(`Sonic The Hedgehog (World).md`) is included unencrypted.

Default Steam paths:
- **Linux:** `~/.steam/steam/steamapps/common/Sega Classics/uncompressed ROMs/`
- **Windows:** `C:\Program Files (x86)\Steam\steamapps\common\Sega Classics\uncompressed ROMs\`

Copy the `.md` file to `./roms/`, then run `make import-rom`.

---

## Project Structure

```
sonic-dqn-generalization/
├── configs/
│   └── dqn_sonic.yaml       # All hyperparameters — edit here, not in code
├── scripts/
│   ├── import_rom.sh        # ROM importer (Linux/Mac)
│   └── setup_windows.ps1    # Windows setup helper
├── src/sonic_dqn/
│   ├── agents/              # DQN model + replay buffer
│   ├── envs/                # Sonic environment wrapper
│   ├── training/            # Training loop
│   ├── evaluation/          # Evaluation loop
│   └── utils/               # Seeding, config loading
├── roms/                    # Place your .md ROM files here (git-ignored)
├── runs/                    # Training outputs (git-ignored)
├── train.py
├── evaluate.py
├── Makefile
├── pyproject.toml
└── SETUP.md
```

---

## Configuration

All hyperparameters live in `configs/dqn_sonic.yaml`.
Override any value at the command line:

```bash
uv run python train.py --config configs/dqn_sonic.yaml \
    --override training.learning_rate=0.0005 \
    --override training.total_timesteps=500000
```

---

## Reproducibility

- Python, NumPy, and PyTorch are seeded from `experiment.seed` in the config.
- `torch.use_deterministic_algorithms(True)` is enabled by default.
- All dependencies are pinned in `uv.lock`.

**Known limitations:**
- CUDA convolution ops can be non-deterministic across different GPU hardware
  even with seeding. CPU runs are fully deterministic.
- gym-retro's emulation is deterministic given the same seed and action
  sequence, but may differ across OS versions or libretro builds.

---

## Known Issues

| Issue | Fix |
|---|---|
| `FileNotFoundError: ROM not found` | Run `make import-rom` |
| `RuntimeError: no deterministic algorithm` | Set `deterministic_cuda: false` in config |
| `ModuleNotFoundError: retro` | Run `make install` first |
| GPU PyTorch installed on CPU machine | Run `make install-cpu` to reinstall |
