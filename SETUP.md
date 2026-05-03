# SETUP.md

The project runs inside Docker. No host Python version matters.

---

## Prerequisites

### Arch Linux

```bash
sudo pacman -S docker docker-compose
sudo systemctl enable --now docker
sudo usermod -aG docker $USER   # log out and back in after this
```

**GPU (optional):**
```bash
sudo pacman -S nvidia-container-toolkit
sudo systemctl restart docker
```

### Windows

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).  
For GPU support, also install the [NVIDIA Container Toolkit for WSL2](https://docs.nvidia.com/cuda/wsl-user-guide/).

---

## Setup

```bash
git clone https://github.com/Gjorgjievski11/sonic-dqn-generalization.git
cd sonic-dqn-generalization
```

---

## Build

Only needed once, or when `pyproject.toml` / `Dockerfile` change.
Source code changes do **not** require a rebuild.

**CPU image:**
```bash
docker compose build train-cpu
# or
make build-cpu
```

**GPU image:**
```bash
docker compose build train-gpu
# or
make build-gpu
```

> **Note:** Building the GPU image does not require the NVIDIA Container Toolkit — only running it does.

---

## ROM Import

You need a legal copy of *Sonic The Hedgehog* for the Sega Genesis.
The easiest way is to purchase *Sega Mega Drive & Genesis Classics* on Steam.

Place your `.md` ROM file in `./roms/`, then import it:

```bash
make import-rom
```

The ROM is stored in a persistent Docker volume (`retro-data`) and only needs to be imported once.

Verify it worked:
```bash
docker compose run --rm train-cpu python -c "import stable_retro; env = stable_retro.make('SonicTheHedgehog-Genesis-v0'); print('ok')"
```

---

## Train

```bash
make train-cpu
```

You will see a human-readable log in the terminal:

```
Step   Episode   Ep Reward   Avg(10)   Epsilon   Steps/s
     500         3      142.0     138.5     0.938      45.2
    1000         7      198.0     155.3     0.876      47.1
```

Checkpoints are saved to `runs/checkpoints/` automatically.

**GPU (requires NVIDIA GPU + Container Toolkit):**
```bash
make train-gpu
```

---

## TensorBoard

```bash
make tensorboard
# open http://localhost:6006
```

---

## Evaluate

```bash
make evaluate
```

---

## Common Issues

| Problem                              | Fix                                                                                    |
|--------------------------------------|----------------------------------------------------------------------------------------|
| `permission denied` on docker socket | `sudo usermod -aG docker $USER`, re-login                                              |
| GPU container fails to start         | Install `nvidia-container-toolkit`, restart docker                                     |
| `ROM not found` inside container     | Run `make import-rom`                                                                  |
| `Imported 0 games`                   | Wrong ROM — needs the Steam version (SHA1: `69e102855d4389c3fd1a8f3dc7d193f8eee5fe5b`) |
| Build fails on `stable-retro`        | Ensure `libsdl2-dev` is in the Dockerfile (it is)                                      |
| `.pt` files appear corrupt in editor | They are binary files — do not open them in your IDE                                   |