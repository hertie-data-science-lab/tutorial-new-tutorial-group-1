# Installation Guide

This guide explains how to set up the tutorial environment.

## Quick Start

### Using Makefile (Recommended)

```bash
# Create virtual environment and install dependencies
make create_venv

# Activate virtual environment
source .venv/bin/activate

# Install PyTorch (CPU-only)
make install-torch
```

### Manual Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package dependencies (without PyTorch)
pip install -e .

# Install PyTorch CPU-only version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Why PyTorch is Optional

PyTorch is not included as a direct dependency because:

1. **Size**: The CUDA version is ~7GB, which can cause disk space issues
2. **CPU-only alternative**: The CPU-only version is ~730MB and sufficient for this tutorial
3. **Flexibility**: Users can choose between CPU and GPU versions based on their needs

## PyTorch Installation Options

### CPU-only (Recommended for most users)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### CUDA (If you have an NVIDIA GPU)
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For more installation options, see the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Development Setup

### Pre-commit Hooks

This project uses pre-commit hooks for code quality. To set them up:

```bash
# Install development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run hooks manually on all files
pre-commit run --all-files
```

### Makefile Targets

- `make help` - Show available targets
- `make create_venv` - Create virtual environment and install dependencies
- `make install` - Install package with dev dependencies
- `make install-torch` - Install PyTorch CPU-only version
- `make lint` - Run pre-commit hooks on all files

## Devcontainer Setup

If you're using VS Code with devcontainers:

1. Open the repository in VS Code
2. Click "Reopen in Container" when prompted
3. The container will build and install dependencies automatically
4. **Important**: PyTorch is not installed by default. To install it, run:
   ```bash
   source .venv/bin/activate && make install-torch
   ```

## Troubleshooting

### Disk Space Issues

If you encounter disk space issues during installation:
- Make sure you're using the CPU-only version of PyTorch
- Clear pip cache: `pip cache purge`
- Consider using a cloud development environment with more storage

### Import Errors

If you get import errors for torch:
- Make sure PyTorch is installed: `pip list | grep torch`
- Verify you're using the correct virtual environment: `which python`
- Reinstall if necessary: `make install-torch`
