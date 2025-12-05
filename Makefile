SHELL := /bin/bash

.PHONY: create_venv install install-torch lint help

help:
	@echo "Available targets:"
	@echo "  create_venv    - Create virtual environment and install dependencies (without torch)"
	@echo "  install        - Install package with dev dependencies (without torch)"
	@echo "  install-torch  - Install PyTorch CPU-only version"
	@echo "  lint           - Run pre-commit hooks on all files"

create_venv:
	@echo "Creating virtual environment..."
	python3 -m venv .venv
	@echo "Installing dependencies (without torch)..."
	.venv/bin/python3 -m pip install --upgrade pip
	.venv/bin/pip install -e .[dev]
	@echo ""
	@echo "Virtual environment created successfully!"
	@echo "To install PyTorch (CPU-only), run: make install-torch"
	@echo "Or manually install with:"
	@echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"

install:
	python3 -m pip install --upgrade pip
	pip install -e .[dev]
	@echo "Dependencies installed (without torch)."
	@echo "To install PyTorch (CPU-only), run: make install-torch"

install-torch:
	@echo "Installing PyTorch CPU-only version..."
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	@echo "PyTorch installed successfully!"

lint:
	pre-commit run --all-files
