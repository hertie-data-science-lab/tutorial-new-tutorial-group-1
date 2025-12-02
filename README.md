# Few-Shot Learning for Rooftop Detection in Satellite Imagery
### GRAD-E1394 Deep Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hertie-data-science-lab/tutorial-new-tutorial-group-1/blob/elena-setup/notebooks/tutorial_few_shot_learning.ipynb)


**Author(s):** Elena Dreyer, Giorgio Coppola, Nadine Daum, Nicolas Reichardt

## Tutorial Overview

This tutorial demonstrates few-shot learning techniques for semantic segmentation of satellite imagery. The dataset contains high-resolution satellite images of Geneva, Switzerland, with corresponding segmentation labels for rooftop detection.

📓 **[View Tutorial Notebook (HTML)](docs/tutorial_few_shot_learning.html)**

### Learning Outcomes
- Understanding few-shot learning concepts for image segmentation
- Working with satellite imagery and segmentation masks
- Implementing and evaluating few-shot learning models for rooftop detection

## Video Tutorial

<!-- VIDEO PLACEHOLDER: Replace the link below with your tutorial video -->
[![Tutorial Video](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)

*Click the image above to watch the tutorial video*

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/hertie-data-science-lab/tutorial-new-tutorial-group-1.git
cd tutorial-group-1

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package (without PyTorch)
pip install -e .

# Install PyTorch (CPU-only version recommended to save disk space)
# CPU-only: ~730MB vs CUDA version: ~7GB
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Or use the Makefile
make install
make install-torch
```

**Note:** PyTorch is not installed by default due to its large size. The CPU-only version is recommended for most use cases and saves significant disk space.

### Running the Tutorial

Open the main tutorial notebook:
```bash
jupyter notebook notebooks/tutorial_few_shot_learning.ipynb
```

## Project Structure

```
├── notebooks/
│   └── tutorial_few_shot_learning.ipynb     # Main tutorial notebook
├── src/few_shot_utils/
│   ├── __init__.py                          # Package initialization
│   ├── data.py                              # Data loading utilities
│   ├── models.py                            # Model architectures
│   ├── train.py                             # Training functions
│   └── evaluate.py                          # Evaluation metrics
└── docs/                                    # Documentation
```

## Dataset Description

The dataset consists of:
- **Satellite Images**: High-resolution RGB satellite images of Geneva, Switzerland
- **Segmentation Labels**: Binary masks indicating rooftop locations
- **Resolution**: Images at various resolutions suitable for few-shot learning

## Development

### Linting and Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, including Jupyter notebooks.

```bash
# Run linting with automatic fixes
ruff check . --fix
ruff format .

# Run pre-commit hooks
pre-commit run --all-files
```

## References

-
-

## License

This project is licensed under the MIT License - see the LICENSE file for details.
