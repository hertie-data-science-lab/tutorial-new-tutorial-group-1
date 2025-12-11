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

- Alsentzer, E., Li, M. M., Kobren, S. N., Noori, A., Undiagnosed Diseases Network, Kohane, I. S., & Zitnik, M. (2025). Few shot learning for phenotype-driven diagnosis of patients with rare genetic diseases. *npj Digital Medicine, 8*(1), 380. [https://doi.org/10.1038/s41746-025-01749-1](https://doi.org/10.1038/s41746-025-01749-1)

- Castello, R., Walch, A., Attias, R., Cadei, R., Jiang, S., & Scartezzini, J.-L. (2021). Quantification of the suitable rooftop area for solar panel installation from overhead imagery using convolutional neural networks. *Journal of Physics: Conference Series, 2042*(1), 012002. [https://doi.org/10.1088/1742-6596/2042/1/012002](https://doi.org/10.1088/1742-6596/2042/1/012002)

- Chen, Y., Wei, C., Wang, D., Ji, C., & Li, B. (2022). Semi-supervised contrastive learning for few-shot segmentation of remote sensing images. *Remote Sensing, 14*(17), 4254. [https://doi.org/10.3390/rs14174254](https://doi.org/10.3390/rs14174254)

- Ding, H., Zhang, H., & Jiang, X. (2022). Self-regularized prototypical network for few-shot semantic segmentation. *Pattern Recognition, 132*, 109018. [https://doi.org/10.1016/j.patcog.2022.109018](https://doi.org/10.1016/j.patcog.2022.109018)

- Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. In *International Conference on Machine Learning* (pp. 1126–1135). PMLR. [https://doi.org/10.48550/arXiv.1703.03400](https://doi.org/10.48550/arXiv.1703.03400)

- Ge, Z., Fan, X., Zhang, J., & Jin, S. (2025). SegPPD-FS: Segmenting plant pests and diseases in the wild using few-shot learning. *Plant Phenomics*, 100121. [https://doi.org/10.1016/j.plaphe.2025.100121](https://doi.org/10.1016/j.plaphe.2025.100121)

- Hu, Y., Liu, C., Li, Z., Xu, J., Han, Z., & Guo, J. (2022). Few-shot building footprint shape classification with relation network. *ISPRS International Journal of Geo-Information, 11*(5), 311. [https://doi.org/10.3390/ijgi11050311](https://doi.org/10.3390/ijgi11050311)

- Jadon, S. (2021, February). COVID-19 detection from scarce chest x-ray image data using few-shot deep learning approach. In *Medical Imaging 2021: Imaging Informatics for Healthcare, Research, and Applications* (Vol. 11601, pp. 161–170). SPIE. [https://doi.org/10.1117/12.2581496](https://doi.org/10.1117/12.2581496)

- Lee, G. Y., Dam, T., Ferdaus, M. M., Poenar, D. P., & Duong, V. (2025). Enhancing Few-Shot Classification of Benchmark and Disaster Imagery with ATTBHFA-Net. *arXiv preprint arXiv:2510.18326.* [https://doi.org/10.48550/arXiv.2510.18326](https://doi.org/10.48550/arXiv.2510.18326)

- Li, X., He, Z., Zhang, L., Guo, S., Hu, B., & Guo, K. (2025). CDCNet: Cross-domain few-shot learning with adaptive representation enhancement. *Pattern Recognition, 162*, 111382. [https://doi.org/10.1016/j.patcog.2025.111382](https://doi.org/10.1016/j.patcog.2025.111382)

- Puthumanaillam, G., & Verma, U. (2023). Texture based prototypical network for few-shot semantic segmentation of forest cover: Generalizing for different geographical regions. *Neurocomputing, 538*, 126201. [https://doi.org/10.1016/j.neucom.2023.03.062](https://doi.org/10.1016/j.neucom.2023.03.062)

- Sung, F., Yang, Y., Zhang, L., Xiang, T., Torr, P. H., & Hospedales, T. M. (2018). Learning to compare: Relation network for few-shot learning. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 1199–1208). [https://doi.org/10.1109/CVPR.2018.00131](https://doi.org/10.1109/CVPR.2018.00131)


## License

This project is licensed under the MIT License - see the LICENSE file for details.
