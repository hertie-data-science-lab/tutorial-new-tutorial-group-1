# Few-Shot Learning for Rooftop Segmentation in Satellite Imagery

### GRAD-E1394 Deep Learning <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Hertie_School_of_Governance_logo.svg/1200px-Hertie_School_of_Governance_logo.svg.png" width="150px" align="right"/>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hertie-data-science-lab/tutorial-new-tutorial-group-1/blob/main/tutorial.ipynb)


## Author(s)

<div align="center">

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/elenaivadreyer">
        <img src="https://github.com/elenaivadreyer.png" width="90" style="border-radius:50%">
      </a><br>
      <sub><b>Elena Dreyer</b></sub><br>
      <sub>e.dreyer@students.hertie-school.org</sub>
    </td>
    <td align="center">
      <a href="https://github.com/giocopp">
        <img src="https://github.com/giocopp.png" width="90" style="border-radius:50%">
      </a><br>
      <sub><b>Giorgio Coppola</b></sub><br>
      <sub>g.coppola@students.hertie-school.org</sub>
    </td>
    <td align="center">
      <a href="https://github.com/NadineDaum">
        <img src="https://github.com/NadineDaum.png" width="90" style="border-radius:50%">
      </a><br>
      <sub><b>Nadine Daum</b></sub><br>
      <sub>n.daum@students.hertie-school.org</sub>
    </td>
    <td align="center">
      <a href="https://github.com/nicolasreichardt">
        <img src="https://github.com/nicolasreichardt.png" width="90" style="border-radius:50%">
      </a><br>
      <sub><b>Nicolas Reichardt</b></sub><br>
      <sub>n.reichardt@students.hertie-school.org</sub>
    </td>
  </tr>
</table>

</div>






## Tutorial Overview

This tutorial introduces few-shot learning techniques for semantic segmentation in satellite imagery using high-resolution images from Geneva, Switzerland. We will demonstrate how Prototypical Networks can learn meaningful rooftop representations from only a few labeled examples and generalize to new geographic areas with minimal annotation effort.

### Learning Outcomes

By the end of the tutorial, you will be able to:

- Understand the core concepts behind **Few-Shot Learning** and **Few-Shot Semantic Segmentation**
- Work with **satellite imagery**, geographic splits, and pixel-level segmentation masks
- Implement **Prototypical Networks** with episodic training for segmentation tasks
- Evaluate model performance using metrics such as **IoU** and interpret FSL model behavior
- Reflect on **policy-relevant applications** such as rooftop solar assessment and data-scarce mapping tasks


### Prerequisites

- Intermediate Python programming
- Familiarity with PyTorch
- Basics of Machine and Deep Learning
- Understanding of convolutional neural networks

## Dataset Description

🤗 [View on Hugging Face Hub](https://huggingface.co/datasets/raphaelattias/overfitteam-geneva-satellite-images) 🤗

The dataset being used for the demonstration of this tutorial consists of:
- **Satellite Images**: High-resolution RGB satellite images of Geneva, Switzerland
- **Segmentation Labels**: Binary masks indicating rooftop locations

## Quick Start

Either have a quick walk through the tutorial notebook or watch the video tutorial below to get started!

### 📓 Tutorial Notebook

**[View Tutorial Notebook (HTML)](tutorial.html)**


### 📹 Video Tutorial

<!-- VIDEO PLACEHOLDER: Replace the link below with your tutorial video -->
[![Tutorial Video](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)

*Click the image above to watch the tutorial video*


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
