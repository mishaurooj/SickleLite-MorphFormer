# SickleLite-MorphFormer

> **A Lightweight Morphology-Aware Hybrid Network for Efficient Sickle Cell Smear Classification**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Conda](https://img.shields.io/badge/environment-Anaconda-green.svg)](https://www.anaconda.com/)

---

## Overview

SickleLite-MorphFormer is a compact hybrid deep learning model for binary classification of sickle cell disease from peripheral blood smear images.

The framework combines:

- **FastStem** for lightweight local feature extraction
- **Directional Sickle Response Module (DSRM)** for elongated and crescent-like RBC morphology
- **Lite Context Transformer Block (LCTB)** for efficient global context modeling
- **Gated Morph-Context Fusion (GMCF)** for adaptive feature fusion
- **Dual-head learning** for classification and morphology consistency

---

## Dataset Layout

Expected dataset structure:

```text
D:\sickle\dataset\
├── Positive\
│   ├── Labelled\
│   └── Unlabelled\
└── Negative\
```

- `Positive/Labelled`: positive images with annotations or marked cells
- `Positive/Unlabelled`: positive images without annotations
- `Negative`: normal or non-sickle images

Dataset source: Kaggle Sickle Cell Disease Dataset.

---

## Anaconda Environment Setup

### 1. Install Anaconda or Miniconda

Install one of these first:

- **Anaconda** for a full distribution
- **Miniconda** for a smaller setup

After installation, open:

- **Anaconda Prompt** on Windows, or
- a terminal with `conda` available on Linux/macOS

---

### 2. Create the project environment

```bash
conda create -n sicklelite python=3.10 -y
```

Activate it:

```bash
conda activate sicklelite
```

---

### 3. Install PyTorch

For CUDA 12.1:

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

If you want CPU only:

```bash
conda install pytorch torchvision cpuonly -c pytorch -y
```

If your GPU driver is older, switch to a compatible CUDA version supported by your system.

---

### 4. Install the remaining dependencies

```bash
conda install numpy pandas scikit-learn matplotlib seaborn scipy pillow tqdm jupyter ipykernel -y
pip install albumentations opencv-python timm
```

Optional profiling package:

```bash
pip install thop
```

---

### 5. Register the environment for Jupyter

```bash
python -m ipykernel install --user --name sicklelite --display-name "Python (sicklelite)"
```

This lets you run the notebook in the same environment.

---

## Optional: `environment.yml`

Save this as `environment.yml` in the project root:

```yaml
name: sicklelite
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch
  - torchvision
  - pytorch-cuda=12.1
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - scipy
  - pillow
  - tqdm
  - jupyter
  - ipykernel
  - pip
  - pip:
      - albumentations
      - opencv-python
      - timm
      - thop
```

Create the environment from file:

```bash
conda env create -f environment.yml
conda activate sicklelite
```

---

## Verify the Environment

Run:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected output:

- PyTorch version prints successfully
- `True` if CUDA is working

You can also confirm the GPU name:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

---

## Recommended Project Structure

```text
SickleLite-MorphFormer/
├── README.md
├── environment.yml
├── sickle_morph.py
├── outputs/
└── dataset/
```

If your main script is the uploaded implementation, rename or keep it as:

```text
sickle_morph.py
```

---

## Running the Project

### Train

```bash
python sickle_morph.py --data_root "D:/sickle/dataset" --mode train
```

### Run ablations

```bash
python sickle_morph.py --data_root "D:/sickle/dataset" --mode ablation
```

### Evaluate a saved checkpoint

```bash
python sickle_morph.py --data_root "D:/sickle/dataset" --mode eval --checkpoint outputs/best_model.pth
```

---

## Suggested VS Code / Jupyter Setup

If you use VS Code:

1. Open the project folder.
2. Select the interpreter from the `sicklelite` conda environment.
3. If you use notebooks, choose the kernel **Python (sicklelite)**.

If you use Jupyter Notebook:

```bash
jupyter notebook
```

Then select the `sicklelite` kernel.

---

## Common Issues

### `ModuleNotFoundError: No module named 'timm'`

```bash
pip install timm
```

### OpenCV import errors

```bash
pip install opencv-python
```

### CUDA not available

Check:

- NVIDIA driver is installed
- PyTorch CUDA build matches your driver
- you installed `pytorch-cuda` and not CPU-only PyTorch

### DataLoader worker issues on Windows

If needed, reduce workers in the config from `8` to `0` or `2`.

---

## Recommended Versions

Tested setup target:

- Python 3.10
- PyTorch 2.x
- torchvision 0.15+
- CUDA 12.1 build
- timm 0.9+
- albumentations 1.3+

---

## Reproducibility Notes

For stable runs:

- keep the same random seed
- keep the same train/val/test split
- log package versions using:

```bash
conda env export > exported_environment.yml
pip freeze > requirements_frozen.txt
```

---

## Output Files

Training generates outputs such as:

```text
outputs/
├── logs/
├── figures/
├── *_best.pth
├── ablation_A1_backbone.csv
├── ablation_A2_dsrm.csv
├── ablation_A3_context.csv
├── ablation_A4_fusion.csv
├── ablation_A5_learning.csv
└── ablation_A6_loss.csv
```

---

## Citation

```bibtex
@article{sickleLiteMorphFormer2026,
  title   = {SickleLite-MorphFormer: A Lightweight Morphology-Aware Hybrid Network for Efficient Sickle Cell Smear Classification},
  author  = {Your Name},
  year    = {2026}
}
```

