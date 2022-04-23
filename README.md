# Kaggle: Image Segmentation

[![CI complete testing](https://github.com/Borda/kaggle_image-segm/actions/workflows/ci_testing.yml/badge.svg?branch=main&event=push)](https://github.com/Borda/kaggle_image-segm/actions/workflows/ci_testing.yml)
[![Code formatting](https://github.com/Borda/kaggle_image-segm/actions/workflows/code-format.yml/badge.svg?branch=main&event=push)](https://github.com/Borda/kaggle_image-segm/actions/workflows/code-format.yml)
[![codecov](https://codecov.io/gh/Borda/kaggle_image-segm/branch/main/graph/badge.svg)](https://codecov.io/gh/Borda/kaggle_image-segm)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Borda/kaggle_image-segm/main.svg)](https://results.pre-commit.ci/latest/github/Borda/kaggle_image-segm/main)

### install this tooling

A simple way how to use this basic functions:

```bash
! pip install https://github.com/Borda/kaggle_image-segm/archive/refs/heads/main.zip
```

## Kaggle: [Cell Instance Segmentation](https://www.kaggle.com/c/sartorius-cell-instance-segmentation)

The goal of this challenge is to detect cells in microscope images.

![Sample cells visual](./assets/cells-annot.png)

### run notebooks in Kaggle

- [🦠Cell Instance Segm: 🔍 interactive data browsing](https://www.kaggle.com/jirkaborovec/cell-instance-segm-interactive-data-browsing)
- [🦠Cell Instance Segm. ~ Lightning⚡Flash](https://www.kaggle.com/jirkaborovec/cell-instance-segm-lightning-flash)

## Kaggle: [Tract Image Segmentation](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation)

The goal of this challenge is to segment organs in medical scans to improve cancer treatment

![Sample organs visual](./assets/organs-annot.png)

### run notebooks in Kaggle

- [Tract🩻Segm: EDA 🔎 & 3D🗄️data browser](https://www.kaggle.com/code/jirkaborovec/tract-segm-eda-3d-data-browser)

## Experimentation

### local notebooks

- TBD
-

### some results

Training progress with EfficientNet3D with training  for 10 epochs > over 96% validation accuracy:

![Training process](./assets/metrics.png)
