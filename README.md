# Kaggle: [Cell Instance Segmentation](https://www.kaggle.com/c/sartorius-cell-instance-segmentation)

[![CI complete testing](https://github.com/Borda/kaggle_cell-inst-segm/actions/workflows/ci_testing.yml/badge.svg?branch=main&event=push)](https://github.com/Borda/kaggle_cell-inst-segm/actions/workflows/ci_testing.yml)
[![Code formatting](https://github.com/Borda/kaggle_cell-inst-segm/actions/workflows/code-format.yml/badge.svg?branch=main&event=push)](https://github.com/Borda/kaggle_cell-inst-segm/actions/workflows/code-format.yml)
[![codecov](https://codecov.io/gh/Borda/kaggle_cell-inst-segm/branch/main/graph/badge.svg)](https://codecov.io/gh/Borda/kaggle_cell-inst-segm)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Borda/kaggle_cell-inst-segm/main.svg)](https://results.pre-commit.ci/latest/github/Borda/kaggle_cell-inst-segm/main)

The goal of this challenge is to detect cells in microscope images.

![Sample brain visual](./assets/cells-annot.png)

with simple view on how many cels have been annotated per image:

![Sample brain visual](./assets/hist_cells-per-image.png)

## Experimentation

### install this tooling

A simple way how to use this basic functions:

```bash
! pip install https://github.com/Borda/kaggle_cell-inst-segm/archive/refs/heads/main.zip
```

### run notebooks in Kaggle

- [🦠Cell Instance Segm: 🔍 interactive data browsing](https://www.kaggle.com/jirkaborovec/cell-instance-segm-interactive-data-browsing)
- [🦠Cell Instance Segm. ~ Lightning⚡Flash](https://www.kaggle.com/jirkaborovec/cell-instance-segm-lightning-flash)

### local notebooks

- TBD
-

### some results

Training progress with EfficientNet3D with training  for 10 epochs > over 96% validation accuracy:

![Training process](./assets/metrics.png)
