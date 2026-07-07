# ML4HEP-V4 Differential Programming

This repository contains hands-on notebooks for introducing differentiable programming and applying it to a simple cosmological inference problem.

The material is intended for a tutorial-style session where students first learn the basic mechanics of automatic differentiation and then use those ideas in a compact Lyα forest inference exercise.

## Contents

- `Autograd_tutorial.ipynb`  
  Introductory PyTorch autograd tutorial covering computation graphs, `.backward()`, gradient accumulation, forward- and reverse-mode automatic differentiation, JVPs, VJPs, Jacobians, and simple differentiable modelling.

- `Cosmological_Inference_exercise.ipynb`  
  A compact inference exercise using CAMELS Lyα forest spectra. The notebook converts optical depths to transmitted flux, computes 1D flux power-spectrum summaries, trains a differentiable emulator, and performs simple likelihood-based inference for $\sigma_8$.


## Required setup

Create and activate a virtual environment in the repository directory, then install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib scikit-learn h5py torch jupyterlab ipykernel
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name ml4hep-dp --display-name "Python (ML4HEP-DP)"
```

Start JupyterLab from the repository directory:

```bash
jupyter lab
```

You may also open the notebooks directly in VS Code, provided the Python and Jupyter extensions are installed.

## CAMELS data

The cosmological inference notebook uses mock Lyman-alpha forest spectra from the CAMELS simulation suite. Students should download the required CAMELS Lyman-alpha spectra files before running the inference notebook.

Official CAMELS documentation:

- Data access: https://camels.readthedocs.io/en/latest/data_access.html
- Lyman-alpha spectra data product: https://camels.readthedocs.io/en/latest/Lya.html

For large downloads, CAMELS recommends using Globus. For small or individual files, the URL access option may be sufficient.

For this tutorial, place the selected spectra files in the following local structure:

```text
data/
└── Spectra_004/
    ├── 1P_2_n5/
    │   └── Lya-spectra.hdf5
    ├── 1P_2_n4/
    │   └── Lya-spectra.hdf5
    ├── ...
    └── 1P_2_5/
        └── Lya-spectra.hdf5
```


If your downloaded CAMELS directory has a different folder name, either rename the local directory or update `DATA_ROOT` and `SPECTRA_FOLDER` inside the notebook.

The first notebook introduces the differentiable-programming concepts needed for the second. The second notebook assumes that the CAMELS spectra files are already available locally.
