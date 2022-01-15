# ranking

## Environment Setup
### Overview
<!-- The underlying project environment composes of following componenets: -->
The project has been tested on the following environment specification:
1. Ubuntu 18.04.5 LTS (Other x86_64 based Linux distributions should also be fine, such as Fedora 32)
2. Nvidia Graphic Card (NVIDIA Tesla T4 with driver version 450.142.00) and CPU (Intel Core i7-10700 CPU @ 2.90GHz)
3. Python 3.7 (and Python 3.6.12)
4. CUDA 11.0 (and CUDA 9.2)
5. Pytorch 1.10.1 (built against CUDA 11.0) and Pytorch 1.8.0 (build against CUDA 9.2)
6. Other libraries and python packages (See below)

### Installation method 1 (.yml files)
You should handle (1),(2) yourself. For (3), (4), (5) and (6), we provide a list of steps to install them.

<!-- We place those python packages that can be easily installed with one-line command in the requirement file for `pip` (`requirements_pip.txt`). For all other python packages, which are not so well maintained by [PyPI](https://pypi.org/), and all C/C++ libraries, we place in the conda requirement file (`requirements_conda.txt`). Therefore, you need to run both conda and pip to get necessary dependencies. -->

We provide two examples of envionmental setup, one with CUDA 11.0 and GPU, the other with CPU.

Following steps assume you've done with (1) and (2).
1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Both Miniconda and Anaconda are OK.

2. Create an environment and install python packages (GPU):
```
conda env create -f environment_GPU.yml
```

3. Create an environment and install python packages (CPU):
```
conda env create -f environment_CPU.yml
```


### Installation method 2 (manual installation)
The codebase is implemented in Python 3.6.12. package versions used for development are below.
```
networkx           2.6.3
tqdm               4.62.3
numpy              1.20.3
pandas             1.3.4
texttable          1.6.4
latextable         0.2.1
scipy              1.7.1
argparse           1.1.0
scikit-learn       1.0.1
stellargraph       1.2.1 (for link direction prediction: conda install -c stellargraph stellargraph)
torch              1.10.1
torch-scatter      2.0.9
pyg                2.0.3 (follow https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
sparse             0.13.0
```

### Execution checks
When installation is done, you could check you enviroment via:
```
cd execution
bash setup_test.sh
```