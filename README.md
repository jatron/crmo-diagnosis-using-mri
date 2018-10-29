# crmo-diagnosis-using-mri

## Setup

1. Install [Miniconda](https://conda.io/docs/user-guide/install/index.html#regular-installation)
  - Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
  - Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)
2. cd into src, run `conda env create -f environment.yml`
  - This creates a Conda environment called `cs229-project`
3. Run `source activate cs229-project`
  - This activates the `cs229-project` environment
  - Do this each time you want to write/test your code
