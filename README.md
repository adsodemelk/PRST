# Python Reservoir Simulation Toolbox

[![Build Status](https://travis-ci.org/roessland/PyRST.png?branch=master)](https://travis-ci.org/roessland/PyRST)
[![Coverage Status](https://coveralls.io/repos/roessland/PyRST/badge.png?branch=master&service=github)](https://coveralls.io/github/roessland/PyRST?branch=master)

The Python Reservoir Simulation Toolbox is a Python version of the MATLAB
Reservoir Simulation Toolbox. The goal is to clone the functionality found in
MRST, and add tests.


## MRST-2015a features available in PyRST

#### plotting

- [ ] plotGrid - partial support by exporting to ParaView
- [ ] plotCellData - partial support by exporting to ParaView

#### gridprocessing
- [x] tensorGrid
- [x] cartGrid
- [ ] computeGeometry
- [ ] makeLayeredGrid

#### utils

- [x] rlencode
- [x] rldecode


## Installation

Currently there are no Python or Conda packages for this project. Download the
whole repository and copy the "PyRST/pyrst" folder into your project. Any
scientific Python3 distribution such as Anaconda will satisfy the requirements.


## Linux installation (using Miniconda/Anaconda)

The installation procedure used for automatic deployments (Travis CS) is the following:

    # Download PyRST
    git clone https://github.com/roessland/PyRST.git

    # Download and update Miniconda for Python 2
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b -f $HOME/miniconda2
    export PATH=$HOME/miniconda2/bin:$PATH
    conda update conda

    # Install Conda packages necessary
    conda install python=2.7 numpy scipy six pytest
    conda install -c https://conda.anaconda.org/bokeh pytest-cov

    # Run tests to see that everything is working correctly
    cd PyRST
    py.test


## Linux installation (using Pip only)

To setup the development environment, and run tests, do the following:

    git clone https://github.com/roessland/PyRST.git
    cd PyRST
    virtualenv --python=python3 venv3
    source venv3/bin/activate
    pip install -r requirements.txt
    py.test

To resume working after the initial steps have been completed:

    cd PyRST
    source venv3/bin/activate
    py.test

SciPy needs to be compiled when installed using `pip`, and this can take up to 30 minutes.


## Windows installation

Should be simple, but has not been tested yet.


## Troubleshooting installation

* `py.test` executable does not exist: Remember to add the Miniconda
  environment to your $PATH, as done in the steps above.

* `py.test: error: unrecognized arguments: --cov-report=term-missing ...`: This
  means that `pytest-cov` has not been installed, or that you are running the
  wrong `py.test` executable. Run `type -a py.test` to see where `py.test` is
  located. It should be in the `miniconda2`-folder if you followed the
  instructions above, or in a virtual environment folder if using the method
  below.


## Code style

To remain similar to MRST, functions and variables should be `camelCased`. All
code should be tested.


## License

MRST is GPLv3-licensed, and since PyRST is a derivative work, it will also be
GPLv3-licensed. TODO: Add license.


