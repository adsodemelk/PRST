# Python Reservoir Simulation Toolbox

[![Build Status](https://travis-ci.org/roessland/PyRST.png?branch=master)](https://travis-ci.org/roessland/PyRST)
[![Coverage Status](https://coveralls.io/repos/roessland/PyRST/badge.png?branch=master&service=github)](https://coveralls.io/github/roessland/PyRST?branch=master)

The Python Reservoir Simulation Toolbox is a Python version of the MATLAB
Reservoir Simulation Toolbox. The goal is to clone the functionality found in
MRST, and add tests.


## Installation

Currently there are no Python packages for this project. Download the whole
repository and copy the "PyRST/pyrst" folder into your project. Any scientific
Python3 distribution such as Anaconda will satisfy the requirements.


## Development:

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

## License

MRST is GPLv3-licensed, and since PyRST is a derivative work, it will also be
GPLv3-licensed. TODO: Add license.


