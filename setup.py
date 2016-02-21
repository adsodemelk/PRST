# -*- coding: utf-8 -*-
import os
from setuptools import setup, Command

class CleanCommand(Command):
    """Custom clean command to tidy up project root."""
    # http://stackoverflow.com/questions/3779915/
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system("rm -vrf ./build ./dist ./*.pyc ./*.egg-info MANIFEST coverage.xml")

# Read file in this dir to string
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "PRST",
    packages = ["prst"],
    version = "0.0.1",
    description = "Python Reservoir Simulation Toolbox",
    long_description = read("README.md"),
    license = "GPLv3",
    author = "Andreas Røssland",
    author_email = "andreas.roessland@gmail.com",
    url = "https://github.com/roessland/PRST",
    download_url = "https://github.com/roessland/PRST/tarball/0.0.1",
    keywords = ["MRST", "reservoir", "simulation", "PDEs"],
    classifiers = [
        "Development Status :: 2 - Pre-Alpha",
    ],
    cmdclass = {
        "clean": CleanCommand,
    }
)
