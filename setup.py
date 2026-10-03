#
# Copyright Tim Molteno 2020-21 tim@elec.ac.nz
#

from setuptools import setup

with open("README.md", 'r', encoding="utf-8") as f:
    readme = f.read()

setup(
    name="fastfix",
    version="0.1.0b4",
    description="FastFix Positioning",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="http://github.com/elec-otago/projects/fastfix",
    author="Tim Molteno",
    author_email="tim@elec.ac.nz",
    license="GPLv3",
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "pyfftw",
        "unlzw3",
        "pymc>=5.0",
        "arviz",
        "pytensor",
    ],
    extras_require={
        "tests": ["pytest", "ephem", "astropy", "tart"],
    },
    packages=["fastfix"],
    scripts=["bin/fastfix", "bin/acquire"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Intended Audience :: Science/Research",
    ],
)
