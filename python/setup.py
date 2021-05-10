#
# Copyright Tim Molteno 2020 tim@elec.ac.nz
#

from setuptools import setup, find_packages

import setuptools.command.test

with open('README.md') as f:
    readme = f.read()

setup(name='fastfix',
    version='0.1.0b2',
    description='FastFix Positioning',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='http://github.com/elec-otago/projects/fastfix',
    author='Tim Molteno',
    test_suite='nose.collector',
    tests_require=['nose', 'ephem'],
    author_email='tim@elec.ac.nz',
    license='GPLv3',
    install_requires=['numpy', 'matplotlib', 'scipy', 'pyfftw', 'unlzw'],
    packages=['fastfix'],
    scripts=['bin/fastfix', 'bin/acquire'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering",
        "Topic :: Communications :: Ham Radio",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "Intended Audience :: Science/Research"])
