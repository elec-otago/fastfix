develop:
	pip3 install -e .
# sudo python3 setup.py develop

test:
	pytest-3 -k TestEphem
	
.PHONY: roof_test fastfix

mcmc:
	fastfix --json-file test_data/roof_test_AF25.json --mcmc --plot --n 1 --output-file 'roof_test_AF25_mcmc.json'

fastfix:
	fastfix --json-file test_data/roof_test_AF25.json --output-file 'roof_test_AF25_fastfix.json'

acquire:
	acquire --binfile test_data/FIX00033.BIN --fc0 4.092e6 --outfile test_data/FIX00033.out

iq:
	acquire --binfile test_data/IQ_DATA.BIN --fc0 32000 --iq --outfile iq_results.json --spectrum --corr-plot

dir:
	for f in `find /freenas/tag/albatross_data/ -name 'AF*'`; do \
		echo $f ; \
	done

roof_test:
	mkdir -p albatross_results/stationary_data
	sh roof_tests.sh

albatross:
	mkdir -p albatross_results/albatross_data
	sh albatross.sh
	
all:
	sh testbench_py.sh

install:
	sudo apt install python3-numpy python3-matplotlib python3-scipy python3-pyfftw

test_upload:
	python3 setup.py sdist
	twine upload --repository testpypi dist/*

upload:
	python3 setup.py sdist
	twine upload --repository pypi dist/*
