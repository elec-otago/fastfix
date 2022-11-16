# FastFix GPS positioning algorithm

Author: Tim Molteno tim@elec.ac.nz

This repository contains code to implement the fastfix GPS positioning algorithm. 
FastFix is a new approach to GPS positioning is described in which the post-processing of ultra-short
sequences of captured GPS signal data can produce an estimate of receiver location. The algorithm,
needs only 2–4 ms of stored L1-band data sampled at ∼16 MHz. The algorithm uses a
least-squares optimization to estimate receiver position and GPS time from measurements of the relative
codephase, and Doppler-shift of GNSS satellite signals. A practical application of this algorithm is
demonstrated in a small, lightweight, low-power tracking tag that periodically wakes-up, records and
stores 4 ms of GPS L1-band signal and returns to a low-power state—reducing power requirements by
a factor of ∼10,000 compared to typical GPS devices. Stationary device testing shows a median error
of 27.7 m with a small patch antenna. Results from deployment of this tag on adult Royal Albatross
show excellent performance, demonstrating lightweight, solar-powered, long-term tracking of these
remarkable birds. This work was performed on the GPS system; however, the algorithm is applicable to
other GNSS systems.

The software in this respository is written in the Python language (see the python directory) and
can be installed using the python package manager

    sudo pip3 install fastfix
    
Data from wildlife tags is described in the second reference below.


## Usage ##

The process for use is a two-step process. Step 1 is acquisition of the satellite signals. There is a command-line tool for that.

### Acquisition ###

    usage: acquire [-h] [--tag-dir TAG_DIR] [--binfile BINFILE] --outfile OUTFILE [--decodedfile DECODEDFILE] --fc0 FC0 [--resolution RESOLUTION] [--epochs EPOCHS] [--iq] [--spectrum] [--corr-plot] [--start-date START_DATE]

    Acquire SV signals from Max2769B GPS data.

    optional arguments:
        -h, --help            show this help message and exit
        --tag-dir TAG_DIR     The FastFix tag directory.
        --binfile BINFILE     The binary data file.
        --outfile OUTFILE     The output data file.
        --decodedfile DECODEDFILE
                                The decoded file.
        --fc0 FC0             Center Frequency.
        --resolution RESOLUTION
                                Number of bits per sample.
        --epochs EPOCHS       Number of code epochs.
        --iq                  Use I/Q complex baseband.
        --spectrum            Plot the spectrum of the data.
        --corr-plot           Plot the correlations of each SV.
        --start-date START_DATE
                                Date and time for the start of the clock.
Example:
	acquire --binfile test_data/FIX00033.BIN --fc0 4.092e6 --outfile test_data/FIX00033.json

### FastFix algorith ###

This processes a json file that contains one or more acquisition outputs.

    usage: fastfix [-h] --json-file JSON_FILE --output-file OUTPUT_FILE [--mcmc] [--plot] [--n N] [--clock-offset-std CLOCK_OFFSET_STD]

    FastFix algorithm.

    optional arguments:
    -h, --help            show this help message and exit
    --json-file JSON_FILE
                            The JSON acquisition file.
    --output-file OUTPUT_FILE
                            The JSON output file.
    --mcmc                Use an MCMC.
    --plot                Make plots of the posteriors.
    --n N                 How many fixes to process.
    --clock-offset-std CLOCK_OFFSET_STD
                            Estimated error in the local clock (seconds).

Example:

	fastfix --json-file test_data/FIX00033.json --output-file test_data.json

## Test Data ##

A full dataset including test data is available at doi:10.5281/zenodo.4266994.

## Changelog

* 0.1.0b3 Initial Release
* 0.2.0b1 Add VonMises distributions for the MCMC.

## References

The FastFix algorithm is described in Molteno, Timothy CA. "Estimating Position from Millisecond Samples of GPS Signals (the “FastFix” Algorithm)." Sensors 20.22 (2020): 6480.

Citing this work:
    @article{molteno2020estimating,
        title={Estimating Position from Millisecond Samples of GPS Signals (the “FastFix” Algorithm)},
        author={Molteno, Timothy CA},
        journal={Sensors (Basel, Switzerland)},
        volume={20},
        number={22},
        year={2020},
        publisher={Multidisciplinary Digital Publishing Institute (MDPI)}
    }

* Molteno, Timothy CA, and Keith W. Payne. "FastFix Albatross Data: Snapshots of Raw GPS L-1 Data from Southern Royal Albatross." Data 6.4 (2021): 37.
