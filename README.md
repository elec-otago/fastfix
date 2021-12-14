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

## References

* Molteno, Timothy CA. "Estimating Position from Millisecond Samples of GPS Signals (the “FastFix” Algorithm)." Sensors 20.22 (2020): 6480.
* Molteno, Timothy CA, and Keith W. Payne. "FastFix Albatross Data: Snapshots of Raw GPS L-1 Data from Southern Royal Albatross." Data 6.4 (2021): 37.
