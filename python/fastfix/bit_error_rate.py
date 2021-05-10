#!\bin/python
# -*- coding: utf-8 -*-
import numpy as npy

import acquisition

# aptitude install python-scipy python-matplotlib
from scipy.io.numpyio import fwrite, fread

import load_data

from time import clock
from sys import stdout

def acquire(rx, fs, fc0, searchBand):	
	svMax = 2;
	strength = npy.zeros(svMax)
	phase = npy.zeros(svMax)
	freq = npy.zeros(svMax)
	
	t = clock();
	for i in range(1,svMax):
		PRN = i+1
		[strength[i], phase[i], freq[i]] = acquisition.acquire(rx, fs, fc0,searchBand, PRN);
		if (strength[i] > 0.0):
			print((' %02d %f %f %f' % (PRN, phase[i], freq[i], strength[i])))
		else:
			print((' .'), end=' ')
		stdout.flush()
	
	print(" Acquire time %f seconds\n" % (clock() - t));
	
	fp = open('gps.dat','w')
	fp.write("GPS time %s\n" % "Date not set yet")
	fp.write("END OF HEADER\n")
	
	for i in range(svMax):
		imax = npy.argmax(strength)
		if (strength[imax] > 1):
			fp.write("%2.2d, %2.2d, %8.5e, %+8.5e, %1.9f\n" % (i+1, imax+1, strength[imax], freq[imax], phase[imax]))
		strength[imax] = 0
	
	fp.close()
	
import pylab as p



def ber(filename, fs, resolution):
	fc0 = 000.0;	# center freq without Doppler
	searchBand = 52000.0 # Range of frequencies to search
	# find the satellites
	print('Calculating Correlations over 2 epochs: ', filename)
	print('Resolution (bits/data): ', resolution)
	rx = load_data.load_iq_data(filename, fs, resolution, 2);
	
	# Do some local plotting
	p.figure()
	p.plot(npy.real(rx[1:100]));
	p.plot(npy.imag(rx[1:100]));
	p.title("Filename=%s" % filename)
	p.savefig('IQ%s.pdf' % filename)

	p.figure()
	p.plot(npy.real(rx),npy.imag(rx));
	p.title("Filename=%s" % filename)
	p.savefig('Eye%s.pdf' % filename)

	print("size of data %05d " % (npy.size(rx)))
	
	acquire(rx, fs, fc0, searchBand)
 
 
# To test on known data use
# python bit_error_rate.py ../../quickfix/python/30dBm.bin 8.184e6 1

import sys
 
def main(*args):
	ber(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]))
	return 0 # exit errorlessly
 
if __name__ == '__main__':
	sys.exit(main(*sys.argv))
