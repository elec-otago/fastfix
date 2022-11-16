#!\bin/python
# -*- coding: utf-8 -*-
import time
import scipy.io
import json

import numpy as np

from .acquisition import acquire, setup_fftw

# aptitude install python-scipy python-matplotlib

# import load_data

# from time import clock
from sys import stdout


from multiprocessing import Pool
from functools import partial

# class acquire_result:
# def __init__(self, sv, strength, phase, freq):
# self.sv = sv
# self.strength = strength
# self.phase = phase
# self.freq = freq


def acquire_all(rx, fs, fc0, searchBand, plot_corr=False):
    svMax = 32
    strength = np.zeros(svMax)
    phase = np.zeros(svMax)
    freq = np.zeros(svMax)

    print(("Tag Acquire fc0 = %f Hz" % (fc0)))
    t = time.time()

    (
        sampling_period,
        samples_per_ms,
        samples_per_chip,
        samples_per_chunk,
        numberOfFrqBins,
    ) = setup_fftw(fs, fc0, searchBand)

    threads = 1
    if threads == 1:
        p = Pool()

        resultList = []

        for i in range(0, svMax):
            sv = i + 1
            resultList.append(
                p.apply_async(acquire, (rx, fs, fc0, searchBand, sv, plot_corr))
            )

        p.close
        p.join

        for sv_thread in resultList:
            [sv, s, p, f] = sv_thread.get(timeout=200)
            strength[sv - 1] = s
            phase[sv - 1] = p
            freq[sv - 1] = f
    else:
        for sv in range(0, svMax):
            [p, strength[sv], phase[sv], freq[sv]] = acquire(
                rx, fs, fc0, searchBand, sv + 1, plot_corr
            )

    for i in range(0, svMax):
        sv = i + 1
        if strength[i] >= 0.0:
            print((" %02d %f %f %f" % (sv, phase[i], freq[i], strength[i])))
        # stdout.flush()
    print((" Acquire time %f seconds\n" % (time.time() - t)))

    ret = {}
    indices = strength > 7.0
    ret["x_max"] = strength[indices].tolist()
    ret["doppler"] = freq[indices].tolist()
    ret["codephase"] = phase[indices].tolist()
    ret["sv"] = (np.arange(0, svMax)[indices] + 1).tolist()

    return ret


import matplotlib.pyplot as plt


# def gps_spectrum(filename, fs, resolution):
# epochs = 2;

# rx = load_data.load_real_data(filename, fs, resolution, epochs);
# meanval = np.mean(rx)
# maxval = max(rx)
# minval = min(rx)
# rx = rx - np.mean(rx)

# n_points = len(rx)

# plt.close('all')
# title = "Captured GPS L1 spectrum"
## Plot a each channel
# power, freq = mlab.psd(rx,Fs=fs, NFFT=2048)
# plt.plot(freq/1e6, 10.0*np.log10(power))
##axarr[0].set_ylim([-80,-60])
# plt.grid()
# plt.xlabel("Freq (MHz)")
# plt.title(title)
# fname = 'spectrum_plot.pdf'
# plt.savefig(fname)
# plt.show()

# import pylab as p


# def tag_acquire(filename, fs, resolution, epochs, fc0, return_filename):
# searchBand = 5000.0 # Range of frequencies to search
## find the satellites
# print(('Calculating Correlations over %d epochs: %s' % (epochs,filename)))
# print(('Resolution (bits/data): ', resolution))
# print(('Center Frequency (Hz): ', fc0))
# print(('Samples Required : ', (fs * epochs / 1000)))
# rx = load_data.load_real_data(filename, fs, resolution, epochs);

## Do some local plotting
# print(("size of data %05d " % np.size(rx)))

# acquire(rx, fs, fc0, searchBand, return_filename)


def bit(n, i):
    # Retrieve the ith bit of a 4-bit number
    ret = (n >> (3 - i)) & 0x01
    if 1 == ret:
        return 1
    else:
        return -1


def write_nibble(n, a, is_complex):
    # Process a single nibble
    if is_complex:
        i0 = bit(n, 0)
        q0 = bit(n, 1)
        a.append(complex(i0, q0))

        i1 = bit(n, 2)
        q1 = bit(n, 3)
        a.append(complex(i1, q1))
    else:
        for i in range(0, 4):
            a.append(bit(n, i))


def write_byte(byte, a, iq):
    write_nibble((byte >> 4) & 0x0F, a, iq)
    write_nibble(byte & 0x0F, a, iq)


import struct


def decode(filename, iq):
    with open(filename, "rb") as f:
        bytes_read = f.read()

        s = struct.Struct("IIIIII")
        (
            rtc_counter_value,
            sample_count,
            sample_bytes,
            sample_ms,
            sampling_rate,
            checksum,
        ) = s.unpack(bytes_read[0:24])
        print(f"rtc_counter_value : {rtc_counter_value}")
        print(f"sample_count : {sample_count}")
        print(f"sample_bytes : {sample_bytes}")
        print(f"sample_ms : {sample_ms}")
        print(f"sampling_rate : {sampling_rate}")
        print(f"checksum : {checksum}")

        data_bytes = bytes_read[24:-1]
        data_words = []

        b = struct.Struct("BB")
        ret = []
        for i in range(0, sample_bytes - 2, 2):
            lsb, msb = b.unpack(data_bytes[i : i + 2])
            data_words.append(lsb + msb * 256)
            write_byte(msb, ret, iq)
            write_byte(lsb, ret, iq)

    data = np.array(ret)

    test_check = calculate_checksum(sample_bytes, data_words)

    if test_check != checksum:
        raise RuntimeError(f"Checksums do not match {checksum} : {test_check}")

    return rtc_counter_value, sample_ms, sampling_rate, checksum, data


def calculate_checksum(sample_bytes, data_buffer):
    ret = 0
    word_count = sample_bytes // 2
    for i in range(0, word_count, 2):
        ret += data_buffer[i]
        print(data_buffer[i])
    return ret


# To test on known data use
# python bit_error_rate.py ../../quickfix/python/30dBm.bin 8.184e6 1
import unittest
import sys
from subprocess import Popen, PIPE


class TestAcquisition(unittest.TestCase):
    def setUp(self):
        self.raw_file = "./test_data/FIX00033.BIN"
        self.filename = "tag_data_file.out"

        self.sample_ms, self.sampling_rate, self.checksum, self.dat = decode(
            self.raw_file
        )
        cmd = "../decode/tag_decode %s >    %s" % (self.raw_file, self.filename)
        process = Popen(cmd, stderr=PIPE, shell=True)
        stdout, stderr = process.communicate()

        self.fs = 16.368e6
        self.fc0 = 4.092e6

        print(stderr)
        # values = Hash.new
        # Open3.popen3(cmd) { |stdin, stdout, stderr|
        # while l = stderr.gets
        # puts l
        # val= l.split(' => ')
        # if    val[1] != nil
        # values[val[0].strip] = val[1].strip.to_i
        # end
        # end
        # }

    def test_decode(self):
        rx = load_data.load_real_data(self.filename, fs=self.fs, resolution=1, epochs=4)
        sample_ms, sampling_rate, checksum, dat = decode(self.raw_file)
        for i in range(0, len(rx)):
            self.assertEqual(rx[i], dat[i])
        # self.assertEqual(len(rx), len(dat))

    def test_spectrum(self):
        gps_spectrum(filename=self.filename, fs=self.fs, resolution=1)

    def test_zsimple(self):
        tag_acquire(
            self.filename,
            fs=self.fs,
            resolution=1,
            epochs=4,
            fc0=self.fc0,
            return_filename="results.out",
        )
