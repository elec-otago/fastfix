# -*- coding: utf-8 -*-
#The data are sent as conventional sign/magnitude with all four levels:
#
#01 -3
#00 -1
#10 +1
#11 +3
def sign_mag(sgn,mag):
    return sgn*(mag + 2)
    
import numpy as npy

def load_file(filename, n_bytes):
  fd = open(filename, 'rb')
  read_data = npy.fromfile(file=fd, dtype=npy.int8, count=n_bytes)
  fd.close()
  print(("points read ", read_data.size))
  return read_data
  
def load_iq_2bit_data(filename, numberOfPoints):
  print(("load_iq_2bit_data %s" % filename))
  read_data = load_file(filename, numberOfPoints*4)
  
  shape = (numberOfPoints,4)
  read_data = read_data.reshape(shape)
  
  data_i1=read_data[:,0].astype('double');
  data_i0=read_data[:,1].astype('double');
  data_q1=read_data[:,2].astype('double');
  data_q0=read_data[:,3].astype('double');
  x = sign_mag(data_i1, data_i0) + sign_mag(data_q1, data_q0) * 1.0j
  return x


def load_iq_1bit_data(filename, numberOfPoints):
  print(("load_iq_1bit_data %s" % filename))
  read_data = load_file(filename, numberOfPoints*2)
  
  shape = (numberOfPoints,2)
  read_data = read_data.reshape(shape)
  
  i1=read_data[:,0].astype('double');
  q1=read_data[:,1].astype('double');
  x = i1 + q1 * 1.0j
  return x

def load_iq_data(filename, fs, resolution, epochs):
  n=int(npy.ceil(fs/1000));        # data points in an epoch
  numberOfPoints = epochs*n;
  print(("Number of samples ",  numberOfPoints))
  print(("Bit Resolution ",  resolution))
  
  if (1 == resolution):
    x = load_iq_1bit_data(filename, numberOfPoints)
  else:
    x = load_iq_2bit_data(filename, numberOfPoints)
  return x

def load_real_data(filename, fs, resolution, epochs):
  n=int(npy.ceil(fs/1000));        # data points in an epoch
  numberOfPoints = epochs*n;
  print(("Number of samples ",  numberOfPoints))
  print(("Bit Resolution ",  resolution))
  
  read_data = load_file(filename, numberOfPoints)
  i1 = read_data.astype('double');
  #x = i1 - npy.mean(i1)
  return i1
