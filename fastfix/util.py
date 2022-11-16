import math
import numpy as np


class Util:
    PI = 3.1415926535897932
    PI2 = 6.2831853071795864
    C = 2.99793e8
    L1 = 1575.42e6
    WAVELENGTH = C / L1


def check_t(t):
    half_week = 302400.0
    tt = t

    if t > half_week:
        tt = t - 2 * half_week
    if t < -half_week:
        tt = t + 2 * half_week

    return tt


def idiv(x, y):
    """ Integer Division """
    return int(math.floor(x / y))


def rem(x, y):
    """ Remainder """
    return x - y * idiv(x, y)


#! \brief remainder = numerator - quotient * denominator
#
def mod(x, y):
    ret = x - y * idiv(x, y)
    return ret


def mod_int(x, y):
    ret = x % y
    if x < 0:
        ret = y - (-x) % y
    return ret


def phase_delta2(x, y):

    mu = np.exp(2j*Util.PI*y)
    z = np.exp(2j*Util.PI*x)

    return np.abs(z - mu) / Util.PI2


def phase_delta(x):
    ret = mod_int(x, 1)
    if ret > 0.5:
        ret = 1.0 - ret
    return ret


def gaussian_llh(x, mu, sigma):

    llh = -np.log(np.sqrt(2.0 * np.pi) * sigma)
    dx = x - mu
    llh += -(dx*dx / (2.0 * sigma ** 2.0))
    return llh


def rad2deg(x):
    return np.degrees(x)


def rem2pi(x):
    return rem(x, Util.PI2)


def safePath(url):
    safePath_chars = set(map(lambda x: ord(x), '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+-_.'))
    return ''.join(map(lambda ch: chr(ch) if ch in safePath_chars else '%%%02x' % ch, url.encode('utf-8')))

