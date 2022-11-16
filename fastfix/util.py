import math
import numpy as np

class Util(object):
    PI = 3.1415926535897932
    PI2 = 6.2831853071795864
    C = 2.99793e8
    L1 = 1575.42e6
    WAVELENGTH = C / L1

    @classmethod
    def check_t(self, t):
        half_week = 302400.0
        tt = t

        if t > half_week:
            tt = t - 2 * half_week
        if t < -half_week:
            tt = t + 2 * half_week

        return tt

    @classmethod
    def idiv(self, x, y):
        """ Integer Division """
        return int(math.floor(x / y))

    @classmethod
    def rem(self, x, y):
        """ Remainder """
        return x - y * Util.idiv(x, y)

    #! \brief remainder = numerator - quotient * denominator
    #
    @classmethod
    def mod(self, x, y):
        ret = x - y * Util.idiv(x, y)
        return ret

    @classmethod
    def mod_int(self, x, y):
        ret = x % y
        if x < 0:
            ret = y - (-x) % y
        return ret

    #@classmethod
    #def phase_delta(self, x, y):

        #mu = np.exp(2j*Util.PI*y)
        #z = np.exp(2j*Util.PI*x)
        
        #return np.abs(z - mu) / Util.PI2

    @classmethod
    def phase_delta(self, x):
        ret = self.mod_int(x, 1)
        if ret > 0.5:
            ret = 1.0 - ret
        return ret

    @classmethod
    def gaussian_llh(self, x, mu, sigma):

        llh = -np.log(np.sqrt(2.0 * np.pi) * sigma)
        dx = x - mu
        llh += -(dx*dx / (2.0 * sigma ** 2.0))
        return llh

    @classmethod
    def rad2deg(self, x):
        return x * 180.0 / Util.PI

    @classmethod
    def rem2pi(self, x):
        return Util.rem(x, Util.PI2)
    
    
    @classmethod
    def safePath(url):
        return ''.join(map(lambda ch: chr(ch) if ch in safePath.chars else '%%%02x' % ch, url.encode('utf-8')))
    safePath.chars = set(map(lambda x: ord(x), '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+-_.'))
