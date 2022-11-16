#
# Helper class for dealing with angles (and to avoid getting tangled between degrees and radians.
#
# Copyright (c) Tim Molteno 2013. tim@elec.ac.nz
#
import numpy as np

TWO_PI = np.pi * 2


def wrap_2pi(x):
    r = np.fmod(x, TWO_PI)
    if r < 0.0:
        r += TWO_PI
    return r


def wrap_360(x):
    r = np.fmod(x, 360.0)
    if r < 0.0:
        r += 360.0
    return r


def wrap_180(x):
    r = np.fmod(x, 360.0)
    if r > 180.0:
        r -= 360.0
    return r


def d2r(x):
    return x * TWO_PI / 360.0

# from functools import total_ordering
# @total_ordering


class Angle(object):
    def __init__(self, deg=0, minute=0, sec=0):
        dec_deg = deg + float(minute) / 60.0 + float(sec) / 3600.0
        self.rad = d2r(dec_deg)

    def __repr__(self):
        return str(self.to_degrees())

    def to_elevation(self):
        deg = self.to_degrees()
        if deg > 90.0:
            raise "Elevation out of RANGE %f" % deg
        if deg < -90.0:
            raise "Elevation out of RANGE %f" % deg

        return from_dms(deg)

    def to_declination(self):
        return self.to_elevation()

    def to_ra(self):
        return from_rad(wrap_2pi(self.rad))

    def to_rad(self):
        return self.rad

    def to_degrees(self):
        return (self.rad * 360 / TWO_PI)

    def to_hours(self):
        return wrap_2pi(self.rad) * (24.0 / TWO_PI)

    def to_dms(self):
        dd = self.to_degrees()
        mnt, sec = divmod(dd * 3600, 60)  # 60 seconds in a
        deg, mnt = divmod(mnt, 60)
        return deg, mnt, sec

    def to_hms(self):
        dd = self.to_degrees()

        hour, rem = divmod(dd, 15)  # 15 Degrees in an hour
        mnt, sec = divmod(rem * 3600 / 15, 60)  # 60 Minutes in an hour
        return hour, mnt, sec

    def sin(self):
        return np.sin(self.rad)

    def cos(self):
        return np.cos(self.rad)

    def tan(self):
        return np.tan(self.rad)

    def __abs__(self):
        return from_rad(abs(self.rad))

    def __add__(self, x):
        return from_rad(self.rad + x.rad)

    def __radd__(self, x):
        return self.__add__(x)

    def __sub__(self, x):
        return from_rad(self.rad - x.rad)

    def __rsub__(self, x):
        return self.__sub__(x)

    def __eq__(self, x):
        return wrap_2pi(self.rad) == wrap_2pi(x.rad)

    def __lt__(self, x):
        return self.rad < x.rad  # TODO should we wrap both?

    def __neg__(self):
        return from_rad(-self.rad)


def from_hours(hour, minute=0.0, sec=0.0):
    hr = hour % 24.0
    return Angle(deg=hr * 15, minute=minute * 15, sec=sec * 15)


def from_dms(deg=0.0, minute=0.0, sec=0.0):
    return Angle(deg, minute, sec)


def from_rad(rad):
    a = Angle(0.0, 0.0, 0.0)
    a.rad = rad
    return a


def atan(x):
    return from_rad(np.arctan(x))


def atan2(x, y):
    return from_rad(np.arctan2(x, y))


def asin(x):
    return from_rad(np.arcsin(x))


def acos(x):
    return from_rad(np.arccos(x))


def add_rad(a, x):
    return from_rad(a.rad + x)
