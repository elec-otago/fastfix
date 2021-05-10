
import os
import jsonrpclib
import datetime
import numpy as np

from tart.util.singleton import Singleton

from .gps_time import GpsTime
from .ephemeris import Ephemeris
#from tart.imaging import sp3_interpolator

class Sp4Ephemeris:
    def __init__(self, name, sv):
        self.name = name
        self.sv = sv

    def get_position(self, date):
        position, velocity = self.sv.propagate(date.year, date.month, date.day, date.hour, date.minute, date.second)
        vel = [velocity[0]*1000.0, velocity[1]*1000.0, velocity[2]*1000.0]
        pos = location.eci_to_ecef(date, position[0]*1000.0, position[1]*1000.0, position[2]*1000.0)
        return pos, vel

    def get_az_el(self, date, loc):
        pos, velocity = self.get_position(date)
        return loc.ecef_to_horizontal(pos[0], pos[1], pos[2] )

class Sp4Ephemerides:
    def __init__(self, local_path, jansky, name_list=None):
        self.jansky = jansky
        self.satellites = []
        f = open(local_path, "r")
        lines = f.readlines()
        for i, l in enumerate(lines):
            #print(i, l)
            if (i % 3 == 0):
                name = l.strip()
            
            if (i % 3 == 1):
                line1 = l.strip()
            
            if (i % 3 == 2):
                line2 = l.strip()
                sv = twoline2rv(line1, line2, wgs84)
                if name_list is None:
                    self.satellites.append(Sp4Ephemeris(name, sv))
                else:
                    # Check that name is in the list.
                    for n in name_list:
                        if n in name:
                            self.satellites.append(Sp4Ephemeris(name, sv))

    def get_positions(self, date):
        ret = []
        for sv in self.satellites:
            p, v = sv.get_position(date)
            ret.append({'name': sv.name, 'ecef': p, 'ecef_dot':v, 'jy':self.jansky})
        return ret

    #def get_az_el(self, date, lat, lon, alt):
        #ret = []
        #loc = location.Location(lat, lon, alt)
        ##print("Location {}".format(loc))
        
        #for sv in self.satellites:
            #_r,_el,_az = sv.get_az_el(date, loc)
            #el, az = np.round([_el.to_degrees(), _az.to_degrees()], decimals=6)
            #r = np.round(_r, decimals=1)
            #if (el > 0.0):
                #ret.append({'name': sv.name, 'r': r, 'el':el, 'az':az, 'jy':self.jansky})
        #return ret


@Singleton
class EphemeridesProxy(object):

    def __init__(self):
        
        gps_cache = norad_cache.GPSCache()
            
        self.server = jsonrpclib.Server('http://%s:8876/rpc/gps' % server_host)
        self.cache = {}
        self.sp3_cache = {}

    def get_date_hash(self, utc_date, sv):
        # Make a hash
        gpst = GpsTime.from_time(utc_date)
        cache_hash = "%02d%04d%02d%02d%02d-%4d" % (sv, utc_date.year, utc_date.month, utc_date.day, utc_date.hour, gpst.m_week)
        return cache_hash

    def get_sp3_hash(self, utc_date):
        # Make a hash
        gpst = GpsTime.from_time(utc_date)
        cache_hash = "%04d%02d%02d-%4d" % (utc_date.year, utc_date.month, utc_date.day, gpst.m_week)
        return cache_hash


    def get_ephemeris(self, utc_date, sv):
        h = self.get_date_hash(utc_date,sv)
        print("hash({}".format(h))
        try:
            eph = self.cache[h]
        except KeyError:
            eph_hash = self.server.get_ephemeris(utc_date.isoformat(), sv)
            eph = Ephemeris(eph_hash)
            self.cache[h] = eph
            print(("Cache miss %s, %d" % (utc_date, sv)))
        return eph

    #def get_sp3_interpolator(self, utc_date):
        #h = self.get_sp3_hash(utc_date)
        #try:
            #sp3 = self.sp3_cache[h]
        #except KeyError:
            #gpst = GpsTime.from_time(utc_date)
            #pts = self.server.get_interp_points(utc_date.isoformat())
            #sp3 = sp3_interpolator.Sp3Interpolator(gpst, pts)
            #self.sp3_cache[h] = sp3
            #print(("sp3 Cache miss %s" % (utc_date)))
        #return sp3

    def get_sv_position(self, utc_date, sv):
        gpst = GpsTime.from_time(utc_date)
        eph = self.get_ephemeris(utc_date, sv)
        pos = eph.get_sv_position(gpst)
        print("get_sv_position({}, {}, {}) -> {}".format(utc_date,  gpst, sv, pos))
        return np.array(pos)

    #def get_sv_position_sp3(self, utc_date, sv):
        #gpst = GpsTime.from_time(utc_date)
        #sp3 = self.get_sp3_interpolator(utc_date)
        #pos = sp3.get_sv_position(gpst,sv)
        #return np.array(pos)

    def get_sv_velocity(self, utc_date, sv):
        gpst = GpsTime.from_time(utc_date)
        eph = self.get_ephemeris(utc_date, sv)
        return eph.get_velocity(gpst.sow())


    def get_sv_positions(self, utc_date):
        gpst = GpsTime.from_time(utc_date)
        ret = []
        for sv in range(1,32):
            try:
                eph =    self.get_ephemeris(utc_date, sv)
                pos = eph.get_sv_position(gpst)
                ret.append([sv, pos])
            except jsonrpclib.ProtocolError:
                pass
        return ret

    def get_remote_position(self, utc_date, sv):
        return self.server.get_sv_position_sp3(utc_date.isoformat(), sv)

