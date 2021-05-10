import unittest
import datetime
import logging
import urllib.request
import json

import numpy as np

from fastfix import GPSFileCache, GpsTime

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler()) # Add a null handler so logs can go somewhere
logger.setLevel(logging.INFO)

class TestEphem(unittest.TestCase):

    def setUp(self):
        self.prox = GPSFileCache()
        self.dt = datetime.datetime.fromisoformat('2020-03-08T20:51:15')
        self.ephs = self.prox.get_ephemerides(self.dt)
            
    def test_get_pos(self):
        gps_t = GpsTime.from_time(self.dt)
        e = self.ephs.get_ephemeris(prn=4, gps_t=gps_t)
        
    def test_against_TART_public(self):
        # Test agains the object position server using the current datetime
        url = "https://tart.elec.ac.nz/catalog/catalog?lat=-45.85&lon=170.54&date=2020-03-08T20:51:15"
        url = "https://tart.elec.ac.nz/catalog/position?date=2020-03-08T20:51:15"
        gps_t = GpsTime.from_time(self.dt)
        
        dat = urllib.request.urlopen(url)
        ops_json = json.loads(dat.read())
        
        for sv in ops_json:
            n = sv['name']
            if (n.find('GPS') != -1):
                x = n.find("PRN ")
                prn = int(n[x+4:x+6])
                e = self.ephs.get_ephemeris(prn=prn, gps_t=gps_t)
                sat_pos = e.get_location(gps_t.sow())
                
                sat_pos2 = e.get_location_new(gps_t.sow())
                logger.info(f"Old: {sat_pos}. New {sat_pos2}. External: {sv['ecef']}")
                for a,b in zip(sat_pos, sat_pos2):
                    self.assertAlmostEqual(a,b)
                
                self.assertAlmostEqual(sv['ecef'][0], sat_pos[0])
                
            
        self.assertTrue(False)
        
