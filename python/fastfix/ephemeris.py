# Copyright (C) Tim Molteno 2008-2019. All rights reserved

import math
import logging
import string

import numpy as np


from .util import Util
from .gps_time import GpsTime

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler()) # Add a null handler so logs can go somewhere
logger.setLevel(logging.INFO)


t = str.maketrans('D', 'E')
def substr(s, x, n):
    ''' Get a substring, ready for conversion to a number. Change the D -> E from the old-fashioned
        FORTRAN number format.
    '''
    return s[int(x):int(x+n)].strip().translate(t)

def str_to_int(s,x,n):
    return int(substr(s,x,n))

def str_to_float(s,x,n):
    try:
        return float(substr(s,x,n))
    except:
        return 0

''' Parse the Broadcast ephemerides file.

COMMENT             | Comment line(s)                          |     A60    |*
 +--------------------+------------------------------------------+------------+
*|ION ALPHA           | Ionosphere parameters A0-A3 of almanac   |  2X,4D12.4 |*
 |                    | (page 18 of subframe 4)                  |            |
 +--------------------+------------------------------------------+------------+
*|ION BETA            | Ionosphere parameters B0-B3 of almanac   |  2X,4D12.4 |*
 +--------------------+------------------------------------------+------------+
*|DELTA-UTC: A0,A1,T,W| Almanac parameters to compute time in UTC| 3X,2D19.12,|*
 |                    | (page 18 of subframe 4)                  |     2I9    |
 |                    | A0,A1: terms of polynomial               |            |
 |                    | T    : reference time for UTC data       |      *)    |
 |                    | W    : UTC reference week number.        |            |
 |                    |        Continuous number, not mod(1024)! |            |
 +--------------------+------------------------------------------+------------+
*|LEAP SECONDS        | Delta time due to leap seconds           |     I6     |*
 +--------------------+------------------------------------------+------------+
 |END OF HEADER       | Last record in the header section.       |    60X     |


For each ephemeris, the files is stored in RINEX format. Here is the data

 +----------------------------------------------------------------------------+
 |                                  TABLE A4                                  |
 |           GPS NAVIGATION MESSAGE FILE - DATA RECORD DESCRIPTION            |
 +--------------------+------------------------------------------+------------+
 |    OBS. RECORD     | DESCRIPTION                              |   FORMAT   |
 +--------------------+------------------------------------------+------------+
 |PRN / EPOCH / SV CLK| - Satellite PRN number                   |     I2,    |
 |                    | - Epoch: Toc - Time of Clock             |            |
 |                    |          year (2 digits, padded with 0   |            |
 |                    |                if necessary)             |  1X,I2.2,  |
 |                    |          month                           |   1X,I2,   |
 |                    |          day                             |   1X,I2,   |
 |                    |          hour                            |   1X,I2,   |
 |                    |          minute                          |   1X,I2,   |
 |                    |          second                          |    F5.1,   |
 |                    | - SV clock bias       (seconds)          |  3D19.12   |
 |                    | - SV clock drift      (sec/sec)          |            |
 |                    | - SV clock drift rate (sec/sec2)         |     *)     |
 +--------------------+------------------------------------------+------------+
 | BROADCAST ORBIT - 1| - IODE Issue of Data, Ephemeris          | 3X,4D19.12 |
 |                    | - Crs                 (meters)           |            |
 |                    | - Delta n             (radians/sec)      |            |
 |                    | - M0                  (radians)          |            |
 +--------------------+------------------------------------------+------------+
 | BROADCAST ORBIT - 2| - Cuc                 (radians)          | 3X,4D19.12 |
 |                    | - e Eccentricity                         |            |
 |                    | - Cus                 (radians)          |            |
 |                    | - sqrt(A)             (sqrt(m))          |            |
 +--------------------+------------------------------------------+------------+
 | BROADCAST ORBIT - 3| - Toe Time of Ephemeris                  | 3X,4D19.12 |
 |                    |                       (sec of GPS week)  |            |
 |                    | - Cic                 (radians)          |            |
 |                    | - OMEGA               (radians)          |            |
 |                    | - CIS                 (radians)          |            |
 +--------------------+------------------------------------------+------------+
 | BROADCAST ORBIT - 4| - i0                  (radians)          | 3X,4D19.12 |
 |                    | - Crc                 (meters)           |            |
 |                    | - omega               (radians)          |            |
 |                    | - OMEGA DOT           (radians/sec)      |            |
 +--------------------+------------------------------------------+------------+
 | BROADCAST ORBIT - 5| - IDOT                (radians/sec)      | 3X,4D19.12 |
 |                    | - Codes on L2 channel                    |            |
 |                    | - GPS Week # (to go with TOE)            |            |
 |                    |   Continuous number, not mod(1024)!      |            |
 |                    | - L2 P data flag                         |            |
 +--------------------+------------------------------------------+------------+
 | BROADCAST ORBIT - 6| - SV accuracy         (meters)           | 3X,4D19.12 |
 |                    | - SV health        (bits 17-22 w 3 sf 1) |            |
 |                    | - TGD                 (seconds)          |            |
 |                    | - IODC Issue of Data, Clock              |            |
 +--------------------+------------------------------------------+------------+
 | BROADCAST ORBIT - 7| - Transmission time of message       **) | 3X,4D19.12 |
 |                    |         (sec of GPS week, derived e.g.   |            |
 |                    |    from Z-count in Hand Over Word (HOW)  |            |
 |                    | - Fit interval        (hours)            |            |
 |                    |         (see ICD-GPS-200, 20.3.4.4)      |            |
 |                    |   Zero if not known                      |            |
 |                    | - spare                                  |            |
 |                    | - spare                                  |            |
 +--------------------+------------------------------------------+------------+

'''
import io
from unlzw import unlzw

class Ephemerides:
    def __init__(self, filename):
        # scan until we get a line that contains END OF HEADER
        self._data = []
        
        logger.debug("Broadcast ephemerides file {}".format(filename))
        with open(filename, 'rb') as ifs:
            compressed_data = ifs.read()
            uncompressed_data = unlzw(compressed_data).decode("utf-8") 
            
        with io.StringIO(uncompressed_data) as ifs:
            ''' search header for ionosphere model parameters...
                    1         2         3         4         5         6         7         8
            012345678901234567890123456789012345678901234567890123456789012345678901234567890
                2              NAVIGATION DATA                         RINEX VERSION / TYPE
            CCRINEXN V1.6.0 UX  CDDIS               11-APR-10 02:51     PGM / RUN BY / DATE 
            IGS BROADCAST EPHEMERIS FILE                                COMMENT             
                0.1118E-07  0.1490E-07 -0.5960E-07 -0.5960E-07          ION ALPHA           
                0.8806E+05  0.1638E+05 -0.1966E+06 -0.1311E+06          ION BETA            
            -0.279396772385E-08-0.266453525910E-14    61440     1579 DELTA-UTC: A0,A1,T,W
                15                                                      LEAP SECONDS        
                                                                        END OF HEADER
            '''
            for i in range(100):
                _str = ifs.readline()
                
                if (_str.find("ION ALPHA") != -1):
                    a0 = str_to_float(_str,3,12)
                    a1 = str_to_float(_str,15,12)
                    a2 = str_to_float(_str,27,12)
                    a3 = str_to_float(_str,39,12)
                
                if (_str.find("ION BETA") != -1):
                    b0 = str_to_float(_str,3,12)
                    b1 = str_to_float(_str,15,12)
                    b2 = str_to_float(_str,27,12)
                    b3 = str_to_float(_str,39,12)
                
                if (_str.find("END OF HEADER") != -1):
                    break

            line = _str
            while line:
                
                eph = Ephemeris()
                eph.a0 = a0
                eph.a1 = a1
                eph.a2 = a2
                eph.a3 = a3

                eph.b0 = b0
                eph.b1 = b1
                eph.b2 = b2
                eph.b3 = b3
                
                line = ifs.readline()
            
                '''
                    read in the ephemeris data...
                          1         2         3         4         5         6         7         8
                012345678901234567890123456789012345678901234567890123456789012345678901234567890
                2 08  9 12  0  0  0.0 0.188095495105E-03-0.284217094304E-11 0.000000000000E+00
                   0.670000000000E+02 0.192500000000E+02 0.537165232231E-08-0.256690701048E+01
                   0.110641121864E-05 0.875317445025E-02 0.500865280628E-05 0.515376602745E+04
                   0.432000000000E+06-0.141561031342E-06-0.205415914935E+01 0.109896063805E-06
                   0.942551693336E+00 0.274468750000E+03 0.263802852256E+01-0.833356141200E-08
                   0.553594488004E-10 0.100000000000E+01 0.149600000000E+04 0.000000000000E+00
                   0.200000000000E+01 0.000000000000E+00-0.172294676304E-07 0.670000000000E+02
                   0.430428000000E+06 0.400000000000E+01 0.000000000000E+00 0.000000000000E+00
                '''

                if len(line) < 32:
                    break

                svprn = str_to_int(line,0,2)

                year = str_to_int(line,3,2)
                month = str_to_int(line,6,2)
                day = str_to_int(line,9,2)
                hour = str_to_int(line,12,2)
                minute = str_to_int(line,15,2)
                second = str_to_float(line,18,4)


                eph.toc = GpsTime(2000+year, month, day, hour, minute, second);
                eph.svprn = svprn

                eph.af0 = str_to_float(line,22,19)
                eph.af1 = str_to_float(line,41,19)
                eph.af2 = str_to_float(line,60,19)
            
                line = ifs.readline(); # line 2
                eph.IODE = str_to_float(line,3,19);
                eph.crs = str_to_float(line,22,19);
                eph.deltan = str_to_float(line,41,19);
                eph.M0 = str_to_float(line,60,19);
                line = ifs.readline(); # line 3
                eph.cuc = str_to_float(line,3,19);
                eph.ecc = str_to_float(line,22,19);
                eph.cus = str_to_float(line,41,19);
                eph.roota = str_to_float(line,60,19);
                line = ifs.readline(); # line 4
                eph.toe = str_to_float(line,3,19);
                eph.cic = str_to_float(line,22,19);
                eph.Omega0 = str_to_float(line,41,19);
                eph.cis = str_to_float(line,60,19);
                line = ifs.readline(); # # line 5
                eph.i0 =  str_to_float(line,3,19);
                eph.crc = str_to_float(line,22,19);
                eph.omega = str_to_float(line,41,19);
                eph.Omegadot = str_to_float(line,60,19);

                line = ifs.readline();
                eph.idot = str_to_float(line,3,19);
                eph.codes = str_to_float(line,22,19);
                eph.weekno = str_to_float(line,41,19);
                eph.L2flag = str_to_float(line,60,19);
                
                line = ifs.readline();
                eph.svaccur = str_to_float(line,3,19);
                eph.svhealth = str_to_float(line,22,19);
                eph.tgd = str_to_float(line,41,19);
                # iodc = line(60,19), null;
                
                line = ifs.readline();
                eph.tom = str_to_float(line,3,19);
                eph.fit = str_to_float(line,22,19);
                #	spare = line(41,19), null;
                #    spare = line(60,19), null;

                self._data.append(eph)

    # \brief Get a valid ephemeris for the PRN 
    def get_ephemeris(self, prn, gps_t):
        fail = True;
        
        best_e = None
        tmin = 4*86400.0; # number of seconds that the ephemeris should be younger than (4 days)

        for eph in self._data:

            if (eph.svprn == prn):
                dt = abs(gps_t.diff(eph.toc)) 
                logger.debug(eph.toc, dt)
                if (dt < tmin):
                    tmin =  dt
                    best_e = eph
                    fail = False
                
        # Check the health status of the received satellite!!!!!
        if fail:
            raise RuntimeError("No suitable ephemeris found");

    #	cout << "SV " << PRN << " health = " << best_e.svhealth << " accuracy = " << best_e.svaccur << endl;

        return best_e




class Ephemeris(object):
    GM = 3.986005e14    # earth's universal gravitational parameter m^3/s^2
    WGS84_EARTH_ROTATION_RATE = 7.2921151467e-5;    # earth rotation rate, rad/s

    def __init__(self):
        pass

    def from_hash(self, in_hash):
        ret = Ephemeris()
        ret.from_hash(in_hash)
        return ret


    def to_s(self):
        return "Ephemeris: SV={}, toe={}, toc={}, a0={}, ecc={}, M0={}, roota={}".format(self.svprn, self.toe, self.toc, self.a0, self.ecc, self.M0, self.roota)


    #\brief Find the clock correction from UTC for this SV
    #         This is done ignoring leap seconds. In other words, this is modulo 1 second.
    #
    def clock_correct(self, Tsv):
        # more accurate calculation
        e = self.getE(Tsv)
        dt = Util.check_t(Tsv - self.toc.sow())
        dtr = -4.442807e-10 * self.ecc * self.roota * math.sin(e)     #relatavistic correction
        dtsv = self.af0 + self.af1*(dt) + self.af2*dt*dt + dtr - self.tgd
        return dtsv


    def get_tk(self, sow):
        tk = Util.check_t(sow-self.toe) #        Time from ephemeris reference epoch (1)
        if (tk < -302400.0):
            raise RuntimeError(f"Invalid time {sow}, toe={self.toe}")
        if (tk > 302400.0):
            raise RuntimeError(f"Invalid time {sow}, toe={self.toe}")
        return tk

    # http:#home-2.worldonline.nl/~samsvl/satpos.htm
    # Get the eccentricity
    def getE0(self, sow):

        x=self.M0             # kepler's equation for eccentric anomaly ek
        y=self.M0 - (x-self.ecc*math.sin(x))
        x1=x
        x=y
        for i in range(0,15):
            x2=x1
            x1=x
            y1=y
            y = self.M0 - (x- self.ecc*math.sin(x))
            if (abs(y-y1)<1.0e-15):
                break
            x=(x2*y-x*y1)/(y-y1)
        ek=x        # end of det. of ecc. anomaly

        tk = self.get_tk(sow)
        n0 = math.sqrt(Ephemeris.GM/(self.roota**6)) #        Computed mean motion
        n = n0+self.deltan                        #        Corrected mean motion
        mk=self.M0+n*tk    #    mean anomaly

        x=mk    #    kepler's equation for eccentric anomaly ek
        y=mk-(x- self.ecc* math.sin(x))
        x1=x
        x=y
        for i in range(0,15):
            x2=x1
            x1=x
            y1=y
            y=mk-(x- self.ecc* math.sin(x))
            if(abs(y-y1)<1.0E-15):
                break
            x=(x2*y-x*y1)/(y-y1)
        ek=x

        print(("    E0 -> {}".format(ek)))
        return ek

    def getE(self, sow):
        a = self.roota*self.roota
        tk = self.get_tk(sow)
        n0 = math.sqrt(Ephemeris.GM/(a**3)) #  Computed mean motion
        n = n0+self.deltan                  #  Corrected mean motion
        m = self.M0+n*tk                    #  Mean anomaly

        #test = self.getE0(sow)
        m = Util.rem2pi(m + Util.PI2)
        e = m
        for i in range(0,15):
            e_old = e
            e = m + self.ecc*math.sin(e_old)
            dE = Util.rem2pi(e - e_old)
            if (abs(dE) < 1.0e-15):
                break

        e = Util.rem2pi(e + Util.PI2)
        return e


    def get_sv_position(self, gt):
        return self.get_location(gt.sow())

    def get_sv_position_utc(self, utc_datetime):
        gpst = gps_time.GpsTime.from_time(utc_datetime)
        return self.get_location(gpst.sow())

    def get_location(self, sow):
        '''
        
        Add a good testbench.
        https://ascelibrary.org/doi/pdf/10.1061/9780784411506.ap03
        '''
        a = self.roota*self.roota        # Semi major axis
        tk = self.get_tk(sow) #     tk = sow-@toe


        e = self.getE(sow)

        v = math.atan2(math.sqrt(1.0-(self.ecc**2))*math.sin(e), math.cos(e)-self.ecc)
        phi = v+self.omega
        phi = Util.rem2pi(phi)
        phi2 = 2.0*phi

        cosphi2 = math.cos(phi2)
        sinphi2 = math.sin(phi2)

        u = phi + self.cuc*cosphi2+self.cus*sinphi2
        r = a*(1.0-self.ecc*math.cos(e)) + self.crc*cosphi2+self.crs*sinphi2
        i = self.i0+self.idot*tk + self.cic*cosphi2+self.cis*sinphi2
        om = self.Omega0 + (self.Omegadot - Ephemeris.WGS84_EARTH_ROTATION_RATE)*tk - \
            Ephemeris.WGS84_EARTH_ROTATION_RATE*self.toe
        om = Util.rem2pi(om + Util.PI2)
        logger.debug(("w_c={}, wdot={}, om={}".format(self.Omega0, self.Omegadot, om)))
        x1 = math.cos(u)*r
        y1 = math.sin(u)*r

        x = x1*math.cos(om) - y1*math.cos(i)*math.sin(om)
        y = x1*math.sin(om) + y1*math.cos(i)*math.cos(om)
        z = y1*math.sin(i)
        logger.debug(("ephemeris.get_location({}) tk={}, {} -> x:{}".format(sow, tk, self.to_s(), x)))
        return np.array([x,y,z])

    
    def get_location_new(self, sow):
        ''' A sanity check from a different source
            http:#gnsstk.sourceforge.net/gps_8c-source.html
            https:#gssc.esa.int/navipedia/index.php/GPS_and_Galileo_Satellite_Coordinates_Computation
        '''
        a = self.roota*self.roota        # Semi major axis
        tk = Util.check_t(sow - self.toe)

        Mk =  self.M0 + (math.sqrt(Ephemeris.GM/(a**3)) + self.deltan)*tk

        E_0 = Mk
        for _ in range(12):
            E_k = Mk + self.ecc*math.sin(E_0)
            E_0 = E_k

        v_k = math.atan(math.sqrt(1.0-(self.ecc**2))*math.sin(E_k)/math.cos(E_k)-self.ecc)

        phi = v_k+self.omega
        phi2 = 2.0*phi

        cosphi2 = math.cos(phi2)
        sinphi2 = math.sin(phi2)

        u_k = phi + self.cuc*cosphi2 + self.cus*sinphi2
        r_k = a*(1.0 - self.ecc*math.cos(E_k)) + self.crc*cosphi2 + self.crs*sinphi2
        i_k = self.i0 + self.idot*tk + self.cic*cosphi2 + self.cis*sinphi2
        
        lambda_k = self.Omega0 + (self.Omegadot - Ephemeris.WGS84_EARTH_ROTATION_RATE)*tk - \
            Ephemeris.WGS84_EARTH_ROTATION_RATE*self.toe

        x1 = math.cos(u_k)*r_k
        y1 = math.sin(u_k)*r_k

        x = x1*math.cos(lambda_k) - y1*math.cos(i_k)*math.sin(lambda_k)
        y = x1*math.sin(lambda_k) + y1*math.cos(i_k)*math.cos(lambda_k)
        z = y1*math.sin(i_k)
        return [x,y,z]

    def get_velocity(self, sow):
        dt = 0.1
        loc1 = np.array(self.get_location(sow - dt))
        loc2 = np.array(self.get_location(sow + dt))
        return (loc1 - loc2) * 0.5 / dt
