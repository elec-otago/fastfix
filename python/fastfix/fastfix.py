import logging
import datetime
import numpy as np
from scipy.optimize import minimize
from scipy import optimize

from .ephemeris import Ephemerides
from .gps_time import GpsTime 
from .location import Location
from .angle import from_dms
from .util import Util

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler()) # Add a null handler so logs can go somewhere
logger.setLevel(logging.INFO)


def process(acq, start_date, brdc_proxy, estimated_clock_offset, plot):
    '''
        Do the FastFix process.
        
        Parameters:
            acq (dict): The acquisition result in JSON format
            start_date (datetime): The tag start date used to calculate the estimated fix datetime.
            brdc_proxy (GpsFileCache): A proxy for requesting Broadcast ephemerides
            
        Returns:
            None. This function modifies  the acq parameter, adding appropriate fields
            i.e., 'doppler_fix', 'codephase_fix', 'spacetime_fix'

        Example acq:
        
        \param  acq       {
            "codephase": [
                0.9564951729194672,
                0.16377758597064598
            ],
            "doppler": [
                1060.6060606060637,
                2654.8400905867384
            ],
            "infile": "/freenas/tag/albatross_data/AF13/00/FIX00149.BIN",
            "rtc": 89564,
            "sample_ms": 4,
            "sampling_rate": 8183833,
            "sv": [
                7,
                8
            ],
            "x_max": [
                7.3863348937104805,
                14.421487133793908
            ]
        },
    '''
    rtc_offset = acq['rtc']
    
    local_clock_offset, clock_offset_std = estimated_clock_offset
    
    t0_uncorrected = start_date + datetime.timedelta(seconds = rtc_offset)
    t0 = start_date + datetime.timedelta(seconds = rtc_offset + local_clock_offset)
    
    logger.info(f"FastFix processing: t0={t0.isoformat()}, offset={estimated_clock_offset}")
    
    ephs = brdc_proxy.get_ephemerides(t0)
    
    gps_t = GpsTime.from_time(t0)
    acq['t0'] = t0.isoformat()
    acq['t_gps'] = gps_t.to_dict()
    acq['t0_uncorrected'] = t0_uncorrected.isoformat()
    acq['local_clock_offset'] = local_clock_offset

    print(acq)
    # Doppler Fix
    r0 = doppler_fix(acq, gps_t, ephs)
    acq['doppler_fix'] = r0
    
    r1, residual = phase_fix(acq, r0, gps_t, ephs)
    acq['phase_fix'] = r1
    
    # Do clock correction
    gps_t_uncorrected = GpsTime.from_time(t0_uncorrected)
    
    potential_offset = r1['sow'] - gps_t_uncorrected.sow()
    logger.info(f"Potential Offset {potential_offset}")
    
    if (residual < -2e7) and (np.abs(potential_offset) < 5) :
        local_clock_offset = potential_offset
    return (local_clock_offset, clock_offset_std)
    
    
    

class Satellite:
    
    def __init__(self, prn, eph):
        self.prn = prn
        self.eph = eph
        pass
    
    def location(self, sow):
        return self.eph.get_location(sow)
    
    def clock_correct(self, sow):
        return self.eph.clock_correct(sow)

    @classmethod
    def elev_angle(self, sv_location, receiver_location):
        dr = sv_location - receiver_location
        dr_unit = dr / np.linalg.norm(dr)

        rec_pos_unit = receiver_location / np.linalg.norm(receiver_location)
        dot_prod = np.dot(rec_pos_unit, dr_unit)
        theta = np.arccos(dot_prod)
        return np.degrees(np.pi/2 - theta)

    def elevation(self, sow, receiver_location):
        return Satellite.elev_angle(self.location(sow), receiver_location)

    def doppler(self, sow, receiver_location):
        dt = 0.1
        range1 = self.dist(sow-dt, receiver_location);
        range2 = self.dist(sow+dt, receiver_location);
        velocity = (range1 - range2) * 0.5 / dt
        return velocity / Util.WAVELENGTH

    ''' \brief Get the range to this satellite from the receiver_location.
            \param sow GPS time in seconds of the week
            \param receiver_location Receiver location (in rectangular co-ordinates)
            \return range in meters.
    '''
    def dist(self, sow, receiver_location):
        rx_time = sow

        dr = receiver_location - self.location(rx_time)
        ret = np.linalg.norm(dr)

        dt = ret / Util.C
        tx_time = rx_time - dt

        # recalculate satellite location and range at the time of sending
        r = receiver_location - self.location(tx_time)
        ret =  np.linalg.norm(dr)
        return ret

    def codephase(self, sow, receiver_location):

        dt = self.dist(sow, receiver_location) / Util.C
        ms = dt * 1000.0
        
        # codephase is started on the millisecond, and received at an arbitrary time.
        uncorrected_codephase = ms % 1.0
        
        corrected_codephase = uncorrected_codephase - 1000.0*self.clock_correct(sow)
        ret = np.mod(corrected_codephase,1)
        return ret


def signal_strength(elevation):
    if (elevation < 0):
        return 0.0;
    
    return np.sqrt(np.sin(elevation))



def doppler_fmap(lat, lon, 
                 delta_f,   # Center frequency offset
                 x_sv,      # Satellite Positions ECEF
                 v_sv):     # Satellite Velocities ECEF
    '''
        Predict doppler at position r for the list of svs
    '''

    rp = Location(from_dms(lat),from_dms(lon), 0.0);    # Reviever location in ECEF
    r_0 = rp.get_ecef()

    N = x_sv.shape[0]
    ret = np.zeros(N)
    
    for i in range(N):
        r_s = x_sv[i]
        v_s = v_sv[i]

        dr = (r_s - r_0)
        dr_unit = dr / np.linalg.norm(dr)
        
        # component of velocity in the direction of dr (from r_0 to r_s)
        v_r = np.dot(dr_unit, v_s)
        
        sim_doppler = v_r / Util.WAVELENGTH
        ret[i] = sim_doppler + delta_f
    return ret


def phase_fmap(svs,
                lat, lon, alt, 
                offset,     # Phase offset
                sow         # Absolute GPS second of week
              ):

    rp = Location(from_dms(lat),from_dms(lon), alt);    # Reciever location in ECEF
    r_0 = rp.get_ecef()

    pred_phases = []
    pred_xmax = []
    pred_prn = []
    
    for s in svs:    
        sim_phase = np.mod(s.codephase(sow, r_0) + offset,1)
        elevation = s.elevation(sow, r_0)
        
        xmax = 7 + 15*signal_strength(np.radians(elevation))
        pred_phases.append(sim_phase)
        pred_xmax.append(xmax)
        pred_prn.append(s.prn)

    return pred_phases, pred_xmax, pred_prn

def doppler_f_opt(x, x_sv, v_sv, xmaxs, meas_doppler):
    lat, lon, delta_f = x

    sim_dopplers = doppler_fmap(lat, lon, delta_f, x_sv, v_sv)
    ret = 0.0
    for xmax, df, sd in zip( xmaxs, sim_dopplers, meas_doppler): 
        ret += (df - sd)**2 * np.max(xmax - 7, 0)
    
    return ret


def doppler_fix(acq, t0, ephs):
    svs = acq['sv']
    xmax = acq['x_max']
    x,v = [], []
    for sv in svs:
        eph = ephs.get_ephemeris(prn=sv, gps_t=t0)
        sat = Satellite(prn=sv, eph=eph)
        x.append(sat.location(t0.sow()))
        v.append(eph.get_velocity(t0.sow()))
    
    logger.info(f"SV Velocities {v}")
    logger.info(f"SV positions {x}")
    
    x = np.array(x)
    v = np.array(v)
    dopp = np.array(acq['doppler'])
    
    x0 = [-45,180, 0]
    r0 = minimize(doppler_f_opt, x0, method='BFGS',  args=(x, v, xmax, dopp), options={'gtol': 1e-6, 'disp': True})
    print(f"Doppler result {r0}")
    return { 'lat': r0.x[0],
            'lon': r0.x[1],
            'delta_f': r0.x[2] }



def phase_f_opt(x, sats, xmaxs, meas_phase):
    lat, lon, alt, offset, sow  = x

    pred_phases, pred_xmax, pred_prn = phase_fmap(sats, lat, lon, alt, offset, sow)
    logp = 0.0
    for meas_ph, pred_ph, meas_xm, pred_xm in zip(meas_phase, pred_phases, xmaxs, pred_xmax):
        sigma = 0.05
        if meas_xm > 10:
            sigma = 0.00002    # Should be close to one sample which is 1/8k
        if pred_xm < 8.0 or meas_xm < 8.0:
            sigma = 2.0   # These should be ignored, therefore a huge variance.
            
        logp += -np.log(np.sqrt(2.0*np.pi)*sigma)
        logp += -(Util.phase_delta(meas_ph - pred_ph))**2.0/(2.0*sigma**2.0)
    
    return logp


def phase_fix(acq, r0, t0, ephs):
    svs = acq['sv']
    meas_phase = np.array(acq['codephase'])  # Measured data
    xmax = np.array(acq['x_max'])
    sats = [Satellite(sv, ephs.get_ephemeris(prn=sv, gps_t=t0)) for sv in svs]
        
    x0 = [r0['lat'], r0['lon'], 5, 0.5, t0.sow()]
    
    bounds = [(x0[0]-3,x0[0]+3), (x0[1] - 3, x0[1]+3), (0,1000), (0,1), (t0.sow()-5, t0.sow()+5)]
    
    if True:
        method = "basinhopping"
        minimizer_kwargs = {"method":"L-BFGS-B", "jac":False, "bounds":bounds, "tol":1e-5, "options":{"maxcor":48}, "args": (sats, xmax, meas_phase)}
        r0 = optimize.basinhopping(phase_f_opt, x0, niter=100, T = 100, stepsize=1.0, disp=True, minimizer_kwargs=minimizer_kwargs)
    else:
        method='L-BFGS-B'
        r0 = minimize(phase_f_opt, x0, method=method,  bounds=bounds, args=(sats, xmax, meas_phase), options={'gtol': 1e-6, 'disp': True})
    print(f"Phase result {r0}")
    return { 'lat': r0.x[0],
            'lon': r0.x[1],
            'alt': r0.x[2],
            'sow': r0.x[4],
            'residual': r0.fun,
            'method': method}, r0.fun
