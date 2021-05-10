import numpy as np
import matplotlib.pyplot as plt
import datetime
import logging
import numpy as np
import pymc3 as pm
import theano.tensor as tt
#import theano.tests.unittest_tools as utt
import arviz as az

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler()) # Add a null handler so logs can go somewhere
logger.setLevel(logging.INFO)

from .fastfix import doppler_fmap, Satellite, phase_fmap
from .gps_time import GpsTime 
from .location import Location
from .angle import from_dms
from .util import Util


class PhaseLogLike(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, acq, t0, ephs):
        self.svs = acq['sv']
        self.data = np.array(acq['codephase'])  # Measured data
        self.xmax = np.array(acq['x_max'])
        self.sats = [Satellite(sv, ephs.get_ephemeris(prn=sv, gps_t=t0)) for sv in self.svs]

    def perform(self, node, inputs, outputs):
        theta, = inputs
        lat, lon, alt, offset, sow  = theta
        
        try:
            pred_phases, pred_xmax, pred_prn = phase_fmap(self.sats, lat, lon, alt, offset, sow)
            
            logp = 0.0
            
            for meas_ph, pred_ph, meas_xm, pred_xm in zip(self.data, pred_phases, self.xmax, pred_xmax):
                sigma = 0.05
                if meas_xm > 12:
                    sigma = 0.0002    # Should be close to one sample which is 1/8k
                if pred_xm < 8.0 or meas_xm < 8.0:
                    sigma = 2.0   # These should be ignored, therefore a huge variance.
                    
                logp += -np.log(np.sqrt(2.0*np.pi)*sigma)
                logp += -(Util.phase_delta(meas_ph - pred_ph))**2.0/(2.0*sigma**2.0)
        except Exception as e:
            logger.info(f"Exception {e}: param: {theta}")
            logp = -999.0e99
            
        outputs[0][0] = np.array(logp)


class DopplerLogLike(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, acq, t0, ephs):
        svs = acq['sv']
        self.xmax = acq['x_max']
        self.sats = [Satellite(sv, ephs.get_ephemeris(prn=sv, gps_t=t0)) for sv in svs]

        x,v = [], []
        for sv in svs:
            eph = ephs.get_ephemeris(prn=sv, gps_t=t0)
            x.append(eph.get_location(t0.sow()))
            v.append(eph.get_velocity(t0.sow()))
        
        logger.info(f"SV Velocities {v}")
        logger.info(f"SV positions {x}")
        
        # Calculate elevations.
        
        self.x = np.array(x)
        self.v = np.array(v)
        self.data = np.array(acq['doppler'])  # Measured data
        

    def perform(self, node, inputs, outputs):
        theta, = inputs
        lat, lon, delta_f = theta
        
        N = self.x.shape[0]
        #logger.info(f"param: {theta}, N={N}")
        mu = doppler_fmap(lat, lon, delta_f, self.x, self.v)

        rp = Location(from_dms(lat),from_dms(lon), 0.0);    # Reviever location in ECEF
        r_0 = rp.get_ecef()
                
        logp = 0
        for sim, meas, x, xmax, sv in zip(mu, self.data, self.x, self.xmax, self.sats):
            sigma = 200.0

            # Change sigma when elevations are below 5 degrees.
            elevation = Satellite.elev_angle(x, r_0)
            #   For low xmax, and low elevation, the sigma should be somwehere like 3000
            if elevation < 5 and xmax < 9:
                sigma = 300.0 * (10.0 - xmax)
        
            logp += -np.log(np.sqrt(2.0*np.pi)*sigma)
            logp += -np.sum((meas - sim)**2.0/(2.0*sigma**2.0))
            
        outputs[0][0] = np.array(logp)


class DopplerPhaseLogLike(tt.Op):
    def __init__(self, acq, t0, ephs):
        self.doll = DopplerLogLike(acq, t0, ephs)
        self.phll = PhaseLogLike(acq, t0, ephs)

    def perform(self, node, inputs, outputs):
        lat, lon, alt, offset, sow, delta_f  = inputs
        out_do = outputs.copy()
        out_ph = outputs.copy()
        
        self.doll.perform(node, (lat, lon, delta_f), out_do)
        self.phll.perform(node, (lat, lon, alt, offset, sow), out_ph )
        
        outputs = out_do + out_ph

def sub_stats(stat1, stat2, key):
    p1 = stat1[key].to_dict()
    p2 = stat2[key].to_dict()
    
    p1['lon'] = p2['lon']
    return p1


def characterize_posterior(trace, plot=False, plot_title="trace"):
    stats = pm.summary(trace)
    print(stats.to_string())

    if plot:
        az.plot_trace(trace)
        plt.savefig(f'{plot_title}_chain_histogram.pdf')
        #plt.show()
        
        az.plot_pair(
            trace,
            var_names=["lat", "lon"],
            kind="hexbin",
            marginals=True,
            #figsize=(8, 6),
        )
        plt.savefig(f'{plot_title}_joint_lat_lon.pdf')
        #plt.show()

    rhat = stats['r_hat']
    
    func_dict = {
        "std": np.std,
        "5%": lambda x: np.percentile(x, 5),
        "median": lambda x: np.percentile(x, 50),
        "95%": lambda x: np.percentile(x, 95),
    }
    
    stats = az.summary(
        trace,
        stat_funcs=func_dict,
        round_to=6,
        extend=False)
    
    # Now scale the longitude to have the median at 0.0 (this avoids wraparound at +/- 180.0
    lon_med = stats['median']['lon']
    
    
    def wrap180(x, med):
        y = x - med
        return ((y + 180) % 360) - 180 + med 
    
    func_dict = {
        "std": lambda x: np.std(wrap180(x, lon_med)),
        "5%": lambda x: np.percentile(wrap180(x, lon_med), 5),
        "median": lambda x: np.percentile(wrap180(x, lon_med), 50),
        "95%": lambda x: np.percentile(wrap180(x, lon_med), 95)
    }

    stats2 = az.summary(
        trace,
        stat_funcs=func_dict,
        round_to=6,
        extend=False)

    ret = {}
    for key in ['std', '5%', 'median', '95%']:
        ret[key] = sub_stats(stats, stats2, key)
        
    # now invert
    ret_swap = {}
    for k in stats['median'].keys():
        ret_swap[k] = {'r_hat': rhat[k] }
        for k2 in ['std', '5%', 'median', '95%']:
            ret_swap[k][k2] = ret[k2][k]

    return ret_swap


def process_mcmc(acq, start_date, brdc_proxy, local_clock_offset, plot=False):
    
    clock_offset, clock_offset_std = local_clock_offset
    
    rtc_offset = acq['rtc']
    
    t0_uncorrected = start_date + datetime.timedelta(seconds = rtc_offset)
    t0 = start_date + datetime.timedelta(seconds = rtc_offset+clock_offset)
    
    logger.info(f"FastFix MCMC processing: t0={t0.isoformat()} offset={local_clock_offset}")
    acq['t0'] = t0.isoformat()
    acq['local_t0'] = t0_uncorrected.isoformat()
    acq['local_clock_offset'] = local_clock_offset

    ephs = brdc_proxy.get_ephemerides(t0)
    
    gps_t = GpsTime.from_time(t0)
    acq['gps_t'] = gps_t.to_dict()

    do_loglike = DopplerLogLike(acq, gps_t, ephs)
    ph_loglike = PhaseLogLike(acq, gps_t, ephs)

    with pm.Model() as model:
        lat = pm.Uniform('lat', lower=-90.0, upper=90.0, testval=-30.0)
        lon = pm.Uniform('lon', lower=-180, upper=180.0, testval=60.0)
        delta_f = pm.Normal('delta_f', mu=0.0, sigma=100.0, testval=0.0)
        alt = pm.Uniform('alt', lower=0.0, upper=1000.0)
        offset = pm.Uniform('offset', lower=0, upper=1)
        
        clk_allowed_range = 2.0*clock_offset_std + 1.0
        
        sow = pm.Uniform('sow', lower=gps_t.sow() - clk_allowed_range, upper=gps_t.sow() + clk_allowed_range)   # Add half a second as the rtc is only accurate to 1 second.
        theta_do = tt.as_tensor_variable([lat, lon, delta_f])
        theta_ph = tt.as_tensor_variable([lat, lon, alt, offset, sow])
        like = pm.Potential('like', 10*do_loglike(theta_do) + ph_loglike(theta_ph))
    with model:
        step = pm.Metropolis([lat, lon, alt, offset, sow, delta_f])
        trace = pm.sample(5000, step = step, random_seed=123, chains=4)
        phase_stats = characterize_posterior(trace, plot=plot, plot_title="joint")
        acq['joint_mcmc'] = phase_stats

    #loglike = DopplerLogLike(acq, gps_t, ephs)
    #with pm.Model() as model:
        #lat = pm.Uniform('lat', lower=-90.0, upper=90.0, testval=-30.0)
        #lon = pm.Uniform('lon', lower=-180, upper=180.0, testval=60.0)
        #delta_f = pm.Normal('delta_f', mu=0.0, sigma=100.0, testval=0.0)
        #theta = tt.as_tensor_variable([lat, lon, delta_f])
        #like = pm.Potential('like', loglike(theta))
    #with model:
        #trace = pm.sample(1000)
        
        #doppler_stats = characterize_posterior(trace, plot=plot, plot_title="doppler")
        #acq['doppler_mcmc'] = doppler_stats
        #logger.info(doppler_stats)
    
    
    #loglike = PhaseLogLike(acq, gps_t, ephs)
    #with pm.Model() as model:
        #lat = pm.Normal('lat', mu=doppler_stats['lat']['median'], sigma = doppler_stats['lat']['std'])
        #lon = pm.Normal('lon', mu=doppler_stats['lon']['median'], sigma = doppler_stats['lon']['std'])
        #alt = pm.Uniform('alt', lower=0.0, upper=1000.0)
        #offset = pm.Uniform('offset', lower=0, upper=1)
        
        #clk_allowed_range = 2.0*clock_offset_std + 1.0
        
        #sow = pm.Uniform('sow', lower=gps_t.sow() - clk_allowed_range, upper=gps_t.sow() + clk_allowed_range)   # Add half a second as the rtc is only accurate to 1 second.
        #theta = tt.as_tensor_variable([lat, lon, alt, offset, sow])
        #like = pm.Potential('like', loglike(theta))
    #with model:
        #step1 = pm.Slice([lat, lon, sow])
        #step2 = pm.Metropolis([alt, offset])
        ##step = pm.DEMetropolis()
        ##step = pm.DEMetropolisZ()  # Cajo C.F. ter Braak (2006). Differential Evolution Markov Chain with snooker updater and fewer chains. Statistics and Computing
        ##step = pm.Slice()  # Cajo C.F. ter Braak (2006). Differential Evolution Markov Chain with snooker updater and fewer chains. Statistics and Computing

        #trace = pm.sample(1000, step = [step1, step2], random_seed=123, chains=4)
        #phase_stats = characterize_posterior(trace, plot=plot, plot_title="phase")
        #acq['phase_mcmc'] = phase_stats
        
    new_sow_err = phase_stats['sow']['std']
    if (new_sow_err < (clock_offset_std + 0.5)) and (new_sow_err > 1e-3):
        gps_t_uncorrected = GpsTime.from_time(t0_uncorrected)

        clock_offset = phase_stats['sow']['median'] - gps_t_uncorrected.sow()
        clock_offset_std = max(float(phase_stats['sow']['std']), 0.5)
    return (clock_offset, clock_offset_std)
