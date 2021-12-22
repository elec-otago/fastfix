import numpy as np
import matplotlib.pyplot as plt
import datetime
import logging
import numpy as np
import pymc3 as pm
import theano.tensor as tt

# import theano.tests.unittest_tools as utt
import arviz as az

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Add a null handler so logs can go somewhere
logger.setLevel(logging.INFO)

from .fastfix import doppler_fmap, Satellite, phase_fmap
from .gps_time import GpsTime
from .location import Location
from .angle import from_dms
from .util import Util
from .vmf import VMF
                
    

class PhaseLogLike(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, acq, t0, ephs):
        self.svs = acq["sv"]
        self.data = np.array(acq["codephase"])  # Measured data
        self.xmax = np.array(acq["x_max"])
        self.sats = [
            Satellite(sv, ephs.get_ephemeris(prn=sv, gps_t=t0)) for sv in self.svs
        ]

    def perform(self, node, inputs, outputs):
        (theta,) = inputs
        lat, lon, alt, offset, sow = theta

        try:
            pred_phases, pred_xmax, pred_prn = phase_fmap(
                self.sats, lat, lon, alt, offset, sow
            )

            logp = 0.0

            for meas_ph, pred_ph, meas_xm, pred_xm in zip(
                self.data, pred_phases, self.xmax, pred_xmax
            ):
                sigma = 0.05
                #if meas_xm > 12:
                    #sigma = 0.0002  # Should be close to one sample which is 1/8k
                #if pred_xm < 8.0 or meas_xm < 8.0:
                    #sigma = 2.0  # These should be ignored, therefore a huge variance.

                logp += Util.gaussian_llh(x=meas_ph, mu=pred_ph, sigma=sigma)
                
        except Exception as e:
            logger.info(f"Exception {e}: param: {theta}")
            logp = -999.0e99

        outputs[0][0] = np.array(logp)


class DopplerLogLike(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, acq, t0, ephs):
        svs = acq["sv"]
        self.xmax = acq["x_max"]
        self.sats = [Satellite(sv, ephs.get_ephemeris(prn=sv, gps_t=t0)) for sv in svs]

        x, v = [], []
        for sv in svs:
            eph = ephs.get_ephemeris(prn=sv, gps_t=t0)
            x.append(eph.get_location(t0.sow()))
            v.append(eph.get_velocity(t0.sow()))

        logger.info(f"SV Velocities {v}")
        logger.info(f"SV positions {x}")

        # Calculate elevations.

        self.x = np.array(x)
        self.v = np.array(v)
        self.data = np.array(acq["doppler"])  # Measured data

    def perform(self, node, inputs, outputs):
        (theta,) = inputs
        lat, lon, delta_f = theta

        N = self.x.shape[0]
        # logger.info(f"param: {theta}, N={N}")
        mu = doppler_fmap(lat, lon, delta_f, self.x, self.v)

        rp = Location(from_dms(lat), from_dms(lon), 0.0)
        # Reviever location in ECEF
        r_0 = rp.get_ecef()

        logp = 0
        for sim, meas, x, xmax, sv in zip(mu, self.data, self.x, self.xmax, self.sats):
            sigma = 200.0

            # Change sigma when elevations are below 5 degrees.
            elevation = Satellite.elev_angle(x, r_0)
            #   For low xmax, and low elevation, the sigma should be somwehere like 3000
            ##if elevation < 5 and xmax < 9:
                ##sigma = 300.0 * (10.0 - xmax)

            logp += Util.gaussian_llh(x=meas, mu=sim, sigma=200.0)

        outputs[0][0] = np.array(logp)


class DopplerPhaseLogLike(tt.Op):
    def __init__(self, acq, t0, ephs):
        self.doll = DopplerLogLike(acq, t0, ephs)
        self.phll = PhaseLogLike(acq, t0, ephs)

    def perform(self, node, inputs, outputs):
        lat, lon, alt, offset, sow, delta_f = inputs
        out_do = outputs.copy()
        out_ph = outputs.copy()

        self.doll.perform(node, (lat, lon, delta_f), out_do)
        self.phll.perform(node, (lat, lon, alt, offset, sow), out_ph)

        outputs = out_do + out_ph


def sub_stats(stat1, stat2, key):
    p1 = stat1[key].to_dict()
    p2 = stat2[key].to_dict()

    p1["lonlat[0]"] = p2["lonlat[0]"]
    return p1


def characterize_posterior(trace, plot=False, plot_title="trace"):
    stats = pm.summary(trace)
    print(stats.to_string())
    print(trace)
    names = [x for x in trace.posterior.mean()]
    print(f"Names:  {names}")
    
    if plot:
        az.plot_trace(trace)
        plt.savefig(f"{plot_title}_chain_histogram.pdf")
        #plt.show()

        az.plot_pair(
            trace,
            #var_names=["lonlat[0]", "lonlat[1]"],
            kind="hexbin",
            marginals=True,
            # figsize=(8, 6),
        )
        plt.savefig(f"{plot_title}_joint_lat_lon.pdf")
        #plt.show()

    rhat = stats["r_hat"]

    func_dict = {
        "std": np.std,
        "5%": lambda x: np.percentile(x, 5),
        "median": lambda x: np.percentile(x, 50),
        "95%": lambda x: np.percentile(x, 95),
    }

    stats = az.summary(trace, stat_funcs=func_dict, round_to=6, extend=False)
    chain=0
    position_samples = trace.posterior.get('lonlat').values[chain, :]
    lon_samples = position_samples[:,0]
    lon_med = np.mean(lon_samples)
    # Now scale the longitude to have the median at 0.0 (this avoids wraparound at +/- 180.0
    #lon_med = stats["median"]["lonlat[0]"]

    def wrap180(x):
        return ((x + 180) % 360) - 180

    func_dict = {
        "std": lambda x: np.std(x),
        "5%": lambda x: np.percentile(wrap180(x), 5),
        "median": lambda x: np.percentile(wrap180(x), 50),
        "95%": lambda x: np.percentile(wrap180(x), 95),
    }

    stats2 = az.summary(trace, stat_funcs=func_dict, round_to=6, extend=False)
    
    
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

def do_mcmc(model, n_samples=3000, method='NUTS'):
    with model:
        n_tune = n_samples
        n_chains = 4
        if method == 'NUTS':
            start = pm.find_MAP()
            idata = pm.sample(n_samples, init='advi+adapt_diag', tune=n_tune, chains=n_chains, start=start, return_inferencedata=True, discard_tuned_samples=True)
        else:
            idata = pm.sample_smc(n_samples, parallel=True)

    return idata
    
def process_mcmc(acq, start_date, brdc_proxy, local_clock_offset, plot=False):

    clock_offset, clock_offset_std = local_clock_offset

    rtc_offset = acq["rtc"]

    t0_uncorrected = start_date + datetime.timedelta(seconds=rtc_offset)
    t0 = start_date + datetime.timedelta(seconds=rtc_offset + clock_offset)

    logger.info(
        f"FastFix MCMC processing: t0={t0.isoformat()} offset={local_clock_offset}"
    )
    acq["t0"] = t0.isoformat()
    acq["local_t0"] = t0_uncorrected.isoformat()
    acq["local_clock_offset"] = local_clock_offset

    ephs = brdc_proxy.get_ephemerides(t0)

    gps_t = GpsTime.from_time(t0)
    acq["gps_t"] = gps_t.to_dict()

    do_loglike = DopplerLogLike(acq, gps_t, ephs)
    ph_loglike = PhaseLogLike(acq, gps_t, ephs)

    #### DO THE DOPPLER FiX
    with pm.Model() as model:

        lonlat = VMF("lonlat", k=0.01, shape=2, testval=np.array([0.0,0.0]))
        lon = lonlat[0]
        lat = lonlat[1]
        
        delta_f = pm.Normal("delta_f_khz", mu=0.0, sigma=1.0, testval=0.0) * 1000

        theta_do = tt.as_tensor_variable([lat, lon, delta_f])
        like = pm.Potential("like", do_loglike(theta_do))
    
    idata = do_mcmc(model, n_samples=1000)
    doppler_stats = characterize_posterior(idata, plot=plot, plot_title=f"doppler_joint_{t0_uncorrected.isoformat()}")
    acq["doppler_mcmc"] = doppler_stats
    print(doppler_stats)
    
    #### NOW DO THE PHASE FiX
    with pm.Model() as model:

        ## 1 / kappa = sigma^2 => kappa = 1 / sigma^2 (assume sigma = np.degrees(3))
        sigma_fix = doppler_stats['lonlat[0]']['std']
        kappa = 0.1 / np.radians(sigma_fix)**2
        print(f"sigma_fix = {sigma_fix}")
        print(f"kappa = {kappa}")
        
        lon_start = doppler_stats['lonlat[0]']['median']
        lat_start = doppler_stats['lonlat[1]']['median']
        
        print(f"lonlat_start = {[lon_start, lat_start]}")
        
        if False:
            lonlat = VMF("lonlat", lon_lat=[lon_start, lat_start],  k=kappa, shape=2, testval=np.array([lon_start, lat_start]))
            lon = lonlat[0]
            lat = lonlat[1]
        else:
            lon = pm.Normal('lonlat[0]', mu = lon_start, sigma=doppler_stats['lonlat[0]']['std'])
            lat = pm.Normal('lonlat[1]', mu = lat_start, sigma=doppler_stats['lonlat[1]']['std'])
        
        alt = pm.HalfNormal("alt", sigma=1.0) * 1000
        if False:
            offset = pm.Uniform("phase_offset", lower=0, upper=1)
        else:
            offset = (pm.VonMises("phase_offset", mu=0, kappa=0.01) + np.pi) / (2*np.pi)
        #sow = pm.VonMises(
            #"sow",
            #mu=0,
            #kappa=0.01,
        #)*(clock_offset_std+0.5) / np.pi + gps_t.sow() # Add half a second as the rtc is only accurate to 1 second.

        clk_err = (2*clock_offset_std+0.5)
        sow = pm.Uniform("sow_offset",  lower=-clk_err, upper=clk_err)  + gps_t.sow() # Add half a second as the rtc is only accurate to 1 second.

        theta_ph = tt.as_tensor_variable([lat, lon, alt, offset, sow])
        phase_like = pm.Potential("phase_like", ph_loglike(theta_ph))
    
    idata = do_mcmc(model, n_samples=5000, method='SGD')
    phase_stats = characterize_posterior(idata, plot=plot, plot_title=f"phase_joint_{t0_uncorrected.isoformat()}")
    acq["phase_mcmc"] = phase_stats

    ## RAW FIX HERE...
    
    
    
    ## CLEANUP

    print(phase_stats)
    new_sow_err = phase_stats["sow_offset"]["std"]
    if (new_sow_err < (clock_offset_std + 0.5)) and (new_sow_err > 1e-3):
        gps_t_uncorrected = GpsTime.from_time(t0_uncorrected)

        clock_offset = phase_stats["sow_offset"]["median"]
        clock_offset_std = max(float(phase_stats["sow_offset"]["std"]), 0.5)
    return (clock_offset, clock_offset_std)
