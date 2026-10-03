
import concurrent.futures

from .vmf import VMF
from .util import gaussian_llh
from .angle import from_dms
from .location import Location
from .gps_time import GpsTime
from .fastfix import doppler_fmap, Satellite, phase_fmap

import datetime
import logging

import matplotlib.pyplot as plt
import numpy as np

import pymc as pm
import pytensor
import pytensor.tensor as pt
import arviz as az

logger = logging.getLogger(__name__)
# Add a null handler so logs can go somewhere
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)


class PhaseLogLike:
    """Log-likelihood of the codephase (phase) measurements.

    Ported from the PyMC3/Theano ``tt.Op`` implementation to a plain callable
    wrapped with :func:`pytensor.wrap_py` so that it can be used as a
    ``pm.Potential`` term in modern PyMC models.
    """

    def __init__(self, acq, t0, ephs):
        self.svs = acq["sv"]
        self.data = np.array(acq["codephase"])  # Measured data
        self.xmax = np.array(acq["x_max"])
        self.sats = [
            Satellite(sv, ephs.get_ephemeris(prn=sv, gps_t=t0)) for sv in self.svs
        ]

    def __call__(self, theta):
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
                if meas_xm > 12:
                    sigma = 0.002  # Should be close to one sample which is 1/8k
                if pred_xm < 8.0 or meas_xm < 8.0:
                    # These should be ignored, therefore a huge variance.
                    sigma = 2.0

                logp += gaussian_llh(x=meas_ph, mu=pred_ph, sigma=sigma)

        except Exception as e:
            logger.info(f"Exception {e}: param: {theta}")
            logp = -999.0e99

        return np.array(logp)

    def as_tensor(self, theta):
        """Return a symbolic scalar log-likelihood for the given theta vector."""

        @pytensor.wrap_py(itypes=[pt.dvector], otypes=[pt.dscalar])
        def _logp(th):
            return np.asarray(self(th), dtype="float64")

        return _logp(theta)


class DopplerLogLike:
    """Log-likelihood of the doppler-shift measurements."""

    def __init__(self, acq, t0, ephs):
        svs = acq["sv"]
        self.xmax = acq["x_max"]
        self.sats = [Satellite(sv, ephs.get_ephemeris(
            prn=sv, gps_t=t0)) for sv in svs]

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

    def __call__(self, theta):
        lat, lon, delta_f = theta

        # logger.info(f"param: {theta}, N={self.x.shape[0]}")
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
            # if elevation < 5 and xmax < 9:
            ##sigma = 300.0 * (10.0 - xmax)

            logp += gaussian_llh(x=meas, mu=sim, sigma=sigma)

        return np.array(logp)

    def as_tensor(self, theta):
        """Return a symbolic scalar log-likelihood for the given theta vector."""

        @pytensor.wrap_py(itypes=[pt.dvector], otypes=[pt.dscalar])
        def _logp(th):
            return np.asarray(self(th), dtype="float64")

        return _logp(theta)


def sub_stats(stat1, stat2, key):
    p1 = dict(stat1[key])
    p2 = dict(stat2[key])

    p1["lonlat[0]"] = p2["lonlat[0]"]
    return p1


def characterize_posterior(model_name, trace, plot=False, plot_title="trace"):
    """Summarize the posterior of a sampled model.

    Parameters
    ----------
    model_name: str
        The name of the model (used to strip the model prefix from variable
        names so results match the historical output format).
    trace: InferenceData
        The sampling result.
    plot: bool
        Whether to save trace/pair plots.
    plot_title: str
        Base name for the plot files.
    """
    stats = pm.stats.summary(trace, round_to="auto")
    print(stats.to_string())
    if plot:
        az.plot_trace(trace)
        plt.savefig(f"{plot_title}_chain_histogram.pdf")
        plt.close()

        az.plot_pair(trace, triangle="lower", marginal=True)
        plt.savefig(f"{plot_title}_joint_lat_lon.pdf")
        plt.close()

    rhat = stats["r_hat"].to_dict()

    post = trace["posterior"]

    def find_samples(name):
        """Return a flat 1-D array of samples for one summary row.

        Modern PyMC stores a free variable either as a single posterior node
        with a trailing element dimension (e.g. ``lonlat`` with shape
        ``(chain, draw, 2)``) or, when it was declared element by element
        (e.g. ``lonlat[0]``, ``lonlat[1]``), as separate scalar nodes.  The
        summary index may also carry the model-name prefix
        (``doppler::lonlat[0]``).  Try each combination in turn.
        """
        candidates = [name]
        for sep in ("::", "_"):
            prefix = f"{model_name}{sep}"
            if name.startswith(prefix):
                candidates.append(name[len(prefix):])
        for cand in list(candidates):
            if "[" in cand and cand.endswith("]"):
                base, _, _rest = cand.partition("[")
                candidates.append(base)

        idx = None
        if "[" in name and name.endswith("]"):
            _, _, rest = name.partition("[")
            idx = int(rest[:-1])

        for cand in candidates:
            if cand not in post:
                continue
            v = post[cand].values
            flat = v.reshape(-1, *v.shape[2:])
            if idx is not None and flat.ndim > 1:
                return np.asarray(flat[:, idx]).reshape(-1)
            return np.asarray(flat).reshape(-1)

        raise KeyError(f"Could not find posterior samples for {name!r} "
                       f"(tried {candidates}; posterior has {list(post.data_vars)})")

    def wrap180(x):
        return ((x + 180) % 360) - 180

    def clean_key(k):
        # Strip the model-name prefix ("doppler::", "phase::") added by modern
        # PyMC named-models so keys match the historical output format.
        for sep in ("::", "_"):
            prefix = f"{model_name}{sep}"
            if k.startswith(prefix):
                return k[len(prefix):]
        return k

    ret_swap = {}
    for key in stats.index:
        k = clean_key(key)
        samples = find_samples(key)
        lon_samples = wrap180(samples) if k == "lonlat[0]" else samples
        print(f"{key} -> {k}")
        ret_swap[k] = {
            'r_hat': rhat[key],
            'std': float(np.std(samples)),
            '5%': float(np.percentile(lon_samples, 5)),
            'median': float(np.percentile(lon_samples, 50)),
            '95%': float(np.percentile(lon_samples, 95)),
        }

    return ret_swap


def do_mcmc(n_samples=3000, method='NUTS'):
    n_tune = n_samples
    n_chains = 4
    if method == 'NUTS':
        idata = pm.sample(draws=n_samples, tune=n_tune, chains=n_chains,
                          init='jitter+adapt_diag',
                          return_inferencedata=True, discard_tuned_samples=True)
    else:
        idata = pm.sample_smc(draws=n_samples)

    return idata


def doppler_model(t0_uncorrected, acq, gps_t, ephs, plot):

    do_loglike = DopplerLogLike(acq, gps_t, ephs)

    # DO THE DOPPLER FiX
    with pm.Model("doppler") as model:

        lonlat = VMF("lonlat", k=0.05, shape=2,
                     initval=np.array([0.0, 0.0]))
        lon = lonlat[0]
        lat = lonlat[1]

        delta_f = pm.Normal("delta_f_khz", mu=0.0,
                            sigma=1.0, initval=0.0) * 1000

        theta_do = pt.as_tensor_variable([lat, lon, delta_f])
        like = pm.Potential("like", do_loglike.as_tensor(theta_do))

        # The likelihood is a black-box Python function, so gradients are not
        # available and NUTS cannot be used.  Metropolis sampling is used
        # instead (it only needs log-probability evaluations).
        step = pm.Metropolis()
        idata = pm.sample(draws=1000, tune=1000, chains=4, step=step,
                          progressbar=False,
                          return_inferencedata=True, discard_tuned_samples=True)

    doppler_stats = characterize_posterior(
        "doppler", idata, plot=plot, plot_title=f"doppler_joint_{t0_uncorrected.isoformat()}")
    return doppler_stats


def run_doppler_model(*args, **kwargs):
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(doppler_model, *args, **kwargs)
        return future.result()


def phase_model(t0_uncorrected, doppler_stats, clock_offset_std, acq, gps_t, ephs, plot):
    ph_loglike = PhaseLogLike(acq, gps_t, ephs)
    with pm.Model("phase") as model:

        # 1 / kappa = sigma^2 => kappa = 1 / sigma^2 (assume sigma = np.degrees(3))
        lon_start = doppler_stats['lonlat[0]']['median']
        lat_start = doppler_stats['lonlat[1]']['median']

        print(f"lonlat_start = {[lon_start, lat_start]}")

        lon = pm.Normal('lonlat[0]', mu=lon_start,
                        sigma=max(doppler_stats['lonlat[0]']['std'], 1e-3))
        lat = pm.Normal('lonlat[1]', mu=lat_start,
                        sigma=max(doppler_stats['lonlat[1]']['std'], 1e-3))

        alt = pm.HalfNormal("alt", sigma=1.0) * 1000
        if False:
            offset = pm.Uniform("phase_offset", lower=0, upper=1)
        else:
            offset = (pm.VonMises("phase_offset", mu=0,
                      kappa=0.01) + np.pi) / (2*np.pi)

        clk_err = (clock_offset_std+0.5)
        # Add half a second as the rtc is only accurate to 1 second.
        sow = pm.Uniform("sow_offset",  lower=-clk_err,
                         upper=clk_err) + gps_t.sow()

        theta_ph = pt.as_tensor_variable([lat, lon, alt, offset, sow])

        phase_like = pm.Potential("phase_like", ph_loglike.as_tensor(theta_ph))

        n_draws = 3000
        if True:
            n_tune = n_draws
            # Black-box likelihood => no gradients => use Metropolis.
            step = pm.Metropolis()
            idata = pm.sample(draws=n_draws, tune=n_tune, chains=4, step=step,
                              progressbar=False,
                              return_inferencedata=True, discard_tuned_samples=True)
        else:
            idata = pm.sample_smc(draws=n_draws)

    phase_stats = characterize_posterior(
        "phase", idata, plot=plot, plot_title=f"phase_joint_{t0_uncorrected.isoformat()}")
    return phase_stats


def run_phase_model(*args, **kwargs):
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(phase_model, *args, **kwargs)
        return future.result()


def process_mcmc(acq, start_date, brdc_proxy, local_clock_offset, plot=False):

    clock_offset, clock_offset_std = local_clock_offset

    rtc_offset = acq["rtc"]

    t0_uncorrected = start_date + datetime.timedelta(seconds=rtc_offset)
    t0 = start_date + datetime.timedelta(seconds=rtc_offset + clock_offset)

    print("##########################################################")
    print("")
    print(
        f"FastFix MCMC processing: t0={t0.isoformat()} offset={local_clock_offset}")
    print("")
    acq["t0"] = t0.isoformat()
    acq["local_t0"] = t0_uncorrected.isoformat()
    acq["local_clock_offset"] = local_clock_offset

    ephs = brdc_proxy.get_ephemerides(t0)

    gps_t = GpsTime.from_time(t0)
    acq["gps_t"] = gps_t.to_dict()

    # DO THE DOPPLER FiX
    doppler_stats = run_doppler_model(t0_uncorrected, acq, gps_t, ephs, plot)

    acq["doppler_mcmc"] = doppler_stats
    print(doppler_stats)

    # NOW DO THE PHASE FiX

    phase_stats = run_phase_model(
        t0_uncorrected, doppler_stats, clock_offset_std, acq, gps_t, ephs, plot)
    acq["phase_mcmc"] = phase_stats

    # RAW FIX HERE...

    # CLEANUP

    print(phase_stats)
    new_sow_err = phase_stats["sow_offset"]["std"]
    new_sow_rhat = phase_stats["sow_offset"]["r_hat"]
    if (new_sow_err < (clock_offset_std + 0.5)) and (new_sow_rhat < 1.02):
        gps_t_uncorrected = GpsTime.from_time(t0_uncorrected)

        clock_offset = clock_offset + phase_stats["sow_offset"]["median"]
        clock_offset_std = max(float(phase_stats["sow_offset"]["std"]), 0.5)
    return (clock_offset, clock_offset_std)
