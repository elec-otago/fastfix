"""Von Mises-Fisher distribution on the sphere, ported to PyMC 6 / PyTensor.

See https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution

The distribution is over a 2-vector ``[lon, lat]`` in degrees (longitude
0..360, latitude -90..90).  The PyMC3/Theano implementation has been updated
for the current PyMC (>=5) and PyTensor stack:

* ``theano`` -> ``pytensor``
* ``theano.compile.ops.as_op`` -> a hand-written ``pytensor.graph.Op``
  subclass (``VmfLogpOp``), keeping the same black-box numerical logp
  evaluation as before,
* the PyMC3 ``Continuous`` subclass API is replaced with a ``pm.CustomDist``
  which supplies that black-box ``logp`` (and a ``random`` function for prior
  predictive sampling).
"""
import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from scipy.spatial.transform import Rotation as R

import pymc as pm


def construct_euler_rotation_matrix(alpha, beta, gamma):
    r = R.from_euler('zyx', [alpha, beta, gamma], degrees=True)
    return r.as_matrix()


eps = 1e-4
d2r = np.pi / 180
r2d = 180.0 / np.pi


def cart2dir(cart):
    """
    Converts a direction in cartesian coordinates into declination, inclinations
    Parameters
    ----------
    cart : input list of [x,y,z] or list of lists [[x1,y1,z1],[x2,y2,z2]...]
    Returns
    -------
    direction_array : returns an array of [declination, inclination, intensity]
    Examples
    --------
    >>> cart2dir([0,1,0])
    array([ 90.,   0.,   1.])
    """
    cart = np.array(cart)
    rad = np.pi / 180.  # constant to convert degrees to radians
    if len(cart.shape) > 1:
        Xs, Ys, Zs = cart[:, 0], cart[:, 1], cart[:, 2]
    else:  # single vector
        Xs, Ys, Zs = cart[0], cart[1], cart[2]
    if np.iscomplexobj(Xs):
        Xs = Xs.real
    if np.iscomplexobj(Ys):
        Ys = Ys.real
    if np.iscomplexobj(Zs):
        Zs = Zs.real
    Rs = np.sqrt(Xs**2 + Ys**2 + Zs**2)  # calculate resultant vector length
    # calculate declination taking care of correct quadrants (arctan2) and
    # making modulo 360.
    Decs = (np.arctan2(Ys, Xs) / rad) % 360.
    try:
        # calculate inclination (converting to degrees) #
        Incs = np.arcsin((Zs / Rs)) / rad
    except Exception:
        print('trouble in cart2dir')  # most likely division by zero somewhere
        return np.zeros(3)

    return np.array([Decs, Incs, Rs]).transpose()  # return the directions list


def dir2cart(d):
    """
    Converts a list or array of vector directions in degrees (declination,
    inclination) to an array of the direction in cartesian coordinates (x,y,z)
    Parameters
    ----------
    d : list or array of [dec,inc] or [dec,inc,intensity]
    Returns
    -------
    cart : array of [x,y,z]
    Examples
    --------
    >>> dir2cart([200,40,1])
    array([-0.71984631, -0.26200263,  0.64278761])
    """
    ints = np.ones(len(d)).transpose(
    )  # get an array of ones to plug into dec,inc pairs
    d = np.array(d).astype('float')

    if len(d.shape) > 1:  # array of vectors
        decs, incs = d[:, 0] * d2r, d[:, 1] * d2r
        if d.shape[1] == 3:
            ints = d[:, 2]  # take the given lengths
    else:  # single vector
        decs, incs = np.array(float(d[0])) * d2r, np.array(float(d[1])) * d2r
        if len(d) == 3:
            ints = np.array(d[2])
        else:
            ints = np.array([1.])
    cart = np.array([ints * np.cos(decs) * np.cos(incs), ints *
                     np.sin(decs) * np.cos(incs), ints * np.sin(incs)]).transpose()
    return cart


def angle(D1, D2):
    """
    Calculate the angle between two directions.
    Parameters
    ----------
    D1 : Direction 1 as an array of [declination, inclination] pair or pairs
    D2 : Direction 2 as an array of [declination, inclination] pair or pairs
    Returns
    -------
    angle : angle between the directions as a single-element array
    Examples
    --------
    >>> angle([350.0,10.0],[320.0,20.0])
    array([ 30.59060998])
    """
    D1 = np.array(D1)
    if len(D1.shape) > 1:
        D1 = D1[:, 0:2]  # strip off intensity
    else:
        D1 = D1[:2]
    D2 = np.array(D2)
    if len(D2.shape) > 1:
        D2 = D2[:, 0:2]  # strip off intensity
    else:
        D2 = D2[:2]
    X1 = dir2cart(D1)  # convert to cartesian from polar
    X2 = dir2cart(D2)
    angles = []  # set up a list for angles
    for k in range(X1.shape[0]):  # single vector
        angle = np.arccos(np.dot(X1[k], X2[k])) * r2d  # take the dot product
        angle = angle % 360.
        angles.append(angle)
    return np.array(angles)


def _vmf_logp_numeric(lon_lat, k, x):
    """Numerical Von Mises-Fisher log-density at ``x`` (degrees) for mean
    direction ``lon_lat`` (degrees) and concentration ``k``."""
    x = np.asarray(x, dtype=float)
    lon_lat = np.asarray(lon_lat, dtype=float)
    if x[1] < -90. or x[1] > 90.:
        # raise RuntimeError(f"Value out of range {x}")
        return -1e6  # np.array(-np.inf)
    if k < eps:
        return float(np.log(1. / 4. / np.pi))
    theta = angle(x, lon_lat)[0]
    PdA = k*np.exp(k*np.cos(theta*d2r))/(2*np.pi*(np.exp(k)-np.exp(-k)))
    return float(np.log(PdA))


class VmfLogpOp(Op):
    """Black-box numerical Von Mises-Fisher log-density.

    Inputs: ``lon_lat`` (2-vector), ``k`` (scalar), ``x`` (2-vector).
    Output: scalar log-density.  This replaces the PyMC3-era
    ``theano.compile.ops.as_op`` decorator.
    """

    __props__ = ()

    def make_node(self, lon_lat, k, x):
        lon_lat = pt.as_tensor_variable(lon_lat)
        k = pt.as_tensor_variable(k)
        x = pt.as_tensor_variable(x)
        return Apply(self, [lon_lat, k, x], [pt.scalar()])

    def perform(self, node, inputs, outputs):
        lon_lat, k, x = inputs
        outputs[0][0] = np.array(
            _vmf_logp_numeric(lon_lat, k, x), dtype="float64")


# Module-level singleton so the Op is picklable/shared across the graph.
vmf_logp_op = VmfLogpOp()


def vmf_logp(value, lon_lat, k):
    """Symbolic Von Mises-Fisher log-density.

    Used as the ``logp`` callable of :class:`pm.CustomDist`, so it receives
    the symbolic value tensor plus symbolic parameters and must return a
    symbolic scalar.
    """
    return vmf_logp_op(lon_lat, k, value)


def vmf_random(lon_lat, k, rng=None, size=None):
    """Draw random ``[lon, lat]`` samples from the Von Mises-Fisher
    distribution (used for prior/posterior predictive sampling)."""
    lon_lat = np.asarray(lon_lat, dtype=float)
    k = float(k)
    if rng is None:
        rng = np.random.default_rng()

    alpha = 0.
    beta = np.pi / 2. - lon_lat[1] * d2r
    gamma = lon_lat[0] * d2r

    rotation_matrix = construct_euler_rotation_matrix(alpha, beta, gamma)

    lamda = np.exp(-2*k)

    n = int(np.prod(size)) if size is not None else 1
    out = np.empty((n, 2))
    for i in range(n):
        r1 = rng.random()
        r2 = rng.random()
        colat = 2*np.arcsin(np.sqrt(-np.log(r1*(1-lamda)+lamda)/2/k))
        this_lon = 2*np.pi*r2
        lat = 90-colat*r2d
        lon = this_lon*r2d

        unrotated = dir2cart([lon, lat])[0]
        rotated = np.transpose(np.dot(rotation_matrix, unrotated))
        rotated_dir = cart2dir(rotated)
        out[i] = [rotated_dir[0], rotated_dir[1]]
    return out.reshape(size if size is not None else (2,))


class VMF:
    """Von Mises-Fisher distribution on the sphere.

    This is now a thin factory around :class:`pm.CustomDist` that keeps the
    historical calling convention::

        lonlat = VMF("lonlat", k=0.05, shape=2)

    which registers a free random variable called ``lonlat`` in the current
    PyMC model.  Additional keyword arguments (e.g. ``initval``) are passed
    through to ``pm.CustomDist``.
    """

    def __new__(cls, name, lon_lat=(0.0, 0.0), k=0.0, **kwargs):
        lon_lat = np.asarray(lon_lat, dtype=float)
        return pm.CustomDist(
            name,
            lon_lat,
            k,
            logp=vmf_logp,
            random=vmf_random,
            signature="(2),()->(2)",
            **kwargs,
        )
