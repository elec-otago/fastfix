import pymc3 as pm
import numpy as np
import theano.tensor as tt
from theano.compile.ops import as_op

from scipy.spatial.transform import Rotation as R

def construct_euler_rotation_matrix(alpha, beta, gamma):
    r = R.from_euler('zyx', [alpha, beta, gamma], degrees=True)
    return r.as_matrix()

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
    >>> pmag.dir2cart([200,40,1])
    array([-0.71984631, -0.26200263,  0.64278761])
    """
    ints = np.ones(len(d)).transpose(
    )  # get an array of ones to plug into dec,inc pairs
    d = np.array(d).astype('float')
    rad = np.pi/180.
    if len(d.shape) > 1:  # array of vectors
        decs, incs = d[:, 0] * rad, d[:, 1] * rad
        if d.shape[1] == 3:
            ints = d[:, 2]  # take the given lengths
    else:  # single vector
        decs, incs = np.array(float(d[0])) * rad, np.array(float(d[1])) * rad
        if len(d) == 3:
            ints = np.array(d[2])
        else:
            ints = np.array([1.])
    cart = np.array([ints * np.cos(decs) * np.cos(incs), ints *
                     np.sin(decs) * np.cos(incs), ints * np.sin(incs)])
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
    >>> pmag.angle([350.0,10.0],[320.0,20.0])
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
        angle = np.arccos(np.dot(X1[k], X2[k])) * \
            180. / np.pi  # take the dot product
        angle = angle % 360.
        angles.append(angle)
    return np.array(angles)

eps = 1e-4
d2r = 180 / np.pi

@as_op(itypes=[tt.dvector, tt.dscalar, tt.dvector], otypes=[tt.dscalar])
def vmf_logp(lon_lat, k, x):

    if x[1] < -90. or x[1] > 90.:
        #raise pm.ZeroProbability
        return np.array(-np.inf)
    if k < eps:
        return np.log(1. / 4. / np.pi)
    theta = angle(x, lon_lat)[0]
    PdA = k*np.exp(k*np.cos(theta*d2r))/(2*np.pi*(np.exp(k)-np.exp(-k)))
    logp = np.log(PdA)
    return np.array(logp)


class VMF(pm.Continuous):
    ''' 
        https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
    '''
    def __init__(self, lon_lat=[0.0,0.0], k=None,
             *args, **kwargs):
        super(VMF, self).__init__(*args, **kwargs)
        
        self._k = tt.as_tensor_variable(pm.floatX(k))
        self._lon_lat = tt.as_tensor(np.array(lon_lat))

    def logp(self, value):
        lon_lat = self._lon_lat
        k = self._k
        value = tt.as_tensor_variable(value)
        return vmf_logp(lon_lat, k, value)


    def _random(self, lon_lat, k, size = None):

        alpha = 0.
        beta = np.pi / 2. - lon_lat[1] * d2r
        gamma = lon_lat[0] * d2r

        rotation_matrix = construct_euler_rotation_matrix(alpha, beta, gamma)

        lamda = np.exp(-2*k)

        r1 = np.random.random()
        r2 = np.random.random()
        colat = 2*np.arcsin(np.sqrt(-np.log(r1*(1-lamda)+lamda)/2/k))
        this_lon = 2*np.pi*r2
        lat = 90-colat*r2d
        lon = this_lon*r2d

        unrotated = dir2cart([lon, lat])[0]
        rotated = np.transpose(np.dot(rotation_matrix, unrotated))
        rotated_dir = cart2dir(rotated)
        return np.array([rotated_dir[0], rotated_dir[1]])

    def random(self, point=None, size=None):
    
        lon_lat, k = draw_values([self._lon_lat, self._k], point=point, size=size)
        return generate_samples(self._random, lon_lat, k,
                                dist_shape=self.shape,
                               size=size)
