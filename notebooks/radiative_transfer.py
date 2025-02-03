import numpy as np
from scipy.special import sph_harm
from scipy.integrate import quad

def calc_path_segment_curved(r, theta, phi, r_layer, thickness):
    """Calculates the path length through a spherically symmetric layer."""
    return 2 * r_layer * np.sqrt(1 - (r / r_layer) ** 2 * np.sin(theta) ** 2) * thickness

def integrate_opacity_nadir(tau, mu0, mu):
    """Calculates the opacity along nadir/off-nadir viewing paths."""
    return (1 - np.exp(-tau / mu0)) + (1 - np.exp(-tau / mu))

def cloud_thermal_brightness(F0, mu0, mu, w0, P, tau, As):
    """Calculates the cloud brightness for thermal emission."""
    I = (F0 * mu0 * w0 * P / (4 * np.pi)) * (1 - np.exp(-tau / mu0 - tau / mu)) + (As * F0 * mu0 / np.pi) * np.exp(-tau / mu0) * np.exp(-tau / mu)
    return I
