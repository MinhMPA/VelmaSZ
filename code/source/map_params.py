import numpy as np
import astropy.units as au
import astropy.constants as ac
from astropy.coordinates import Angle
from .cosmological_params import Tcmb0

# default params for maps and patches
Planck_default_nside = 2048
frequency = 143e9 * au.Hz
CMB_unit = 'muK_CMB'
patch_size = (14.66/4.) * au.deg
n_pixels = int(512/4)
beam_size_fwhm = Angle(7.27, au.arcmin)
lowell_taper_pivot = 500.

## default params for plots
cmap_min = -400.  # minimum for color bar
cmap_max = 400.   # maximum for color bar

# default params for models
x = (ac.h*frequency.to(au.Hz)) / (ac.k_B*Tcmb0.to(au.Kelvin))
f_freq = (x*(np.exp(x)+1)/(np.exp(x)-1) - 4) * (Tcmb0.to_value(au.microKelvin))
y0 = 1e-4 #Central Comptonization parameter (model  amplitude)
central_SZ_amplitude = (y0 * f_freq).value
mean_Poisson_amplitude = 200.
mean_exp_amplitude = 1000.
SZ_beta = 0.86
SZ_theta_core = 2.0 * au.arcmin
white_noise_level = 10.
