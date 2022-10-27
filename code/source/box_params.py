import numpy as np
import astropy.units as au
import astropy.constants as ac
from astropy.coordinates import SkyCoord
from . import cosmology

h = 0.6774
box_length = 4000.
ngrid = 256
lgrid = box_length / ngrid
lx = -2200.
ly = -2000.
lz = -300.
box_corners = np.array([lx, ly, lz])
box_volume = box_length**3
box_model = cosmology.FlatLambdaCDMCosmo()
box_model.get_astropy_cosmo()
