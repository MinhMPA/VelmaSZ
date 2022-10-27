import numpy as np
import h5py as h5
import astropy
from astropy.io import ascii
from astropy import units as au
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d

class CatalogASCII():
    """
    Class to access and manipulate objects in ASCII catalogs
    """

    def __init__(
            self,
            catFile,
            id_col=0,
            ra_col=1,
            dec_col=2,
            use_photoz=True,
            photoz_col=3,
            use_specz=True,
            specz_col=4,
            N200_col=10):
        self.cat = ascii.read(catFile)
        self.id = self.cat[id_col][:].astype(np.uint32)
        self.coord = SkyCoord(
            ra=self.cat[ra_col][:] * au.deg,
            dec=self.cat[dec_col][:] * au.deg)
        if (use_photoz):
            self.photoz = self.cat[photoz_col][:].astype(np.float)
        if (use_specz):
            self.specz = self.cat[specz_col][:].astype(np.float)
        self.N200 = self.cat[N200_col][:].astype(np.float)

    def get_galactic_lonlat(self):
        """
        Compute Galactic longitude and latitute
        """
        lon = np.float64(self.coord.galactic.l.value)
        lat = np.float64(self.coord.galactic.b.value)
        return lon, lat

    def get_M500c(self):
        """
        Compute M500c from N200
        """
        self.M500c = np.exp(0.95) * ((self.N200 / 40.)**1.06) * \
            (1E14 * au.M_sun)  # Rozo et al. 2009
        return True

    def get_M180m(self):
        """
        Compute M180m from N200
        """
        M180m = np.exp(0.48) * ((self.N200 / 20.)**1.13) * \
            1E14  # Rozo et al. 2009
        return M180m

    def get_M200c(self, M180m):
        """
        Interpolate M200c from M180m
        """
        M180m_table = np.loadtxt('M180m_M200c.dat', usecols=0)
        M200c_table = np.loadtxt('M180m_M200c.dat', usecols=1)
        f180mto200c = interp1d(M180m_table, M200c_table, kind='cubic')
        self.M200c = f180mto200c(self.get_M180m()) * au.M_sun
        return True
