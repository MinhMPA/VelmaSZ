import numpy as np
from scipy.interpolate import interpn
import astropy.units as au
from astropy import cosmology
from astropy.cosmology import Planck15
from . import box_params
from . import cosmology

class Box():
    """
    Base class for a cubic volume constrained by observed galaxies

    Parameters
    ----------
    box_cosmo : astropy Cosmology object
        the assumed cosmological model
    h : float
        h = H0/100
    box_length : float
        the length of a side of the cubic box
    box_corners: float, array_like
        the position of the box corners w.r.t. the observer
    ngrid : int
        the number of grid cells per side
    """

    def __init__(self, box_cosmo=box_params.box_model.cosmo, h=box_params.h, box_length=box_params.box_length, box_corners=box_params.box_corners, ngrid=box_params.ngrid):
        if box_cosmo is None:
            self.box_cosmo = Planck15
            self.h = Planck15.h
        else:
            self.box_cosmo = box_cosmo
            self.h = h
        self.box_length = (box_length/self.h) * au.Mpc
        self.BoxVolume = np.prod(self.box_length.value) * au.Mpc**3
        self.box_corners = (box_corners/self.h) * au.Mpc
        self.ngrid = ngrid
        self.lgrid = self.box_length / self.ngrid

    def configure(self, box_length=None, box_corners=None, ngrid=None):
        """
        Adjust box configuration
        """
        if box_length is not None:
            self.box_length = (box_length/self.h) * au.Mpc
            self.BoxVolume = np.prod(self.box_length.value) * au.Mpc**3
        if box_corners is not None:
            self.box_corners = (box_corners/self.h) * au.Mpc
        if ngrid is not None:
            self.ngrid = ngrid
        self.lgrid = self.box_length / self.ngrid
        return True
    
    def copy_velocity_grid(self, vfield):
        """
        Load 3D grid of velocity
        """
        self.vel_grid = np.copy(vfield)
        return True

    def load_object_onto_grid(self, objxyz):
        """
        Load objects into constrained volume 
        """
        self.obj_grid_idx = objxyz / self.lgrid.to_value(au.Mpc)
        return True

    def get_object_velocity(self):
        """
        Interpolate from vfield grid points to object positions to obtain object velocity 
        """
        grid_points = (np.arange(self.ngrid), np.arange(self.ngrid), np.arange(self.ngrid))
        self.obj_vxyz = interpn(grid_points, self.vel_grid, self.obj_grid_idx)
        return True

    def project_vlos(self):
        """
        Project velocity onto the LOS of the observer to get the radial component 
        """
        obj_radial_distance = np.sqrt(np.einsum('...i,...i->...', self.obj_grid_idx, self.obj_grid_idx))
        vlos = np.einsum('...i,...i->...', self.obj_vxyz, self.obj_grid_idx) / obj_radial_distance
        return vlos
