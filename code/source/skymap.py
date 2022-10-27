import numpy as np
import healpy as hp
from astropy.io import fits
import astropy.units as au
from functools import partial
from . import cosmological_params
from . import map_params

import matplotlib
import matplotlib.pyplot as plt
from .color.colormap import Planck_cmap, RdBu_cmap

__all__ = ['PlanckMap', 'SquarePatch']


class PlanckMap():
    """
    Class to read and manipulate sky maps in HEALPix format
    """

    def __init__(self, map_array=None, map_coord=None, mapFile=None,
                 maskFile=None, unit=map_params.CMB_unit):
        if (mapFile is not None):
            if (maskFile is None):
                self.map = hp.read_map(mapFile)
            else:
                self.map = hp.ma(hp.read_map(mapFile))
                self.map.mask = np.logical_not(hp.read_map(maskFile))

            with fits.open(mapFile) as hdul:
                self.header = hdul[1].header[:]
            if map_coord is None:
                if (self.header['COORDSYS'] == 'GALACTIC'):
                    self.coord = 'G'
                elif (self.header['COORDSYS'] == 'CELESTIAL'):
                    self.coord = 'C'
                else:
                    self.coord = 'E'
            else:
                self.coord = map_coord
            if (((self.header['TUNIT1'] == 'K_CMB') or (
                    self.header['TUNIT1'] == 'Kcmb')) and (unit == 'muK_CMB')):
                print("Map unit changed from K to muK")
                self.map *= 1E6
        elif (map_array is not None):
            self.map = map_array
            self.coord = map_coord 

        else:
            print("Warning: Initiating an empty map of default NSIDE=2048.")
            self.map = np.zeros((hp.nside2npix(2048),))

        self.nside = hp.get_nside(self.map)
        self.cmap = Planck_cmap

    def add_mask(self, mask2):
        """
        Multiply existing mask with another mask, same shape
        """

        self.map.mask += mask2

        return True

    def combine_mask(self, maskFile):
        """
        Combine existing mask with another given mask, already stored in healpix format
        """

        mask2 = np.logical_not(hp.read_map(maskFile))
        self.add_mask(mask2)

        return True

    def rotate(self,
               rot=None, coord=None, alms=False, lmax=None):
        """
        Rotate the map arbitrarily, can be used to change between coordinate systems too
        """

        rot = hp.Rotator(rot=rot, coord=coord)
        if (alms):
            if (lmax is None):
                lmax = 3 * hp.get_nside(self.map) - 1
            self.map = rot.rotate_map_alms(self.map, lmax=lmax)
        else:
            self.map = rot.rotate_map_pixel(self.map)
        self.coord = coord[1]

        return True

    def get_Cartesian_square_patch(
            self,
            ra,
            dec,
            patch_size=map_params.patch_size,
            n_pixels=map_params.n_pixels,
            coord='C',
            alms=False,
            lmax=None,
            return_proj=True):
        """
        Cut out a square patch of the sky at given location using Cartesian projection
        This operation is done in Celestial coordinate system
        """

        # check if the map is already in Celestial coordinates
        if (self.coord != coord):
            self.rotate(coord=[self.coord, coord], alms=alms, lmax=lmax)
        # get to the cluster and project
        lonra = [(ra - 0.5 * patch_size).to_value(au.deg), (ra + 0.5 * patch_size).to_value(au.deg)]
        latra = [(dec - 0.5 * patch_size).to_value(au.deg), (dec + 0.5 * patch_size).to_value(au.deg)]
        Cartesian_projector = hp.projector.CartesianProj(
            lonra=lonra, latra=latra,
            coord=self.coord,
            xsize=n_pixels, ysize=n_pixels)
        patch = Cartesian_projector.projmap(
            self.map, vec2pix_func=partial(
                hp.vec2pix, self.nside), coord=coord)

        if (return_proj):
            return patch, Cartesian_projector
        else:
            return patch


class SquarePatch():
    """
    Class to handle rectangular sky patches
    """

    def __init__(
            self,
            patch_array=None,
            # enforce patch size to be given in degrees
            size_x=map_params.patch_size.to_value(au.deg),
            # enforce patch size to be given in degrees
            size_y=map_params.patch_size.to_value(au.deg),
            unit='muK_CMB',
            cmap=Planck_cmap):
        # make a copy to avoid modifying the original array
        self.map = np.copy(patch_array)
        # add the internal unit back in
        self.size = np.array([size_x, size_y]) * au.deg
        self.pix_size = self.size / self.map.shape 
        self.unit = unit
        self.cmap = cmap

    def save(self, filename):
        """
        Save patches
        """

        np.save(filename, self.map)

        return True

    def stack(self, map2, weight=1.):
        """
        Stack patches
        """

        self.map += weight * map2

        return True

    def subtract(self, map2, weight=1.):
        """
        Subtract patches
        """

        self.map -= weight * map2

        return True

    def construct_2d_configuration_grid(self, norm=False):
        """
        Construct a 2D radial grid in configuration space
        """

        x, y = np.indices((self.map.shape))
        x = (x-self.map.shape[0]//2.)
        y = (y-self.map.shape[0]//2.)
        r = np.sqrt(x**2 + y**2)

        if norm:
            x /= self.map.shape[0]
            y /= self.map.shape[1]
            r /= np.sqrt(np.prod(self.map.shape))

        return x, y, r

    def construct_2d_multipole_grid(self, fftcenter=True):
        """
        Construct a 2D radial grid in multipole space
        """

        x, y, r = self.construct_2d_configuration_grid()
        lx = x * (2. * np.pi) / self.size[0].to_value(au.rad)
        ly = y * (2. * np.pi) / self.size[0].to_value(au.rad)
        lr = np.sqrt(lx**2+ly**2)
        if not fftcenter:
            lr = np.fft.ifftshift(lr)

        return lx, ly, lr

    def construct_2d_fourier_grid(self, fftcenter=False):
        """
        Construct a 2D radial grid in Fourier space
        """

        lx = (2. * np.pi) * np.fft.fftfreq(self.map.shape[0])
        ly = (2. * np.pi) * np.fft.fftfreq(self.map.shape[1])
        lr = np.sqrt(lx[:, None]**2 + ly[None, :]**2)
        if fftcenter:
            lr = np.fft.fftshift(lr)

        return lx, ly, lr

    def construct_2d_multipole_kernel(self, kernel, fftcenter=True):  
        """
        Construct a 2D symmetric grid kernel from input multipole-space kernel
        """

        # Prepare a 2D multipole grid
        lx, ly, lr = self.construct_2d_multipole_grid(fftcenter=fftcenter)

        # Construct the 2D multipole kernel 
        # prepare an expanded kernel (of zeros)
        kernel_expanded = np.zeros(int(lr.max()) + 1)
        # fill the expanded Cl_exp with the original angular power spectrum Cl
        if (kernel_expanded.size>=kernel.size):
            kernel_expanded[0:(kernel.size)] = kernel 
        else:
            kernel_expanded = kernel[0:(kernel_expanded.size)]
        # fill the 2D grid Cl_2d with Cl_exp, using NGP interpolation
        kernel_2d = kernel_expanded[lr.astype(int)]

        return kernel_2d
        
    def convolve_gaussian(
            self, beam_size_fwhm=map_params.beam_size_fwhm):
        """
        Convolve a map with a gaussian beam.

        Parameters
        ----------
        beam_size_fwhm: float
                        The FWHM of the Gaussian beam, as an astropy quantity
        """

        if (self.map is None):
            raise ValueError('No input map to convolve.')
        else:
            npix = np.prod(self.map.shape)
        # ensure that both lr and beam_sigma are in pixel unit
        beam_sigma = (beam_size_fwhm.to_value(au.arcmin)/(2.*np.sqrt(2*np.log(2)))) \
                / self.pix_size[0].to_value(au.arcmin)
        lx, ly, lr = self.construct_2d_fourier_grid(fftcenter=False)
        map_fft = np.fft.fft2(np.fft.ifftshift(self.map)) \
                * np.exp(-0.5 * lr**2 * beam_sigma**2)
        self.map = np.fft.fftshift(np.fft.ifft2(map_fft)).real

        return True

    def convolve(self, input_beam):
        """
        Convolve a map with an input beam.

        Parameters
        ----------
        input_beam: 1D array, float
                    The input beam in multipole space.
        """

        if (self.map is None):
            raise ValueError('No input map to convolve.')
        else:
            npix = np.prod(self.map.shape)
        beam_2d = self.construct_2d_multipole_kernel(input_beam, fftcenter=False)
        map_fft = np.fft.fft2(np.fft.ifftshift(self.map)) * beam_2d
        self.map = np.fft.fftshift(np.fft.ifft2(map_fft)).real

        return True

    def average_radial(self, image, return_k=False):
        """
        Compute the azimuthally-averaged radial profile of a given map.

        Parameters
        ----------
	image: 2D float or complex array
	  Input 2D profile, must be of same size and shape with the patch object.
        return_k: bool, optional
        If set to True, the provided image is assumed to be a power spectrum
        and the x-coordinate of the returned data is converted to the
        two-dimensional spatial frequency k. Default: None

        Returns
        -------
        radial_distance: float array
        r-coordinate. Either the radial separation
        from the image center or spatial frequency k, depending on the
        value of the variable return_k.
        radial_profile: float or complex array
        azimuthally averaged radial profile.
        """

        x, y, r = self.construct_2d_configuration_grid()

        if return_k is True:
            r_scaled = r / np.sqrt(self.map.shape[0]*self.map.shape[1])
            k = (360./self.pix_size[0].to_value(au.deg)) * r_scaled
        else:
            k = np.copy(r)

        r_int = r.astype(np.int)

        weight = np.bincount(r_int.ravel())
        radial_distance = np.bincount(r_int.ravel(), k.ravel()) / weight
        radialprofile_real = np.bincount(
            r_int.ravel(), np.real(image.ravel())) / weight
        radialprofile_imag = np.bincount(
            r_int.ravel(), np.imag(image.ravel())) / weight
        radial_profile = radialprofile_real + radialprofile_imag * 1j

        return radial_distance, radial_profile

    def get_autopower(self, return_k=False):
        """
        Compute the azimuthally-averaged power spectrum of a given map.

        Parameters
        ----------
        image: 2D float array
                Input image
        return_k: bool, optional
                If set to True, the provided image is assumed to be a power spectrum
                and the x-coordinate of the returned data is converted to the
                two-dimensional spatial frequency k. Default: None

        Returns
        -------
        k: float array
                x-coordinate of the radial profile. Either the radial separation
                from the image center or spatial frequency k, depending on the
                value of the variable return_k.
        Pk: float or complex array
                azimuthally-averaged power spectrum
        """

        npix = np.prod(self.map.shape)
        Fk = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.map))) / npix
        ps = (np.absolute((Fk))**2)

        k, Pk = self.average_radial(ps, return_k=return_k)

        return k, Pk

    def construct_noise_template(self, R, profile):
        """
        Construct a 2D template from an azimuthally-averaged radial profile.
        Useful for the computing 2D window function from some given profile.

        Parameters
        ----------
        R: float array
                Radial coordinate of the azimuthally-averaged radial profile.
                Radial separation from the image center in usints of pixels.
        window: float array
                Azimuthally averaged radial profile to be used to construct the
                2D map

        Returns
        -------
        image: 2D float or complex array
                2D map created from the provided azimuthally-averaged radial
                profile.
        """

        x, y, r = self.construct_2d_configuration_grid() 
        r = r.ravel()

        image = np.interp(r, R, profile)
        image = image.reshape(self.map.shape[0], self.map.shape[1])

        return image

    def apply_matched_filter(
            self,
            signal_profile,
            noise_angular_power=None,
            l0_taper=300.):
        """
        Apply a matched filter

        Parameters
        ----------
        signal_profile: 2D float
                        2D real-space profile of the signal

        noise_angular_power: 1D float
                                1D array of the noise angular power spectrum, i.e. the azimuthally-averaged noise power spectrum. Default: None, computed directly from the input map.

        l0_taper: float
                    The multipole ell at which tapering starts to take effect 

        Returns
        -------

        filtered_map: 2D float 
                        2D filtered map

        Fourier_filter_template: 2D float
                                2D fourier-space template of the filter, for inspection

        noise: float
                The noise of the estimate, i.e. standard deviation of the filtered map

        """

        # Construct the filter
        # 1 - signal template
        npix = np.prod(self.map.shape)
        Fourier_signal_template = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(signal_profile)))/npix
        Fourier_signal_template = self.taper_lowell_exp(Fourier_signal_template, l0_taper=l0_taper, fftcenter=True)
        # 2 - noise template
        if noise_angular_power is not None:
            Fourier_noise_template = self.construct_2d_multipole_kernel(noise_angular_power) / np.prod(self.size.to_value(au.rad))
        else:
            k, noise_ps = self.get_autopower(self.map)
            Fourier_noise_template = self.construct_noise_template(k, noise_ps)

        # 3 - construct filter
        Fourier_filter_template = (Fourier_signal_template/Fourier_noise_template) \
                / np.sum((Fourier_signal_template**2)/Fourier_noise_template)
        # 4 - apply filter through multiplication in Fourier space
        filtered_map = Fourier_filter_template.conj() * \
            np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.map)))/npix
        # 5 - FFT back, keeping only the real part
        filtered_map = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(filtered_map))).real*npix
        # centering the 0 frequency of the filter, for easy visualization
        sigma = np.sqrt(np.real(np.sum(Fourier_filter_template**2*Fourier_noise_template))) # return also the std of the estimate

        return filtered_map, Fourier_filter_template, sigma

    def apply_AP_filter():
        """
        Apply an aperture photometry filter - to be implemented later, right now do nothing
        """

        filtered_map = self.map

        return filtered_map

    def simulate_CMB_patch(self, Cl, seed=1):
        """
        Simulate a CMB patch given the angular power spectrum D_ell
        """

        # Construct the 2D Cl spectrum
        Cl_2d = self.construct_2d_multipole_kernel(Cl, fftcenter=False)

        # Simulate a Fourier-space patch given the 2D CMB power spectrum above
        np.random.seed(seed)
        patch_FFT_2d = np.sqrt(Cl_2d/np.prod(self.pix_size.to_value(au.rad))) \
                * np.fft.fft2(np.random.normal(0, 1, (self.map.shape[0], self.map.shape[1])))

        # FFT back to real-space
        self.map = np.fft.fftshift(np.fft.ifft2(patch_FFT_2d)).real

        return True

    def get_beta_profile(self, SZ_beta, SZ_theta_core, SZ_theta_cutoff=None):
        """
        Return the map of the beta function, length is in arcmin
        """

        x, y, r = self.construct_2d_configuration_grid()
        # below we ensure that both radial distances are in pixel unit
        theta_core = SZ_theta_core.to_value(au.arcmin) / self.pix_size[0].to_value(au.arcmin)
        beta = ( 1. + (r/theta_core)**2 )**( 0.5 * (1.-3.*SZ_beta) )
        if SZ_theta_cutoff is not None:
            theta_cutoff = SZ_theta_cutoff.to_value(au.arcmin) / self.pix_size[0].to_value(au.arcmin)
            beta[r>theta_cutoff] = 0.

        return beta

    def simulate_SZ_source_beta(
            self,
            SZ_beta=map_params.SZ_beta,
            SZ_theta_core=map_params.SZ_theta_core,
            central_SZ_amplitude=map_params.central_SZ_amplitude,
            SZ_theta_cutoff=None):
        """
        Simulate a patch with a SZ source at the center
        """

        self.map = central_SZ_amplitude \
                * self.get_beta_profile(SZ_beta, SZ_theta_core, SZ_theta_cutoff=SZ_theta_cutoff)

        return True

    def apodize_cos(self):
        """
        Apodize the 2D map to avoid edges effects"
        """

        x, y, r = self.construct_2d_configuration_grid(norm=True)
        window_map = np.cos(x*np.pi) * np.cos(y*np.pi)

        # apodize
        self.map *= window_map

        # return the window map
        return window_map

    def taper_lowell_exp(self, template, l0_taper=300, fftcenter=True):
        """
        Taper a 2D template in Fourier space to exponentially suppress low-multipole contribution
        """

        lx, ly, lr = self.construct_2d_multipole_grid(fftcenter=fftcenter)
        tapered_template = np.copy(template)
        tapered_template[np.where(lr<=l0_taper)] *= np.exp(lr[np.where(lr<=l0_taper)]-l0_taper)
        
        return tapered_template

    def compute_2d_spectrum(self, delta_l=50.,lmax=cosmological_params.camb_lmax_out, fftcenter=True):
        """
        Compute the power spectrum from a 2d patch
        """

        lx, ly, lr = self.construct_2d_multipole_grid(fftcenter=fftcenter)
        
        # Prepare the array for Cl
        npix = np.prod(self.map.shape)
        N_bins = int(lmax/delta_l)
        lbin = np.arange(N_bins)
        bin_avg_Cl = np.zeros(N_bins)
        
        # Compute a 2D grid of Cl from the 2d map
        Fourier_map = np.fft.fftshift(np.fft.fft2(self.map)) / npix
        Cl_2D = np.abs(Fourier_map)**2

        # Average in radial bin
        i = 0
        while (i < N_bins):
            lbin[i] = (i + 0.5) * delta_l
            inds_in_bin = ((lr >= (i* delta_l)) * (lr < ((i+1)* delta_l))).nonzero()
            bin_avg_Cl[i] = np.mean(Cl_2D[inds_in_bin])
            i = i + 1

        # Remove the bin(s) with NaN, if any
        bin_avg_Cl_new = bin_avg_Cl[~np.isnan(bin_avg_Cl)]
        lbin_new = lbin[~np.isnan(bin_avg_Cl)]

        # Scale back to full-sky case, to facilitate comparison
        bin_avg_Cl_new *= np.prod(self.size.to_value(au.rad))

        return(lbin_new,bin_avg_Cl_new)

    def plot_patch(
            self,
            size_x=map_params.patch_size.to_value(au.deg),
            size_y=map_params.patch_size.to_value(au.deg),
            cmap_min=map_params.cmap_min,
            cmap_max=map_params.cmap_max):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        print("map mean:", np.mean(self.map), "map rms:", np.std(self.map))
        plt.gcf().set_size_inches(5, 5)
        im = plt.imshow(
            self.map,
            interpolation='None',
            origin='lower',
            cmap=self.cmap)
        im.set_clim(cmap_min, cmap_max)
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = plt.colorbar(im, cax=cax)
        im.set_extent([-size_x/2., size_x/2., -size_y/2., size_y/2.])
        plt.ylabel('Angle $[^\\circ]$')
        plt.xlabel('Angle $[^\\circ]$')
        cbar.set_label(r'Temperture [$\mu K$]', rotation=270, labelpad=15)

        plt.show()

        return True

