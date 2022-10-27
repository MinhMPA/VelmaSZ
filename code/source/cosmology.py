import numpy as np
import camb
from astropy.cosmology import FlatLambdaCDM
import astropy.units as au
from .cosmological_params import *

class FlatLambdaCDMCosmo():
    """
    Flat LambdaCDM Cosmology class
    """

    def __init__(self,
                 H0=H0,
                 Tcmb0=Tcmb0,
                 mnu=[mnu,0.,0.],
                 Om0=(omch2+ombh2)/(H0/100.)**2,
                 Ob0=ombh2/(H0/100.)**2):

        self.H0 = H0
        self.Tcmb0 = Tcmb0
        self.mnu = mnu
        self.Om0 = Om0
        self.Ob0 = Ob0

    def get_astropy_cosmo(self):
        self.cosmo = FlatLambdaCDM(H0=self.H0,
                                   Om0=self.Om0, Tcmb0=self.Tcmb0.to_value(au.Kelvin),
                                   m_nu = self.mnu*au.eV, Ob0 = self.Ob0) 
        return True

class LambdaCDMCosmo():
    """
    Generic LambdaCDM Cosmology class
    """

    def __init__(self,
                 H0=H0,
                 omk=omk, ombh2=ombh2, omch2=omch2,
                 mnu=mnu, tau=tau,
                 As=As, ns=ns):
        self.H0 = H0
        self.omk = omk
        self.ombh2 = ombh2
        self.omch2 = omch2
        self.mnu = mnu
        self.tau = tau
        self.As = As
        self.ns = ns

    def get_Cl(self, camb_lmax_out=camb_lmax_out,
               camb_lmax=camb_lmax,
               camb_lens_potential_accuracy=camb_lensing_potential_accuracy):
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=self.H0,
                           omk=self.omk, ombh2=self.ombh2, omch2=self.omch2,
                           mnu=self.mnu, tau=self.tau)
        pars.InitPower.set_params(As=self.As, ns=self.ns)
        pars.set_for_lmax(camb_lmax,
                          lens_potential_accuracy=camb_lens_potential_accuracy)
        self.CMBcosmo = camb.get_results(pars)
        camb_CMB_power = self.CMBcosmo.get_cmb_power_spectra(
            pars, CMB_unit='muK', raw_cl=True)
        self.CMBpower = camb_CMB_power['total'][:camb_lmax_out + 1, 0]

        return True
