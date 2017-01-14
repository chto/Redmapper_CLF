#!/usr/bin/env python

from math import *
import numpy as np

#H(z)
def Hubble(z, H0=100., omegaM=0.27, omegaL=0.73):
    return H0*sqrt(omegaM*(1.+z)**3+omegaL)

#comoving distance as a function of z
#Defaults set to Planck cosmology
def rco(z, H0=67.04, omegaM=0.317, omegaL=0.683):
    c = 299890.

    #perform numerical integration
    dz = 0.001
    if len(z) > 1:
        zmax = max(z)
    else:
        zmax = z
    nbins = int(ceil(zmax/dz))
    if zmax == 0:
        return z
    dz = zmax/(nbins-1)
    if (nbins == 0):
        return np.zeros(len(z))

    myr = np.zeros(nbins)
    myz = np.zeros(nbins)
    for i in range(nbins):
        myz[i] = dz*i

    for i in range(nbins):
        if myz[i] == 0:
            continue
        myr[i] = myr[i-1] + c*(1./Hubble(dz*(i+1), H0=H0, omegaM=omegaM, omegaL=omegaL)+
                    1./Hubble(dz*i, H0=H0, omegaM=omegaM, omegaL=omegaL) )*dz/2.

    #now interpolate to get all desired values
    #print z, myz, myr, nbins
    r = np.interp(z,myz,myr)
    return r

#luminosity distance modulus given distance and (true, non-peculiar) redshift
#Assumes that r is in Mpc
def lm_distmod(r,z):
    dm = 5.*np.log10((1.+np.array(z))*np.array(r)*1e6/10.)
    return dm

#Gives volume in a spherical shell
def comoving_volume(zmin,zmax,H0=67.04, omegaM=0.317, omegaL=0.683):
    [rmin] = rco(np.array([zmin]),H0=H0,omegaM=omegaM,omegaL=omegaL)
    [rmax] = rco(np.array([zmax]),H0=H0,omegaM=omegaM,omegaL=omegaL)
    
    vol = (rmax**3-rmin**3)*4.*np.pi/3.

    return vol
