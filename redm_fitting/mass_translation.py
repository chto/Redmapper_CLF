#!/u/ki/dapple/bin/python

import numpy as np
import scipy

import cosmo

#Functions here are used to get the change in halo mass that are
#purely due to evolution of the background density, 
#and estimate the change that should be applied to a
#lambda-mass relationship

#For a given halo mass, Rvir, r_s, calculate rho_s
def get_rho_s(m,rvir,r_s):
    c = rvir/r_s

    rho_s = m/4/np.pi/(1./(c+1) + np.log(c+1) -1)/r_s**3

    return rho_s

#Calculates mass within radius for a given Rvir, c, rho_s
def mass_contained(rvir,r_s,rho_s):
    mass = 4*np.pi*rho_s*(1./(rvir/r_s+1) + np.log(rvir/r_s+1) -1 )*r_s**3

    return mass

#Calculate halo radius given Mass, delta, z
def radius_from_delta(m,delta,z):
    #Get local critical density
    rho = cosmo.rho_crit(z)

    #get the radius
    r = (m/(delta*rho*4.*np.pi/3.))**(1./3.)

    return r

#Function minimized while searching for Rnew at fixed z, Mold
def func_rnew(Rnew,data):
    Mold = data[0]
    zold = data[1]
    c = data[2]
    znew = data[3]

    #Calculate radius of original halo
    rold = radius_from_delta(Mold,cosmo.virial_dens(zold),zold)
    r_s = rold/c #Won't change when halo changes redshift
    rho_s = get_rho_s(Mold,rold,r_s) #Won't change when halo changes redshift

    #For correct Rnew, this equation is zero
    diff = abs( mass_contained(Rnew,r_s,rho_s) / (4./3.*np.pi*Rnew**3) - (cosmo.virial_dens(znew)*cosmo.rho_crit(znew) ) )
    
    return diff

#Get the translated mass at various redshifts given an initial M, z pair
#Note that this assumed a fixed initial concentration of c=5
def mass_translate(Minit,zinit,zvals,c=5.):
    nz = len(zvals)
    #Get scale radius for this initial value
    Rinit = radius_from_delta(Minit,cosmo.virial_dens(zinit),zinit)
    r_s = Rinit/c
    #And scaled density
    rho_s = get_rho_s(Minit,Rinit,r_s)
    
    Mvals = np.zeros_like(zvals)
    for i in range(nz):
        data = [Minit, zinit, c, zvals[i]]
        [rtemp] = scipy.optimize.fmin(func_rnew,[1.0],args=([data]),disp=False)
        #Convert the new radius into a mass
        Mvals[i] = mass_contained(rtemp,r_s,rho_s)

    return Mvals

#Calculation of halo concentration as a function of sigma
#Note that this requires sigma as an input
#Based on Prada et al 2012
#A couple of utility functions are needed first
def prada_cmin(x):
    c0 = 3.681
    c1 = 5.033
    alpha = 6.948
    x0 = 0.424
    cmin = c0 + (c1 - c0)*(1/np.pi*np.arctan(alpha*(x-x0))+0.5)
    return cmin

def prada_smin(x):
    s0 = 1.047
    s1 = 1.646
    beta = 7.386
    x1 = 0.526
    smin = s0 + (s1 - s0)*(1/np.pi*np.arctan(beta*(x-x1))+0.5)
    return smin

def conc_prada(sigma, z):
    x = (0.684/0.317)**(1/3.)/(1+z)

    B0 = prada_cmin(x)/prada_cmin(1.393)
    B1 = prada_smin(x)/prada_smin(1.393)

    sigma_p = B1*sigma
    A = 2.881
    b = 1.257
    c = 1.022
    d = 0.060
    Cp = A*((sigma_p/b)**c + 1)*np.exp(d/sigma_p**2)

    c = B0*Cp
    return c


#For fitting power laws to our M(z) relations
#Want to find value of slope that minimizes fraction error
#Note that this is normalized to =1 at z=0.2
def func_mz(slope,zvals,mvals):
    
    #mvals_out = ((1+zvals)/1.2)**slope
    mvals_out = np.exp(((1+zvals)-1.2)*slope)

    ferr = ((mvals_out-mvals)/mvals)**2

    return np.sum(ferr)

