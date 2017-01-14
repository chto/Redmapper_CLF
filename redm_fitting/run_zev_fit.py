#!/usr/bin/python

import numpy as np
import sys

import emcee

#Likelihood function for constant
def like_constant(param,z,x,err):
    s = param[1]
    if s <= 0 or s > 100:
        return -np.inf
    chi2 = np.sum((param[0] - x)**2./err/err)
    
    lnp = -chi2/2.
    
    return lnp

#Likelihood function for evolution with (1+z)
def like_power(param,z,x,err):
    s = param[2]
    if s <= 0 or s > 100:
        return -np.inf
    chi2 = np.sum((param[0]*((1+z)/1.3)**param[1]-x)**2/err/err)

    lnp = -chi2/2.

    return lnp

#Likelihood function for evolution with (1+z) -- switch to log-log
def like_power_log(param,z,x,err):
    s = param[2]
    if s <= 0 or s > 100:
        return -np.inf
    chi2 = np.sum((param[0] + np.log((1+z)/1.3)*param[1]-x)**2/err/err)

    lnp = -chi2/2.

    return lnp

#General function for fitting redshift evolution
def general_zev_fit(z,x,err,zrange=[0,1],start=[],fit_type='constant'):
    zlist = np.where( (z > zrange[0]) & (z < zrange[1]) )[0]
    if len(zlist) == 0:
        print "ERROR: Unable to compute, no data points in redshift range"
        return

    args = [z[zlist],x[zlist],err[zlist]]
    nwalkers = 50
    nsteps_burn = 1000
    nsteps = 200

    if fit_type=='constant':
        if len(start) != 2:
            start = [1., 1.]
        nparam = 2
        err_guess = [0.1,0.1]
        sampler = emcee.EnsembleSampler(nwalkers,nparam,like_constant,args=args)
    elif fit_type=='power' or fit_type == 'power_log':
        if len(start) != 3:
            start = [1., 0., 1.]
        err_guess = [0.1,0.1,0.1]
        nparam = 3
        if fit_type == 'power':
            sampler = emcee.EnsembleSampler(nwalkers,nparam,like_power,args=args)
        else:
            sampler = emcee.EnsembleSampler(nwalkers,nparam,like_power_log,args=args)
    else:
        print "ERROR: Fit type \"",fit_type,"\" not recognized; returning"
        return

    #Set up the walkers
    walk_start = np.zeros([nwalkers,nparam])
    for i in range(nparam):
        walk_start[:,i] =  np.random.normal(start[i],err_guess[i],nwalkers)
    
    #Fast burn-in
    pos, prob, state = sampler.run_mcmc(walk_start,nsteps_burn)
    sampler.reset()

    #Run for realz
    pos, prob, state = sampler.run_mcmc(pos,nsteps)

    param = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
    ehi = np.percentile(sampler.flatchain,84,axis=0)-param
    elo = param-np.percentile(sampler.flatchain,16,axis=0)

    print param
    print ehi
    print elo
                              
    if fit_type == 'constant':
        chi2 = np.sum((x[zlist]-param[0])**2/err[zlist]/err[zlist])
    if fit_type == 'power':
        chi2 = np.sum((x[zlist]-param[0]*((1+z[zlist])/1.3)**param[1])**2/err[zlist]/err[zlist])
    if fit_type == 'power_log':
        chi2 = np.sum((x[zlist]-param[0] - np.log((1+z[zlist])/1.3)*param[1])**2/err[zlist]/err[zlist])

    print "Best fit chisq,n: ",chi2,len(zlist)

    return param, ehi, elo, sampler.flatchain, sampler.flatlnprobability


if __name__ == "__main__":
    #Input of command-line arguments:
    #filename fit_type zmin zmax start
    if len(sys.argv) < 7:
        print >> sys.stderr, "ERROR: required input format is:"
        print >> sys.stderr, "       filename fit_type zmin zmax start0 start1 [start2]"
        sys.exit(1)

    filename = sys.argv[1]
    fit_type = sys.argv[2]
    if fit_type != 'constant' and fit_type != 'power' and fit_type != 'power_log':
        print >> sys.stderr, "WARNING: Unrecognized fit_type (options are constant or power)"
        print >> sys.stderr, "Setting to default (constant)"
        fit_type = 'constant'
    if fit_type == 'constant':
        start = np.zeros(2)
    if fit_type == 'power' or fit_type == 'power_log':
        start = np.zeros(3)
    zmin = float(sys.argv[3])
    zmax = float(sys.argv[4])
    zrange = [zmin, zmax]
    
    for i in range(len(start)):
        start[i] = float(sys.argv[5+i])

    #Read the input file
    dat = np.loadtxt(filename)

    #Now actually run the fit
    param, ehi, elo, chain, prob = general_zev_fit(dat[:,0],dat[:,1],dat[:,2],zrange=zrange,start=start,
                                                   fit_type=fit_type)
