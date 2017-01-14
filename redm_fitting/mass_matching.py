#!/u/ki/dapple/bin/python

import numpy as np
import scipy
import pyfits
import sys
from glob import glob

import cosmo

import emcee

#Various functions for abundance matching to redmapper clusters
#In order to obtain a lambda-mass relationship

#Main abundance matching function
def mass_matching(volume,lambda_val,mass,mf,scatter):
    '''
    Inputs: volume lambda (list of richnesses) Mvir dN/d ln M scatter(lambda|M) in dex
    Note that all values should be provided with consistent h units
    Returns a list of halo masses corresponding to each of the input richness values
    '''

    #First, add scatter to the richnesses
    if scatter > 0:
        lambda_sc = lambda_val*np.exp(np.random.normal(scale=np.sqrt(scatter**2+1./lambda_val),size=len(lambda_val)))
    else:
        lambda_sc = lambda_val*np.exp(np.random.normal(scale=np.sqrt(1./lambda_val),size=len(lambda_val)))

    #First, get the ordering of the richness list, most to least massive
    ranking = np.argsort(np.argsort(lambda_sc)[::-1])

    #Turn this into an abundance value
    abundance = (ranking+1.)/volume

    #Convert mass function into N(>M) /dM
    nbins = len(mf)
    dlnM = np.log(mass[1]/mass[0])
    ngtm = 0*mf
    for i in range(nbins):
        ngtm[i] = np.sum( mf[i:]*dlnM )

    #Interpolate to get the relevant ln(mass) value
    lnmass = np.interp(np.log(abundance),np.log(ngtm)[::-1],np.log(mass)[::-1]-dlnM/2.)
    mass_val = np.exp(lnmass)

    return [ranking, ngtm, mass_val]

#Functions for estimating a lambda-mass power law fit w/scatter
def func_power(param,x):
    A_lm = param[0]
    B_lm = param[1]
    lnlm0 = param[2]
    
    #Note this uses a fixed pivot at 1e14
    Mpiv = 2.35e14

    y = (np.array(x[0])/Mpiv)**A_lm*np.exp(lnlm0)*(1+np.array(x[1]))**B_lm

    return y

def func_power_ext(param,x):
    '''
    Modified power law -- extra curvy
    '''
    A_lm = param[0]
    B_lm = param[1]
    C_lm = param[2]
    lnlm0 = param[3]

    #Fixed pivot point
    Mpiv = 2.35e14

    y = np.exp(lnlm0 + A_lm*np.log(np.array(x[0])/Mpiv) + B_lm*np.log(1+np.array(x[1])) + C_lm*np.log(1+np.array(x[1]))**2  )

    return y

def power_likelihood_with_scatter(param,scatter,lambda_val,mass,z):
    '''
    Calculate the likelihood for a power law lambda-mass relationship, given an
    input scatter value and the abundance-matched lambda-mass relationship
    
    Scatter should be given in terms of log(lambda) at fixed mass
    
    Includes redshift dependence
    '''
    Mpiv = 2.35e14

    A_lm = param[0]
    B_lm = param[1]
    lnlm0 = param[2]
    
    delta = np.log(lambda_val/func_power(param,[mass,z]))

    logp = -np.sum( abs(delta/np.sqrt(scatter**2+1./lambda_val)) )/2.
    
    return logp

def power_likelihood_noev(param,scatter,lambda_val,mass):
    '''
    Calculate the likelihood for a power law lambda-mass relationship, given an
    input scatter value and the abundance-matched lambda-mass relationship
    
    Scatter should be given in terms of log(lambda) at fixed mass
    
    Includes redshift dependence
    '''
    Mpiv = 2.35e14

    A_lm = param[0]
    B_lm  = 0.
    lnlm0 = param[1]
    in_param = [A_lm, B_lm, lnlm0]

    delta = np.log(lambda_val/func_power(in_param,[mass,0*mass]))

    logp = -np.sum( abs(delta/np.sqrt(scatter**2+1./lambda_val)) )/2.
    
    return logp

def power_likelihood_ext(param,scatter,lambda_val,mass,z):
    '''
    Calculate the likelihood for a power law lambda-mass relationship, given an
    input scatter value and the abundance-matched lambda-mass relationship
    This version includes an additional redshift evolution term
    
    Scatter should be given in terms of log(lambda) at fixed mass
    
    Includes redshift dependence
    '''
    Mpiv = 2.35e14

    A_lm = param[0]
    B_lm = param[1]
    C_lm = param[2]
    lnlm0 = param[3]
    
    delta = np.log(lambda_val/func_power_ext(param,[mass,z]))

    logp = -np.sum( abs(delta/np.sqrt(scatter**2+1./lambda_val)) )/2.
    
    return logp

def determine_mass_lambda_relationship(zmin,zmax,area,mf,scatter,lambda_val,z,
                                       start=[],nsteps=0,use_ext=False,silent=False,
                                       mcut = 10.**14.):
    '''
    Overall function that gets the mass-lambda relationship

    Inputs:
    zmin,zmax,area,mf,scatter,lambda_val,z
    zmin: list of lower redshift bin limits
    zmax: list of upper redshift bin limits
    area: area of data section, important for abundance matching
    scatter: Scatter in ln(lambda) as fixed mass.  Should set to 0.182
    lambda_val: List of 
    z: associated list of redshifts

    Outputs:
    matchlist: index list of clusters that got abundance matched
    mass: abundance matched mass list
    param (parameter fit results)
    
    '''
    #Abundance match in each redshift bin
    #Note we're using Planck cosmology (default)
    nz = len(zmin)
    h = 0.6704
    matchlist = []
    mass = []
    for i in range(nz):
        if isinstance(area,list) or isinstance(area,np.ndarray):
            volume = cosmo.comoving_volume(zmin[i],zmax[i],H0=100.)*area[i]/41253.
        else:
            volume = cosmo.comoving_volume(zmin[i],zmax[i],H0=100.)*area/41253.
        zlist = np.where( (z >= zmin[i]) & (z < zmax[i]))[0]
        #print np.max(zlist), len(lambda_val), len(mf), i
        rank, ngtm, mass_temp = mass_matching(volume,lambda_val[zlist],mf[i][:,0]/h,mf[i][:,1],scatter)
        if not silent:
            print i, len(matchlist), len(zlist), len(mass_temp)
        if i == 0:
            matchlist = np.copy(zlist)
            mass = np.copy(mass_temp)
        else:
            matchlist = np.append(matchlist,zlist)
            mass = np.append(mass,mass_temp)

    if not silent:
        print "Done running abundance matching"

    #Abundance matched lambda values
    lambda_match = lambda_val[matchlist]
    masscutlist = np.where(mass > mcut)[0]

    #Using matched data results, run a fit with emcee
    #A_lm, B_lm, lnlm0
    if len(start)==0:
        start = [0.8,0.8,3.0]
        if use_ext:
            start = [0.8,0.8,0.,3.0]
    nparam = len(start)
    err_guess = np.zeros(len(start))+0.1
    nwalkers = 50
    if nsteps <=0:
        nsteps = 200

    walk_start = np.zeros([nwalkers,nparam])
    for i in range(nparam):
        walk_start[:,i] = np.random.normal(start[i],err_guess[i],nwalkers)

    #Running the fitting
    if use_ext:
        sampler = emcee.EnsembleSampler(nwalkers,nparam,power_likelihood_ext,
                                        args=[scatter,lambda_match[masscutlist],
                                              mass[masscutlist],
                                              z[matchlist[masscutlist]]])
    else:
        sampler = emcee.EnsembleSampler(nwalkers,nparam,power_likelihood_with_scatter,
                                        args=[scatter,lambda_match[masscutlist],
                                              mass[masscutlist],
                                              z[matchlist[masscutlist]]])

    pos,prob,state = sampler.run_mcmc(walk_start, nsteps)
    
    if not silent:
        print "Done with sampling"
    
    #Print some fit diagnostics

    return matchlist, mass, sampler.flatchain, sampler.flatlnprobability

#Similar to above function, but operates on a single redshift bin
#DOES NOT include redshift evolution
def determine_mass_lambda_single(zmin,zmax,area,mf,scatter,lambda_val,z,nsteps=0,
                                 start=[],mcut=1e14,silent=False):
    h = 0.6704    
    volume = cosmo.comoving_volume(zmin,zmax,H0=100.)*area/41253.
    
    matchlist = np.where( (z > zmin) & (z < zmax) )[0]
    #Do basic mass matching in this bin
    rank, ngtm, mass = mass_matching(volume,lambda_val[matchlist],mf[:,0]/h,mf[:,1],scatter)

    if not silent:
        print "Done running abundance matching"

    lambda_match = lambda_val[matchlist]
    masscutlist = np.where(mass > mcut)[0]
    
    if len(start)==0:
        start = [0.8, 3.06]
    nparam = len(start)
    err_guess = np.zeros(nparam)+0.1
    nwalkers = 50
    if nsteps <= 0:
        nsteps = 150

    walk_start = np.zeros([nwalkers,nparam])
    for i in range(nparam):
        walk_start[:,i] = np.random.normal(start[i],err_guess[i],nwalkers)

    #Setting up the sampler -- final prep to run
    sampler = emcee.EnsembleSampler(nwalkers,nparam,power_likelihood_noev,
                                    args=[scatter,lambda_match[masscutlist],
                                          mass[masscutlist]])

    #Burn-in
    pos, prob, state = sampler.run_mcmc(walk_start,50)
    sampler.reset()

    #Run the main part
    pos, prob, state = sampler.run_mcmc(pos,nsteps)

    if not silent:
        print "Done with fitting"

    return sampler.flatchain, sampler.flatlnprobability

def mass_lambda_with_bootstrap(zmin,zmax,area,mf,scatter,lambda_val,z,start=[],nsteps=200,
                               use_ext = False):
    '''
    Calls determine_mass_lambda_relationship

    Inputs:
    zmin,zmax,area,mf,scatter,lambda_val,z
    zmin: list of lower redshift bin limits
    zmax: list of upper redshift bin limits
    area: area of data section, important for abundance matching
    scatter: Scatter in ln(lambda) as fixed mass.  Should set to 0.182
    lambda_val: List of 
    z: associated list of redshifts

    Outputs: 
    param_main (main parameter results)
    param_boots (parameter results for bootstrap samples)
    '''

    #Get the first abundance matching results
    matchlist, mass, chain, lnprob = determine_mass_lambda_relationship(zmin,zmax,area,mf,
                                                                        scatter,lambda_val,z,start=start,
                                                                        nsteps=nsteps,use_ext=use_ext)
    param_main = chain[np.argmax(lnprob)]

    print >> sys.stderr, "Done with main result; starting bootstrap..."

    #Now, run over 100 boostrap samples
    nboot = 100
    ncl = len(lambda_val)
    param_boots = np.zeros([nboot,len(param_main)])
    for i in range(nboot):
        print >> sys.stderr, i," of ",nboot
        #Generate random sample
        bootlist = np.random.randint(0,ncl,ncl)
        matchlist, mass, chain, lnprob = determine_mass_lambda_relationship(zmin,zmax,area,mf,
                                                                            scatter,lambda_val[bootlist],
                                                                            z[bootlist],start=start,
                                                                            nsteps=nsteps,use_ext=use_ext,silent=True)
        param_boots[i] = chain[np.argmax(lnprob)]

    print >> sys.stderr, "Done"

    return param_main, param_boots

#Function for estimating typical area in a redshift slice (or slices)
def estimate_area(zmin,zmax,z,area):
    nz = len(zmin)
    area_bin = np.zeros(nz)

    for i in range(nz):
        dz = (zmax[i]-zmin[i])/10.
        zvals = zmin[i]+dz/2.+np.array(range(10))*dz
        r = cosmo.rco(zvals,H0=100)
        area_bin[i] = np.sum(r**2*np.interp(zvals,z,area))/np.sum(r**2)

    return area_bin


#Running the fit, with output, for DR8
if __name__ == "__main__":
    print >> sys.stderr, "I am running okay"
    c_dr8 = pyfits.open("/nfs/slac/g/ki/ki19/des/erykoff/clusters/process/dr8_redmapper_v5.10/run_ubermem/dr8_run_redmapper_v5.10_lgt5_catalog.fit")
    c_dr8 = c_dr8[1].data

    lm_list = np.where(c_dr8['lambda_chisq'] > 9)[0]
    c_dr8 = c_dr8[lm_list]
    print >> sys.stderr, len(c_dr8)

    indir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/mass_functions_planck/"
    mf_files = glob(indir+"mf_planck_z_*.dat")
    mf_files = np.sort(mf_files)
    mf_files = mf_files[[1,3,5,7,10]]
    mf = []
    for i in range(len(mf_files)):
        mf.append(np.loadtxt(mf_files[i]))

    zmin = np.array([0.1,0.15,0.2,0.25,0.3])
    zmax = np.array([0.15,0.2,0.25,0.3,0.33])

    param, param_boots = mass_lambda_with_bootstrap(zmin,zmax,10405.,mf,0.184,c_dr8['lambda_chisq'],c_dr8['z_lambda'],start=[0.8,0.8,2.9])

    fitdir_dr8 = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10_v2/"
    f = open(fitdir_dr8+"lm_param.dat",'w')
    print >> f, param[0], param[1], param[2]
    for i in range(len(param_boots)):
        print >> f, param_boots[i,0],param_boots[i,1],param_boots[i,2]
    f.close()
    
    print >> sys.stderr, "All done"

#Run a full mass-matching set -- does all the annoying stuff, and outputs
#fits in individual redshift bins to output
def run_matching_zbin_and_print(cat,zmin,zmax,zlabel,area_bin,outdir,
                                indir="/nfs/slac/g/ki/ki10/rmredd/redmapper_data/mass_functions_planck/"):
    '''
    Inputs:
        cat -- redmapper cluster catalog
        zmin -- min redshift of each bin
        zmax -- max redshift of each bin
        zlabel -- redshift label for mass function files
        area_bin -- effective area in each redshift slice
        outdir -- output directory
    '''

    #Get halo mass functions
    mf_files = []
    for i in range(len(zlabel)):
        mf_files.append(indir + "mf_planck_z_"+zlabel[i]+".dat")
    mf_files = np.array(mf_files)
    #print mf_files
    mf = []
    #Read in the mass function information
    for i in range(len(mf_files)):
        mf.append(np.loadtxt(mf_files[i]))
    
    zmid = (zmin+zmax)/2.
    param = np.zeros([len(zmid),2])
    param_std = np.copy(param)
    for i in range(len(zmin)):
        chain, prob = determine_mass_lambda_single(zmin[i],zmax[i],area_bin[i],mf[i],0.184,cat['lambda_chisq'],cat['z_lambda'],start=[0.8,3.06])
        param[i] = chain[np.argmax(prob)]
        param_std[i] = np.std(chain,axis=0)

    #Printing A_lm
    f = open(outdir+"A_lm_z.dat",'w')
    for i in range(len(zmid)):
        print >> f, zmid[i], param[i,0], param_std[i,0]
    f.close()

    #Printing ln lambda_0
    f = open(outdir+"lnlm0_z.dat",'w')
    for i in range(len(zmid)):
        print >> f, zmid[i], param[i,1], param_std[i,1]
    f.close()

    return param, param_std
