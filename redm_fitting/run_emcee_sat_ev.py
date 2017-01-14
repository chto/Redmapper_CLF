#!/u/ki/dapple/bin/python

import numpy as np
import sys
import os
import time
from glob import glob
import emcee

import fit_with_covar
import fit_plm
import fit_psat
import cosmo

#Parameter file readin
#Note that "#" is the comment symbol
def read_param(filename):
    keys=[]
    vals=[]

    f = open(filename)
    for line in f:
        if line.strip()=='':
            continue
        if line[0] == '#':
            continue
        entries = line.split()
        keys.append(entries[0])
        if len(entries[1:])==1:
            vals.append(entries[1])
        else:
            vals.append(entries[1:])
    f.close()

    #First check for repeated keys -- only hmf_file should be repeated
    setup = []
    hmf_files = []
    pk_files = []
    for i in range(len(keys)):
        if (keys.count(keys[i]) > 1) & (keys[i]!="hmf_file"):
            print >> sys.stderr, "ERROR: Input value "+keys[i]+" should not be repeated"
            sys.exit(1)
        if (keys[i] != "hmf_file"):
            setup.append((keys[i],vals[i]))
        else:
            hmf_files.append(vals[i])
    #Add the hmf_file values; should have zmin, zmax, hmf filename, pk filename values
    if keys.count('hmf_file')==0:
        print >> sys.stderr, "ERROR: Require input of hmf_files, redshift ranges"
        sys.exit(1)
    setup.append(('hmf_files',hmf_files))
    #Make the preliminary dictionary
    params = dict(setup)

    #Format check for each key
    if keys.count('indir')==0:
        print >> sys.stderr, "ERROR: No input data directory"
        sys.exit(1)
    if keys.count('indir_uber')==0:
        print >> sys.stderr, "ERROR: No input data directory"
        sys.exit(1)
    if keys.count('area')==0:
        print >> sys.stderr, "ERROR: No area given"
        sys.exit(1)
    if keys.count('outdir')==0:
        print >> sys.stderr, "WARNING: Outputing to local directory"
        setup.append(('outdir',''))
    if keys.count('glist')==0:
        print >> sys.stderr, "WARNING: Using default glist (DR8)"
        setup.append(('glist',['0','1','3','4','5','6','7']))
    if keys.count('lm_med')==0:
        print >> sys.stderr, "WARNING: Using default lm_med values (DR8)"
        setup.append(('lm_med',['11.8','17.0','27.2','22.1','27.2','33.9','50.4','46.9','70.7']))
    if keys.count('A_lm')==0:
        print >> sys.stderr, "Default A_lm=0.838"
        setup.append(('A_lm','0.838')) 
    if keys.count('B_lm')==0:
        print >> sys.stderr, "Default B_lm=0.835"
        setup.append(('B_lm','0.835'))
    if keys.count('lnlm0')==0:
        print >> sys.stderr, "Default lnlm0=3.018"
        setup.append(('lnlm0','3.018'))
    if keys.count('sigma_lm')==0:
        print >> sys.stderr, "Default sigma_lm=0.1842 (20%)"
        setup.append(('sigma_lm','0.1842'))
#Section for reading in relevant cosmological parameters
    if keys.count('H0')==0:
        print >> sys.stderr, "Default H0 (planck)"
        setup.append(('H0','67.04'))
    if keys.count('omegaM')==0:
        print >> sys.stderr, "Default omegaM (planck)"
        setup.append(('omegaM','0.317'))
    if keys.count('omegaL')==0:
        print >> sys.stderr, "Default omegaL (planck)"
        setup.append(('omegaL','0.683'))
    if keys.count('likelihood_type')==0:
        print >> sys.stderr, "Default likelihood being used (phi_ev)"
        setup.append(('likelihood_type','0'))
    if keys.count('lm_param_file')==0:
        print >> sys.stderr, "Default is to use fixed lambda-mass parameters (no input file)"
        setup.append(('lm_param_file','0'))

    params = dict(setup)
    #Convert the glist key to a list of integers
    glist_temp = params['glist']
    for i in range(len(glist_temp)):
        glist_temp[i] = int(glist_temp[i])
    params['glist'] = glist_temp
    #And the lm_med values to floats
    lm_temp = params['lm_med']
    for i in range(len(lm_temp)):
        lm_temp[i] = float(lm_temp[i])
    params['lm_med'] = lm_temp

    return params

def make_full_sat_data_vector(Mpiv,indir,indir_uber,area,hmf_files,glist,Lcut=[],
                              H0=67.04,omegaM=0.317,omegaL=0.683):
    nz = len(hmf_files)
    nlm = len(glist)

    if len(Lcut) == 0:
        Lcut = np.zeros(nz)
    
    myz = np.zeros(nz)
    volume = np.zeros(nz)
    x = []
    y = []
    covar_all = []
    mass_param = []
    lmed = []
    mf_set = []

    for i in range(nz):
        dat = np.loadtxt(hmf_files[i][2])
        #CORRECTIONS TO PLANCK COSMOLOGY -- STUPID LITTLE H; setting h based on cosmology
        h = H0/100.
        dat[:,0] = dat[:,0]/h
        dat[:,1] = dat[:,1]*h**3

        #Saving the corrected halo mass function
        mf_set.append(dat)

        l_zmin = hmf_files[i][0]
        l_zmax = hmf_files[i][1]
        print >> sys.stderr, i, l_zmin, l_zmax
        myz[i] = (float(l_zmin)+float(l_zmax))/2.

        #Get the approximation to the halo mass function
        mass_param.append(fit_plm.nm_approx_third(dat[:,0],dat[:,1],Mpiv))
        
        #Calculate the volume
        #Note that this defaults to Planck cosmology, h=0.67
        volume[i] = cosmo.comoving_volume(float(l_zmin),float(l_zmax),H0=H0,
                                          omegaM=omegaM,omegaL=omegaL)*(area/41253.)

        #Read in and correct the matter power spectrum
        pkdat = np.loadtxt(hmf_files[i][3])
        k = np.exp(pkdat[:,1])
        Pk = np.exp(pkdat[:,0])

        #Make the generic input data vectors for this particular redshift
        [xtemp, ytemp, covar, lmed_temp] = fit_psat.make_data_vector(indir, indir_uber, 
                                                                     l_zmin, l_zmax, 
                                                                     Lcut=Lcut[i], glist=glist)
        for j in range(nlm):
            x.append(xtemp[j])
            y.append(ytemp[j])
            covar_all.append(covar[j])
            lmed.append(lmed_temp[j])

    myz = np.repeat(myz,nlm)
    return [x, y, covar_all, myz, lmed, volume, mass_param, mf_set]

#Function for getting lambda min/max ranges
def get_lambda_ranges(indir_uber, glist, hmf_files):
    l_zmin = hmf_files[0][0]
    l_zmax = hmf_files[0][1]

    satfiles = glob(indir_uber+"clf_sat_z_"+l_zmin+"_"+l_zmax+"*.dat")
    satfiles = np.array(satfiles)
    satfiles.sort()
    if len(glist) > 0:
        satfiles = satfiles[glist]
    
    lm_min = np.zeros(len(satfiles))
    lm_max = np.zeros_like(lm_min)

    for i in range(len(satfiles)):
        mylist = satfiles[i].split('_')
        lm_min[i] = float(mylist[-2])
        lm_max[i] = float(mylist[-1][0:4])

    return lm_min, lm_max

#Purpose is to take in the descriptive parameter file, then
#run for the full set of all input redshifts
if __name__ == '__main__':
    #Read the input parameters
    if len(sys.argv) < 2:
        print >> sys.stderr, "ERROR:  Required input format is:"
        print >> sys.stderr, "        paramfile"
        sys.exit(1)

    input_params = read_param(sys.argv[1])
    print >> sys.stderr, "Parameter file read successful"
    #print input_params

    #Count number of redshift bins
    nz = len(input_params['hmf_files'])
    indir = input_params['indir']
    indir_uber = input_params['indir_uber']
    outdir = input_params['outdir']
    area = float(input_params['area'])
    likelihood_type = int(input_params['likelihood_type'])

    #Make the main output directory
    os.system("mkdir -p "+outdir)

    #Note glist has been added to the readin
    glist = input_params['glist']

    #Run data setup for every redshift bin
    Mpiv = 2.35e14
    #Using Lcut
    #NOTE THAT THIS IS CURRENTLY NOT A VERY CLEVER APPROACH -- NEED TO 
    #FIND ISSUE WITH L<LCUT
    Lcut = np.zeros(nz)
    for i in range(nz):
        Lcut[i] = fit_psat.Lmin_eli( (float(input_params['hmf_files'][i][0])+float(input_params['hmf_files'][i][1]))/2. )
    print >> sys.stderr, "Lcut: ", Lcut
    [x, y, covar_all, myz, lmed, volume, mass_param, mf_set ] = make_full_sat_data_vector(Mpiv,indir,indir_uber,area,input_params['hmf_files'],glist,Lcut=Lcut,H0=float(input_params['H0']),omegaM=float(input_params['omegaM']),omegaL=float(input_params['omegaL']))

    #Get total number of data points
    npoints = 0
    for i in range(len(y)):
        npoints = npoints + len(y[i])

    print >> sys.stderr, "Total number of data points is: ",npoints
    print >> sys.stderr, "Data readin complete"

    #Quick function for getting lambda min/max ranges
    [lm_min, lm_max] = get_lambda_ranges(indir_uber,glist, input_params['hmf_files'])

    #Fixed lambda-mass relationship parameter values
    #Mpiv A_lm B_lm lnlm0
    #lm_param = [Mpiv, 0.838, 0.835, 3.018]
    #Now updated to take input read in from parameter file
    sigma_lm = float(input_params['sigma_lm'])
    lm_param = [Mpiv, float(input_params['A_lm']), float(input_params['B_lm']), float(input_params['lnlm0']),sigma_lm]
    if input_params['lm_param_file'] != '0':
        #Test to see if input file exists
        if os.path.isfile(input_params['lm_param_file']):
            #If so, get the full list of desired parameters
            #Note Mpiv, sigma_lm still externally fixed in parameter file
            lm_param_dat = np.loadtxt(input_params['lm_param_file'])
            lm_param = np.zeros([len(lm_param_dat),len(lm_param_dat[0])+2])
            lm_param[:,1:len(lm_param_dat[0])+1] = lm_param_dat
            lm_param[:,0] = Mpiv
            lm_param[:,-1] = sigma_lm

    #Initial parameter guess
    #Parameters are:
    #ln phi0 A_phi B_phi lnLs0 A_s B_s alpha B_a s
    #And also includes s term for covariance scaling
    #start = [ 4.3, 0.82, 0., 22.56, 0.01, 2.7, -.85, 0., 4.]
    #start = [ 4.47, 0.855, -0.455, 22.33, 5.31e-3, 3.4, -0.21, -1.31, 3.84 ]
    #Version for running with phist evolution only
    #ln phi0 A_phi lnLs0 A_s B_s alpha B_phi s
    print >> sys.stderr, "Likelihood type: ",likelihood_type
    #start = [ 4.12, 0.8, 22.90, 0.05, 2.0, -0.95, 0.1, 3.84 ]
    start = [ 3.91, 0.78, 23.105, 0.045, 1.5, -0.85, 0.6, 1.5 ]
    if likelihood_type ==2:
        #Need additional beta parameter for these likelihoods
        start = [ 3.91, 0.78, 23.105, 0.045, 1.5, -0.85, 0.6, 0.98, 1.5 ]
    if likelihood_type == 3:
        #Running with fixed mass dependence
        #ln phi0 lnLs0 B_s alpha B_phi s
        start = [3.91, 23.105, 2.0, -0.9, 0.6, 1.5 ]

    nparam = len(start)
        

    #Centrals parameters -- currently irrelevant, but just in case desired later on
    cen_param = np.zeros(5)

    nz = len(mass_param)
    nlm = len(x)/nz
    #Trim down the mass function set -- really low masses aren't helpful 
    #and just slow us down
    clist = np.where(mf_set[0][:,0] >= 1e13)[0]
    for i in range(nz):
        mf_set[i] = mf_set[i][clist]

    #Now, set everything up for emcee
    #How many walkers?  Start with 50 as default
    nwalkers = 50

    #Now, draw randoms from a Gaussian distribution
    #Note fiducial distribution widths
    err_guess = [0.1, 0.1, 0.01, 0.1, 0.05, 0.1, 0.1, 0.5]
    if likelihood_type == 2:
        err_guess = [0.1, 0.1, 0.1, 0.1, 0.05, 0.1, 0.1, 0.02, 0.5]
    if likelihood_type == 3:
        err_guess = [0.1, 0.01, 0.05, 0.1, 0.1, 0.5]
    #Note this produces the initial "ball" of walkers
    walk_start = np.zeros([nwalkers, nparam])
    for i in range(nparam):
        walk_start[:,i] = np.random.normal(start[i],err_guess[i],nwalkers)

    #Make the sampler:
    psat_arg_set = [lm_min, lm_max, mf_set, npoints]
    #Switch between likelihood functions
    if likelihood_type == 0:
        sampler = emcee.EnsembleSampler(nwalkers, nparam, fit_psat.psat_likelihood_conv_phi, args=[lm_param,cen_param,mass_param,x,y,covar_all,myz,psat_arg_set])
    if likelihood_type == 1:
        sampler = emcee.EnsembleSampler(nwalkers, nparam, fit_psat.psat_likelihood_conv_schcorr, args=[lm_param,cen_param,mass_param,x,y,covar_all,myz,psat_arg_set])
    if likelihood_type == 2:
        #Version that runs with likelihood including difference from Schechter
        #at the bright end
        sampler = emcee.EnsembleSampler(nwalkers, nparam, fit_psat.psat_likelihood_sch_beta, args=[lm_param,cen_param,mass_param,x,y,covar_all,myz,psat_arg_set])
    if likelihood_type == 3:
        sampler = emcee.EnsembleSampler(nwalkers, nparam, fit_psat.psat_likelihood_fix_mass, args=[lm_param,cen_param,mass_param,x,y,covar_all,myz,psat_arg_set])

    #Run 20 steps with 50 walkers for burn-in
    nsteps_burn = 20
    print >> sys.stderr, "Beginning burn-in with ",nwalkers, " walkers and ",nsteps_burn," steps"
    pos, prob, state = sampler.run_mcmc(walk_start, nsteps_burn)
    sampler.reset()

    print >> sys.stderr, "First part of MCMC done.  Starting main segment..."
    #Currently running with 500 steps for the main segment
    nsteps = 600
    
    #Now print out the resulting chain to a file
    f = open(outdir+"chain_sat_ev_all.dat",'w')
    f.close()
    
    for result in sampler.sample(pos, iterations=nsteps, storechain=False):
        position = result[0]
        prob = result[1]
        f = open(outdir+"chain_sat_ev_all.dat",'a')
        for i in range(nwalkers):
            for j in range(nparam):
                f.write(str(position[i][j])+" ")
            f.write(str(prob[i])+"\n")
        f.close()

    #print >> sys.stderr, "Acceptance fraction is: ", sampler.acceptance_fraction()
    print >> sys.stderr, "We're all through here."
