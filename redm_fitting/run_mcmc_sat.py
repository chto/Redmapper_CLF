#!/u/ki/dapple/bin/python

import numpy as np
import sys
import os
import time
from glob import glob

import fit_with_covar
import fit_plm
import fit_psat
import cosmo
import mcmc_errors as mcmc

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

def make_full_sat_data_vector(Mpiv,indir,indir_uber,area,hmf_files,glist,Lcut=[]):
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
        #CORRECTIONS TO PLANCK COSMOLOGY -- STUPID LITTLE H; setting h=0.6704
        h = 0.6704
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
        volume[i] = cosmo.comoving_volume(float(l_zmin),float(l_zmax))*(area/41253.)

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
    [x, y, covar_all, myz, lmed, volume, mass_param, mf_set ] = make_full_sat_data_vector(Mpiv,indir,indir_uber,area,input_params['hmf_files'],glist,Lcut=Lcut)

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
    lm_param = [Mpiv, 0.838, 0.835, 3.018]
    #Now updated to take input read in from parameter file
    lm_param = [Mpiv, float(input_params['A_lm']), float(input_params['B_lm']), float(input_params['lnlm0'])]

    #Initial parameter guess
    #Parameters are:
    #ln phi0 A_phi lnLs0 A_s B_s alpha s
    #And also includes s term for covariance scaling
    start = [ 3.99, 0.83, 23.011, 0.051, 2.52, -.919, 1.5]

    #Limits on parameter values
    p_limits = [ [-1e6, 1e6],
                 [-1e6, 1e6],
                 [-1e6, 1e6],
                 [-1e6, 1e6],
                 [-1e6, 1e6],
                 [-1e6, 1e6],
                 [0, 1e6] ]
    nparam = len(start)

    #Initial covariance guess
    covar_start = np.zeros([nparam,nparam])
    covar_start[0,0] = 0.1*0.1
    covar_start[1,1] = 0.1*0.1
    covar_start[2,2] = 0.1*0.1
    covar_start[3,3] = 0.1*0.1
    covar_start[4,4] = 0.1*0.1
    covar_start[5,5] = 0.1*0.1
    covar_start[6,6] = 0.1*0.1

    step_fac = 0.4/np.sqrt(nparam)

    p_covar = np.zeros_like(covar_start)

    #Centrals parameters -- currently irrelevant, but just in case desired later on
    cen_param = np.zeros(5)

    #Run 1000 elements to get a better start point/covariance guess
    #Start by getting the lm_bins, p(lambda), n(lambda) inputs that are needed later on
    n_lm_bins = 25
    lm_bins = []
    plm = []
    n_lambda = []
    nz = len(mass_param)
    nlm = len(x)/nz
    for i in range(nz):
        for j in range(nlm):
            [lm_b_t, plm_t, n_lambda_t] = fit_psat.func_sat_conv_prep(n_lm_bins, lm_min[j], lm_max[j], lm_param, mass_param[i], myz[i], mf_set[i][:,0], mf_set[i][:,1] )
            lm_bins.append(lm_b_t)
            plm.append(plm_t)
            n_lambda.append(n_lambda_t)

    chain = np.array([start])
    time_start = time.clock()
    p_start = fit_psat.psat_likelihood_conv(start,lm_param,cen_param,mass_param,x,y,covar_all,myz,[lm_min, lm_max, mf_set, npoints, lm_bins, plm, n_lambda])
    print >> sys.stderr, "Time for a single likelihood, in s: ",time.clock()-time_start, " Initial ln P=", p_start

    p_arr = [p_start]
    param = np.copy(start)
    p = np.copy(p_start)
    print >> sys.stderr, "Starting first part of MCMC"
    for i in range(1000):
        [param, p] = mcmc.take_mcmc_step_multiz_sat(param,p_limits,p,covar_start*step_fac**2, fit_psat.psat_likelihood_conv,lm_param,cen_param,mass_param,x,y,covar_all,myz,[lm_min, lm_max, mf_set, npoints, lm_bins, plm, n_lambda])
        chain = np.append(chain,[param],axis=0)
        p_arr.append(p)

    print >> sys.stderr, "First part of MCMC done.  Starting main segment..."

    #Get a fresh version of covar_start based on these values
    p_covar = mcmc.get_pcovar(chain[1:1000])
    
    #Run full chain and print results as we go
    nmcmc = 100000
    f = open(outdir+"chain_sat_all.dat",'w')
    for i in range(nmcmc):
        [param, p] = mcmc.take_mcmc_step_multiz_sat(param,p_limits,p,covar_start*step_fac**2, fit_psat.psat_likelihood_conv,lm_param, cen_param, mass_param,x,y,covar_all,myz,[lm_min, lm_max, mf_set, npoints, lm_bins, plm, n_lambda])
        chain = np.append(chain,[param],axis=0)
        p_arr.append(p)
        if i>1000:
            covar_start = mcmc.get_pcovar(chain[i-1000:])
        print >> f, chain[i+1001,0], chain[i+1001,1], chain[i+1001,2], chain[i+1001,3], chain[i+1001,4],chain[i+1001,5], chain[i+1001, 6], p_arr[i+1001]

    f.close()

    print >> sys.stderr, "We're all through here."
