#!/u/ki/dapple/bin/python

import numpy as np
import sys
import os
import emcee
from glob import glob

import fit_with_covar
import fit_plm
import fit_cen_bsat

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

def make_full_data_vector(Mpiv,indir,hmf_files,glist,lm_med,lm_min=0,
                          H0=67.04,omegaM=0.317,omegaL=0.683):
    '''
    Note that this defaults to Planck cosmology
    '''
    nz = len(hmf_files)
    nlm = len(lm_med)

    myz = np.zeros(nz)
    mass_param = np.zeros([nz,4])
    Lc = []
    Lb = []
    p_c_b = []
    p_c_b_err = []

    for i in range(nz):
        dat = np.loadtxt(hmf_files[i][2])
        #CORRECTIONS TO PLANCK (or selected) COSMOLOGY -- STUPID LITTLE H; setting h=0.6704
        h = H0/100.
        dat[:,0] = dat[:,0]/h
        dat[:,1] = dat[:,1]*h**3

        l_zmin = hmf_files[i][0]
        l_zmax = hmf_files[i][1]
        print >> sys.stderr, i, l_zmin, l_zmax
        myz[i] = (float(l_zmin)+float(l_zmax))/2.

        #Get the approximation to the halo mass function
        mass_param[i] = fit_plm.nm_approx_third(dat[:,0],dat[:,1],Mpiv)

        #Get the full list of files
        dfiles = glob(indir+"dist_cen_sat_bright_z_"+l_zmin+"_"+l_zmax+"_lm_*.dat")
        dfiles.sort()
        dfiles = np.array(dfiles)
        errfiles = glob(indir+"dist_err_cen_sat_bright_z_"+l_zmin+"_"+l_zmax+"_lm_*.dat")
        errfiles.sort()
        errfiles = np.array(errfiles)

        Lc.append([])
        Lb.append([])
        p_c_b.append([])
        p_c_b_err.append([])

        #Make the generic input data vectors for this particular redshift
        for j in range(len(glist)):
            pdat = np.loadtxt(dfiles[glist[j]])
            perr_dat = np.loadtxt(errfiles[glist[j]])
            lumbins = pdat[:,0]
            pdat = pdat[:,1:]
            perr_dat = perr_dat[:,1:]

            nbins = len(lumbins)

            #Turn these into 1D arrays
            Lc_temp = np.zeros(nbins*nbins)
            Lb_temp = np.zeros(nbins*nbins)
            p_temp = np.zeros(nbins*nbins)
            perr_temp = np.zeros(nbins*nbins)
            for k in range(nbins):
                for l in range(nbins):
                    Lc_temp[k*nbins+l] = lumbins[k]
                    Lb_temp[k*nbins+l] = lumbins[l]
                    p_temp[k*nbins+l] = pdat[k,l]
                    perr_temp[k*nbins+l] = perr_dat[k,l]
                        
            #Trim away all zero values
            clist = np.where(p_temp > 1e-2)[0]

            #And append to the relevant arrays
            #print len(Lc), j
            Lc[i].append(Lc_temp[clist])
            Lb[i].append(Lb_temp[clist])
            p_c_b[i].append(p_temp[clist])
            p_c_b_err[i].append(np.sqrt(perr_temp[clist]))

    return [ myz, Lc, Lb, p_c_b, p_c_b_err, mass_param ]

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
    indir = input_params['indir_uber']
    outdir = input_params['outdir']
    area = float(input_params['area'])

    #Make the main output directory
    os.system("mkdir -p "+outdir)

    #Note glist has been added to the readin; also add the lm_med values
    glist = input_params['glist']
    lm_med = input_params['lm_med']

    #Run data setup for every redshift bin
    Mpiv = 2.35e14
    [myz, Lc, Lb, p_c_b, p_c_b_err, mass_param] = make_full_data_vector(Mpiv,indir,input_params['hmf_files'],glist,lm_med,H0=float(input_params['H0']),omegaM=float(input_params['omegaM']),omegaL=float(input_params['omegaL']))

    #Count number of data points
    npoints = 0
    for i in range(len(p_c_b)):
        for j in range(len(p_c_b[i])):
            npoints = npoints + len(p_c_b[i][j])

    print >> sys.stderr, "Total number of data points: ", npoints
    print >> sys.stderr, "Data readin complete"

    #Fixed lambda-mass relationship parameter values
    #Mpiv A_lm B_lm lnlm0
    #Now updated to take input read in from parameter file
    sigma_lm = float(input_params['sigma_lm'])
    lm_param = [Mpiv, float(input_params['A_lm']), float(input_params['B_lm']), float(input_params['lnlm0']), sigma_lm]

    print >> sys.stderr, "Using scatter of: ",sigma_lm," equivalent to ",(np.exp(sigma_lm)-1)*100.,"%"
    #Initial parameter guess
    #Parameters are:
    #sigma_L r lnLc0 A_c B_c sigma_b A_sb r_lmb r_c lnLb0 A_b A_c s
    #start = [0.36, -0.9, 24.74, 0.40, 1.21, 0.3, -0.05, 0., 0., 24.34, 0.4, 1.2, 2.]
    start = [0.2765, 0.76, 24.6165, 0.4305, 1.465, 0.2234, -0.1733, -0.3364, -0.3128, 24.353, 0.3373, 1.523, 4.03]
    #Note that s is a nuisance parameter -- scales the covariance matrices
    nparam = len(start)

    #Log(probability) TEST
    print "ln(P) TEST: ",fit_cen_bsat.p_cen_bsat_likelihood(start,mass_param,lm_param,
                                                            sigma_lm,Lc,Lb,p_c_b,
                                                            p_c_b_err,myz,lm_med,npoints),npoints

    #Now, we do the setup for emcee
    #How many walkers?  Starting with 50 as default
    nwalkers = 50

    #Now, drawn randoms from a Gaussian distribution around our fiducial start point
    #First, note the fiducial distribution widths around this point
    err_guess = [0.05, 0.2, 0.02, 0.05, 0.1, 0.05, 0.02, 0.2, 0.2, 0.02, 0.05, 0.1, 0.2]
    #Note that this produces the initial "ball" of walkers
    walk_start = np.zeros([nwalkers, nparam])
    for i in range(nparam):
        walk_start[:,i] = np.random.normal(start[i],err_guess[i],nwalkers)

    #Make the sampler:
    sampler = emcee.EnsembleSampler(nwalkers, nparam, fit_cen_bsat.p_cen_bsat_likelihood,
                                    args=[mass_param,lm_param,sigma_lm,Lc,Lb,p_c_b,
                                          p_c_b_err,myz,lm_med,npoints])

    #Run 20 steps with 50 walkers for burn-in
    nsteps_burn = 20
    print >> sys.stderr, "Beginning burn-in with ",nwalkers, " walkers and ",nsteps_burn," steps"
    pos, prob, state = sampler.run_mcmc(walk_start, nsteps_burn)
    sampler.reset()

    print >> sys.stderr, "First part of MCMC done.  Starting main segment..."
    #Currently running with 500 steps for the main segment
    nsteps = 400

    #Now print out the resulting chain to a file
    f = open(outdir+"chain_cen_bsat_all.dat",'w')
    f.close()
    
    for result in sampler.sample(pos, iterations=nsteps, storechain=False):
        position = result[0]
        prob = result[1]
        f = open(outdir+"chain_cen_bsat_all.dat",'a')
        for i in range(nwalkers):
            for j in range(nparam):
                f.write(str(position[i][j])+" ")
            f.write(str(prob[i])+"\n")
        f.close()

    #print >> sys.stderr, "Acceptance fraction is: ", sampler.acceptance_fraction()
    print >> sys.stderr, "We're all through here."
