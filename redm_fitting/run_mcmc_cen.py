#!/u/ki/dapple/bin/python

import numpy as np
import sys
import os

import fit_with_covar
import fit_plm
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

def make_full_data_vector(Mpiv,indir,area,hmf_files,glist,lm_med,lm_min=0):
    nz = len(hmf_files)
    
    myz = np.zeros(nz)
    volume = np.zeros(nz)
    x = []
    y = []
    covar_all = []
    lambda_big = []
    sigma2_R = np.zeros(nz)
    
    for i in range(nz):
        dat = np.loadtxt(hmf_files[i][2])
        #CORRECTIONS TO PLANCK COSMOLOGY -- STUPID LITTLE H; setting h=0.6704
        h = 0.6704
        dat[:,0] = dat[:,0]/h
        dat[:,1] = dat[:,1]*h**3

        l_zmin = hmf_files[i][0]
        l_zmax = hmf_files[i][1]
        print >> sys.stderr, i, l_zmin, l_zmax
        myz[i] = (float(l_zmin)+float(l_zmax))/2.

        #Get the approximation to the halo mass function
        [At, b1t, b2t, b3t] = fit_plm.nm_approx_third(dat[:,0],dat[:,1],Mpiv)

        #Read in the list of biggest clusters
        lambda_big.append(np.loadtxt(indir+"lambda_max_list_z_"+l_zmin+"_"+l_zmax+".dat"))
        
        #Calculate the volume
        #Note that this defaults to Planck cosmology, h=0.67
        volume[i] = cosmo.comoving_volume(float(l_zmin),float(l_zmax))*(area/41253.)

        #Read in and correct the matter power spectrum
        pkdat = np.loadtxt(hmf_files[i][3])
        k = np.exp(pkdat[:,1])
        Pk = np.exp(pkdat[:,0])

        #Calculate sigma(R)^2
        sigma2_R[i] = cosmo.sigma_sq_matter_perturb( (volume[i]*3./4./np.pi)**(1./3.)/h,k,Pk,h=h)

        #Make the generic input data vectors for this particular redshift
        #Note that entry 6 in the x vector is B_L, which will need to be changed at each step in the chain
        [xtemp, ytemp, covtemp] = fit_plm.make_data_vectors(indir,l_zmin,l_zmax,At,b1t,b2t,b3t,
                                                            Mpiv,myz[i],0,lm_max=lambda_big[i][-1],
                                                            lm_min=lm_min,glist=glist,lm_med=lm_med)
        x.append(xtemp)
        y.append(ytemp)
        covar_all.append(covtemp)

    return [x, y, covar_all, lambda_big, myz, volume, sigma2_R]

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
    outdir = input_params['outdir']
    area = float(input_params['area'])

    #Make the main output directory
    os.system("mkdir -p "+outdir)

    #Note glist has been added to the readin; also add the lm_med values
    glist = input_params['glist']
    lm_med = input_params['lm_med']

    #Run data setup for every redshift bin
    Mpiv = 2.35e14
    [x, y, covar_all, lambda_big, myz, volume, sigma2_R] = make_full_data_vector(Mpiv,indir,area,input_params['hmf_files'],glist,lm_med)

    #Count number of data points
    npoints = 0
    for i in range(len(x)):
        npoints = npoints + x[i][8]

    print >> sys.stderr, "Total number of data points: ", npoints
    print >> sys.stderr, "Data readin complete"

    #Fixed lambda-mass relationship parameter values
    #Mpiv A_lm B_lm lnlm0
    #Now updated to take input read in from parameter file
    lm_param = [Mpiv, float(input_params['A_lm']), float(input_params['B_lm']), float(input_params['lnlm0'])]
    #Add these fixed parameters to the x array
    x = [lm_param, x]

    #Check the setup has gone according to plan
    print >> sys.stderr, "SETUP TEST: ", len(x), len(x[0]), len(x[1]), len(x[1][0])
    
    #Initial parameter guess
    #Parameters are:
    #sigma_L r lnL0 A_L B_L s
    #start = [ 0.35,  -0.55 , 24.7  , 0.56858281 ,  1.31656603]
    start = [0.36, -0.55, 24.69, 0.36, 1.42, 1.5]
    #Note that s is a nuisance parameter -- scales the covariance matrices

    #Limits on parameter values
    p_limits = [ [0, 1e6],
                 [-1, 1],
                 [-1e6, 1e6],
                 [-1e6, 1e6],
                 [-1e6, 1e6],
                 [0, 1e6] ]
    nparam = len(start)

    #Initial covariance guess
    covar_start = np.zeros([nparam,nparam])
    covar_start[0,0] = 0.01*0.01
    covar_start[1,1] = 0.1*0.1
    covar_start[2,2] = 0.04*0.04
    covar_start[3,3] = 0.015*0.15
    covar_start[4,4] = 0.025*0.025
    covar_start[5,5] = 0.1*0.1

    step_fac = 0.4/np.sqrt(nparam)

    p_covar = np.zeros_like(covar_start)

    #Run 1000 elements to get a better start point/covariance guess
    chain = np.array([start])
    p_start = fit_plm.plm_like_cen_only(start,x,y,covar_all,myz,lambda_big,volume,sigma2_R,npoints)
    p_arr = [p_start]
    param = np.copy(start)
    p = np.copy(p_start)
    print >> sys.stderr, "Starting first part of MCMC"
    for i in range(1000):
        [param, p] = mcmc.take_mcmc_step_multiz(param,p_limits,p,covar_start*step_fac**2, fit_plm.plm_like_cen_only,x,y,covar_all,myz,lambda_big,volume,sigma2_R,npoints)
        chain = np.append(chain,[param],axis=0)
        p_arr.append(p)

    print >> sys.stderr, "First part of MCMC done.  Starting main segment..."

    #Get a fresh version of covar_start based on these values
    p_covar = mcmc.get_pcovar(chain[1:1000])
    
    #Run full chain and print results as we go
    nmcmc = 100000
    f = open(outdir+"chain_cen_all.dat",'w')
    for i in range(nmcmc):
        [param, p] = mcmc.take_mcmc_step_multiz(param,p_limits,p,covar_start*step_fac**2, fit_plm.plm_like_cen_only,x,y,covar_all,myz,lambda_big,volume,sigma2_R,npoints)
        chain = np.append(chain,[param],axis=0)
        p_arr.append(p)
        if i>1000:
            covar_start = mcmc.get_pcovar(chain[i-1000:])
        print >> f, chain[i+1001,0], chain[i+1001,1], chain[i+1001,2], chain[i+1001,3], chain[i+1001,4], chain[i+1001,5], p_arr[i+1001]

    f.close()

    print >> sys.stderr, "We're all through here."
