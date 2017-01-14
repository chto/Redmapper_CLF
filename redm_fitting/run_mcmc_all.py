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
        print >> sys.stderr, "ERROR: No input data directory"
        sys.exit(1)
    if keys.count('outdir')==0:
        print >> sys.stderr, "WARNING: Ouputing to local directory"
        setup.append(('outdir',''))
    if keys.count('glist')==0:
        print >> sys.stderr, "WARNING: Using default glist (DR8)"
        setup.append(('glist',['0','1','3','4','5','6','7']))
        
    params = dict(setup)
    #Convert the glist key to a list of integers
    glist_temp = params['glist']
    for i in range(len(glist_temp)):
        glist_temp[i] = int(glist_temp[i])
    params['glist'] = glist_temp

    return params

def make_full_data_vector(Mpiv,indir,area,hmf_files,glist,lm_min=0):
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
        dat[:,0] = dat[:,0]*h
        dat[:,1] = dat[:,1]/h**3

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
        sigma2_R[i] = cosmo.sigma_sq_matter_perturb( (volume[i]*3./4./np.pi)**(1./3.)/h,k,Pk,h=1.)

        #Make the generic input data vectors for this particular redshift
        #Note that entry 6 in the x vector is B_L, which will need to be changed at each step in the chain
        [xtemp, ytemp, covtemp] = fit_plm.make_data_vectors(indir,l_zmin,l_zmax,At,b1t,b2t,b3t,
                                                            Mpiv,myz[i],0,lm_max=lambda_big[i][-1],
                                                            lm_min=lm_min,glist=glist)
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

    #Note that glist has now been added to readin
    glist = input_params['glist']

    #Run data setup for every redshift bin
    Mpiv = 10**14.2
    [x, y, covar_all, lambda_big, myz, volume, sigma2_R] = make_full_data_vector(Mpiv,indir,area,input_params['hmf_files'],glist)

    print >> sys.stderr, "Data readin complete"

    #Initial parameter guess
    #Parameters are:
    #sigma_lm sigma_L r lnlm0 A_lm lnL0 A_L B_L
    #start = [0.2, 0.33, -0.7, 2.6, 1.0, 24.42, 0.4, 1.5]
    #start = [0.011, 0.3667, -0.452, 1.814, 1.305, 24.1127, 0.5187, 1.3785]
    #start = [ 3.46568061e-02 ,  4.22575130e-01 ,  1.26984575e-01 ,  2.19595309e+00 ,  1.15053665e+00  , 2.41357105e+01 ,  4.83577943e-01 ,  1.36921646e+00]
    start = [  0.31732668 ,  0.25354121,  -0.96412498 ,  2.08498327  , 1.23534931 , 24.22358167  , 0.56858281 ,  1.31656603]

    #Limits on parameter values
    p_limits = [ [0, 0.6],
                 [0, 1e6],
                 [-1, 1],
                 [-1e6, 1e6],
                 [-1e6, 1e6],
                 [-1e6, 1e6],
                 [-1e6, 1e6],
                 [-1e6, 1e6] ]
    nparam = len(start)

    #Initial covariance guess
    covar_start = np.zeros([nparam,nparam])
    #covar_start[0,0] = 0.05*0.05
    #covar_start[1,1] = 0.01*0.01
    #covar_start[2,2] = 0.1*0.1
    #covar_start[3,3] = 0.08*0.08
    #covar_start[4,4] = 0.03*0.03
    #covar_start[5,5] = 0.04*0.04
    #covar_start[6,6] = 0.015*0.15
    #covar_start[7,7] = 0.025*0.025
    #covar_start[1,2] = 0.9
    #covar_start[2,1] = 0.9
    #covar_start[3,4] = -0.9
    #covar_start[4,3] = -0.9
    covar_start = [[  8.62769250e-05 ,  1.97043853e-05 ,  1.38460536e-04 , -4.79936720e-05 ,  -1.16726883e-05,  -3.47650177e-05 , -2.52678203e-05  ,-3.34064498e-05],
 [  1.97043853e-05 ,  6.28393334e-06 ,  3.06054081e-05 , -7.60809479e-06 ,  -7.09509812e-07 , -7.45153963e-06 , -3.72630709e-06 , -7.91292794e-06],
 [  1.38460536e-04 ,  3.06054081e-05 ,  2.34212685e-04 , -4.38815742e-05 ,  -1.27884218e-05 , -4.82644427e-05 , -3.24793736e-05 , -5.14106317e-05],
 [ -4.79936720e-05 , -7.60809479e-06 , -4.38815742e-05 ,  1.47824251e-04  ,  3.82206084e-05  , 4.93534840e-05 ,  4.02648454e-05 ,  3.26230998e-05],
 [ -1.16726883e-05 , -7.09509812e-07 , -1.27884218e-05 ,  3.82206084e-05  ,  1.61637044e-05 ,  1.57522179e-05 ,  1.55704000e-06 ,  1.68709907e-05],
 [ -3.47650177e-05 , -7.45153963e-06 , -4.82644427e-05 ,  4.93534840e-05  ,  1.57522179e-05 ,  2.73778227e-05 ,  1.09853416e-05 ,  2.24951653e-05],
 [ -2.52678203e-05,  -3.72630709e-06 , -3.24793736e-05  , 4.02648454e-05   , 1.55704000e-06  , 1.09853416e-05 ,  3.06004059e-05 , -3.32542403e-06],
 [ -3.34064498e-05 , -7.91292794e-06 , -5.14106317e-05  , 3.26230998e-05  ,  1.68709907e-05  , 2.24951653e-05 , -3.32542403e-06 ,  2.96779900e-05]]
    covar_start = np.array(covar_start)

    step_fac = 0.4/np.sqrt(nparam)

    p_covar = np.zeros_like(covar_start)

    #Run 1000 elements to get a better start point/covariance guess
    chain = np.array([start])
    #p_start = fit_plm.plm_like_multiz(start,x,y,covar_all,myz,lambda_big,volume)
    p_start = fit_plm.plm_like_multiz_ev_fix_nl(start,x,y,covar_all,myz,lambda_big,volume,sigma2_R)
    p_arr = [p_start]
    param = np.copy(start)
    p = np.copy(p_start)
    print >> sys.stderr, "Starting first part of MCMC"
    for i in range(1000):
        [param, p] = mcmc.take_mcmc_step_multiz(param,p_limits,p,covar_start*step_fac**2, fit_plm.plm_like_multiz_ev_fix_nl,x,y,covar_all,myz,lambda_big,volume,sigma2_R)
        chain = np.append(chain,[param],axis=0)
        p_arr.append(p)

    print >> sys.stderr, "First part of MCMC done.  Starting main segment..."

    #Get a fresh version of covar_start based on these values
    p_covar = mcmc.get_pcovar(chain[1:1000])
    
    #Run full chain and print results as we go
    nmcmc = 100000
    f = open(outdir+"chain_all.dat",'w')
    for i in range(nmcmc):
        [param, p] = mcmc.take_mcmc_step_multiz(param,p_limits,p,covar_start*step_fac**2, fit_plm.plm_like_multiz_ev_fix_nl,x,y,covar_all,myz,lambda_big,volume,sigma2_R)
        chain = np.append(chain,[param],axis=0)
        p_arr.append(p)
        if i>1000:
            covar_start = mcmc.get_pcovar(chain[1001:])
        print >> f, chain[i+1001,0], chain[i+1001,1], chain[i+1001,2], chain[i+1001,3], chain[i+1001,4], chain[i+1001,5], chain[i+1001,6], chain[i+1001,7], p_arr[i+1001]

    f.close()

    print >> sys.stderr, "We're all through here."
