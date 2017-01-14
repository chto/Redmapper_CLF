#!/u/ki/dapple/bin/python

import numpy as np
import sys

import fit_with_covar
import fit_plm
import cosmo
import mcmc_errors as mcmc

#command-line script for running a full set of P(lambda,L|M) fits
#For each redshift bin of interest

if __name__ == "__main__":
    #Inputs to command line should be:
    #indir zmin zmax outdir
    if len(sys.argv) < 7:
        print >> sys.stderr, "ERROR: Required input format is"
        print >> sys.stderr, "       mffile indir zmin zmax area outdir"

    mffile = sys.argv[1]
    indir = sys.argv[2]
    l_zmin = sys.argv[3]
    l_zmax = sys.argv[4]
    area = float(sys.argv[5])
    outdir = sys.argv[6]

    lm_min = 0

    #Do all the initial setup and file reads
    #Setting up the n(z)
    dat = np.loadtxt(mffile)
    Mpiv = 10.**14.2
    [A, b1, b2, b3] = fit_plm.nm_approx_third(dat[:,0],dat[:,1],Mpiv)
    myz = float(l_zmax)
    #B_L = 0.993*np.log10(np.exp(1))
    B_L = 0.

    #Start by reading in the biggest clusters list
    lambda_big = np.loadtxt(indir+"lambda_max_list_z_"+l_zmin+"_"+l_zmax+".dat")

    #Calculate the volume
    #Note that this assumes the Planck cosmology, h=1
    volume = cosmo.comoving_volume(float(l_zmin),float(l_zmax))*(area/41253.)

    #Initial parameter guess
    start = [0.373, 0.376, -0.0936, 2.915, 0.837, 24.690, 0.4]
    #Adjust the starting L estimate based on redshift
    start[5] = start[5] + 0.993*np.log10(np.exp(1))*np.log(1+float(l_zmax))
    nparam = len(start)
    #Limits on parameter values
    p_limits = [ [0, 0.6], 
                 [0, 1e6],
                 [-1, 1],
                 [-1e6, 1e6],
                 [-1e6, 1e6],
                 [-1e6, 1e6],
                 [-1e6, 1e6] ]

    #Fit test to get x, y values easily
    #[npoints, chi2, res, res_covar, x, y, covar_all] = fit_plm.fit_plm(indir,'0.2','0.25',start,A,b1,b2,b3,Mpiv,myz,B_L,func_fit=fit_plm.func_plm_alt)
    #Construction of the necessary data vectors for the binned part
    print indir
    [x,y,covar_all] = fit_plm.make_data_vectors(indir,l_zmin,l_zmax,A,b1,b2,b3,Mpiv,
                                                myz,B_L,lm_max=lambda_big[-1],
                                                lm_min=lm_min,glist=[0,1,3,4,5,6,7])
    nlm_pts = x[7]

    #Initial covariance guess
    covar_start = np.zeros([nparam,nparam])
    covar_start[0,0] = 0.005
    covar_start[1,1] = 0.0007
    covar_start[2,2] = 0.01
    covar_start[3,3] = 0.004
    covar_start[4,4] = 0.0002
    covar_start[5,5] = 0.006
    covar_start[6,6] = 0.0002

    stepfac = 0.4/np.sqrt(nparam)

    p_covar = np.zeros_like(covar_start)

    #Run 1000 elements to get better covariance guess
    chain = np.array([start])
    p_start = fit_plm.plm_likelihood(start,x,y,covar_all,lambda_big,volume)
    
    p_arr = [p_start]
    param = np.copy(start)
    p = np.copy(p_start)
    print >> sys.stderr, "Starting first part of MCMC"
    for i in range(1000):
        [param, p] = mcmc.take_mcmc_step_tail(param,p_limits,p,covar_start*stepfac**2,fit_plm.plm_likelihood,x,y,covar_all,lambda_big,volume)
        chain = np.append(chain,[param],axis=0)
        p_arr.append(p)
        
    print >> sys.stderr, len(chain)

    #Get a fresh version of covar_start based on these values
    p_covar = mcmc.get_pcovar(chain[1:1000])

    #Run full chain and print results
    nmcmc = 100000
    f = open(outdir+"chain_z_"+l_zmin+"_"+l_zmax+".dat",'w')
    for i in range(nmcmc):
        [param, p] = mcmc.take_mcmc_step_tail(param,p_limits,p,covar_start*stepfac**2,fit_plm.plm_likelihood,x,y,covar_all,lambda_big,volume)
        chain = np.append(chain,[param],axis=0)
        p_arr.append(p)
        print >> sys.stderr, i
        if i>1000:
            p_covar = mcmc.get_pcovar(chain[1001:])
        print >> f, chain[i+1001,0], chain[i+1001,1], chain[i+1001,2], chain[i+1001,3], chain[i+1001,4], chain[i+1001,5], chain[i+1001,6], p_arr[i+1001]
    f.close()

    #Write out the best-fit point, median, and the current covariance matrix
    f = open(outdir+"param_z_"+l_zmin+"_"+l_zmax+".dat",'w')
    place = np.argmax(p_arr)
    print >> f, chain[place,0], chain[place,1], chain[place,2], chain[place,3], chain[place,4],chain[place,5],chain[place,6]
    med_param = np.median(chain[10000:],axis=0)
    print >> f, med_param[0], med_param[1], med_param[2], med_param[3], med_param[4],med_param[5],med_param[6]
    for i in range(len(p_covar)):
        print >> f, p_covar[i,0], p_covar[i,1], p_covar[i,2], p_covar[i,3], p_covar[i,4],p_covar[i,5],p_covar[i,6]

    f.close()
