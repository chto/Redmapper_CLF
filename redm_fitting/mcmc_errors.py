#!/u/ki/dapple/bin/python

import numpy as np
import scipy
import math
import sys

import cosmo
import fit_with_covar
import fit_plm

#Intended to be flexible, but focus is currently on setup and running of 
#MCMC chain for errors estimate in P(lambda,L|M) formalism

#Run a normalization check on the p(L|lambda) values
def pl_norm_check(new_param,x):
    #Sanity check addition to chi2 -- require p_L normalized to 1%
    lumbins = np.array(range(40))*0.08+9
    norm_check = (1-0.08*np.sum(fit_plm.p_L(10.**lumbins,10.,x[0],x[1],x[2],x[3],x[4],new_param,x[5],x[6])))**2*1e4
    norm_check += (1-0.08*np.sum(fit_plm.p_L(10.**lumbins,20.,x[0],x[1],x[2],x[3],x[4],new_param,x[5],x[6])))**2*1e4
    norm_check += (1-0.08*np.sum(fit_plm.p_L(10.**lumbins,60.,x[0],x[1],x[2],x[3],x[4],new_param,x[5],x[6])))**2*1e4

    return norm_check

#Take a single MCMC step in the chain
def take_mcmc_step(param,p_limits,chi2,p_covar,func_fit,x,y,covar):
    #Select a point from the gaussian distribution, with some covariance matrix
    new_param = np.random.multivariate_normal(param,p_covar)
    
    #Check the parameter limits; keep going until we get a valid result
    while(1):
        check = 0
        for i in range(len(param)):
            if (new_param[i] < p_limits[i][0]) | (new_param[i] > p_limits[i][1]):
                check = 1
                break
        #Test that we've got not-too-high sigma(M|lambda)
        #if fit_plm.sigma_m_lm(x[0],x[1],x[2],x[3],x[4],new_param) > 0.92:
        #    check=1
        if check == 1:
            new_param = np.random.multivariate_normal(param,p_covar)
        else:
            break

    #Get the chi2 at the new point
    new_chi2 = fit_with_covar.get_chisq_with_block_covar(new_param,func_fit,x,y,covar)[0]

    #Sanity check addition to chi2 -- require p_L normalized to 1%
    #lumbins = np.array(range(40))*0.08+9
    #norm_check = (1-0.08*np.sum(fit_plm.p_L(10.**lumbins,10.,x[0],x[1],x[2],x[3],x[4],new_param,x[5],x[6])))**2*1e4
    #new_chi2 += norm_check
    #norm_check = (1-0.08*np.sum(fit_plm.p_L(10.**lumbins,20.,x[0],x[1],x[2],x[3],x[4],new_param,x[5],x[6])))**2*1e4
    #new_chi2 += norm_check
    #norm_check = (1-0.08*np.sum(fit_plm.p_L(10.**lumbins,60.,x[0],x[1],x[2],x[3],x[4],new_param,x[5],x[6])))**2*1e4
    #new_chi2 += norm_check
    #new_chi2 += pl_norm_check(new_param,x)

    #print new_chi2, new_param

    #If the new chi2 is less than the old, keep in and take this step
    if new_chi2 < chi2:
        return [new_param, new_chi2]

    #Otherwise, take the step if p is less than the exponential decay
    testval = np.exp( -(new_chi2-chi2)/2. )
    p = np.random.uniform()
    if p > testval:
        return [param, chi2]
    
    return [new_param, new_chi2]

#Version that works with likelihood, and includes "tail" very rich clusters
def take_mcmc_step_tail(param,p_limits,logp,p_covar,func_like,x,y,covar,lambda_big,volume):
    #Select a point from the gaussian distribution, with some covariance matrix
    new_param = np.random.multivariate_normal(param,p_covar)

    #Check parameter limits -- keep going until we get a valid result
    while(1):
        check = 0
        for i in range(len(param)):
            if (new_param[i] < p_limits[i][0]) | (new_param[i] > p_limits[i][1]):
                check = 1
                break
        if check == 1:
            new_param = np.random.multivariate_normal(param,p_covar)
        else:
            break
        
    #Get the new log-likelihood
    new_logp = func_like(new_param,x,y,covar,lambda_big,volume)

    #If new likelihood is higher, take it
    if new_logp > logp:
        return [new_param, new_logp]

    #Otherwise, take the step if p is less than the exponential decay
    testval = np.random.uniform()
    if testval > np.exp(new_logp-logp):
        return [param, logp]

    return [new_param, new_logp]

#Version that works with likelihood, and includes "tail" very rich clusters
#This version also correctly handles the redshift input for the multiz runs
def take_mcmc_step_multiz(param,p_limits,logp,p_covar,func_like,x,y,covar,myz,lambda_big,volume,sigma2_R,npoints):
    #Select a point from the gaussian distribution, with some covariance matrix
    new_param = np.random.multivariate_normal(param,p_covar)

    #Check parameter limits -- keep going until we get a valid result
    while(1):
        check = 0
        for i in range(len(param)):
            if (new_param[i] < p_limits[i][0]) | (new_param[i] > p_limits[i][1]):
                check = 1
                break
        if check == 1:
            new_param = np.random.multivariate_normal(param,p_covar)
        else:
            break
        
    #Get the new log-likelihood
    new_logp = func_like(new_param,x,y,covar,myz,lambda_big,volume,sigma2_R,npoints)

    #If new likelihood is higher, take it
    if new_logp > logp:
        return [new_param, new_logp]

    #Otherwise, take the step if p is less than the exponential decay
    testval = np.random.uniform()
    if testval > np.exp(new_logp-logp):
        return [param, logp]

    return [new_param, new_logp]

#Version that operates on the set of satellite parameters and inputs
def take_mcmc_step_multiz_sat(param,p_limits,logp,p_covar,func_like,lm_param,cen_param,mass_param,x,y,covar,myz,lmed):
    #Select a point from the gaussian distribution, with some covariance matrix
    new_param = np.random.multivariate_normal(param,p_covar)

    #Check parameter limits -- keep going until we get a valid result
    while(1):
        check = 0
        for i in range(len(param)):
            if (new_param[i] < p_limits[i][0]) | (new_param[i] > p_limits[i][1]):
                check = 1
                break
        if check == 1:
            new_param = np.random.multivariate_normal(param,p_covar)
        else:
            break
        
    #Get the new log-likelihood
    new_logp = func_like(new_param,lm_param,cen_param,mass_param,x,y,covar,myz,lmed)

    #If new likelihood is higher, take it
    if new_logp > logp:
        return [new_param, new_logp]

    #Otherwise, take the step if p is less than the exponential decay
    testval = np.random.uniform()
    if testval > np.exp(new_logp-logp):
        return [param, logp]

    return [new_param, new_logp]
    
#Quick estimator of median and one-sigma errors from a chain
def get_errors(chain):
    nparams = len(chain[0])-1
    
    vals = np.percentile(chain[:,0:nparams],50,axis=0)
    e_hi = np.percentile(chain[:,0:nparams],84,axis=0)-vals
    e_lo = vals-np.percentile(chain[:,0:nparams],16,axis=0)

    return vals, e_hi, e_lo

#Get error bounds on a set of input chains, and print results to a file
def get_and_print_errors_cen(indir,zmin,zmax,outfile,noprint=False,
                             has_prior=False):
    #Read in the main chain
    chain = np.loadtxt(indir+"chain_cen_all.dat")
    if has_prior:
        chain = chain[:,3:]

    #Estimate the errors in this chain
    vals, e_hi, e_lo = get_errors(chain[8000:,:])

    #Now, do the same for each of the single redshift bin files,
    #outputting the results as requested
    if not noprint:
        f = open(outfile,'w')
        for i in range(len(zmin)):
            chain = np.loadtxt(indir+"chain_cen_z_"+str(zmin[i])+"_"+str(zmax[i])+".dat")
            if has_prior:
                chain = chain[:,3:]
            vals_i, e_hi_i, e_lo_i = get_errors(chain[5000:])
            f.write( str((zmin[i]+zmax[i])/2.))
            for j in range(len(vals_i)):
                f.write( " "+str(vals_i[j])+" "+str(e_hi_i[j])+" "+str(e_lo_i[j]))
            f.write("\n")
        f.close()
    return vals, e_hi, e_lo


#Get error bounds on a set of input chains, and print results to a file
def get_and_print_errors_sat(indir,zmin,zmax,outfile,has_prior=False):
    #Read in the main chain
    chain = np.loadtxt(indir+"chain_sat_ev_all.dat")

    #Estimate the errors in this chain
    vals, e_hi, e_lo = get_errors(chain[15000:,:])

    #Now, do the same for each of the single redshift bin files,
    #outputting the results as requested
    f = open(outfile,'w')
    for i in range(len(zmin)):
        chain = np.loadtxt(indir+"chain_sat_z_"+str(zmin[i])+"_"+str(zmax[i])+".dat")
        vals_i, e_hi_i, e_lo_i = get_errors(chain[15000:])
        f.write( str((zmin[i]+zmax[i])/2.))
        for j in range(len(vals_i)):
            f.write( " "+str(vals_i[j])+" "+str(e_hi_i[j])+" "+str(e_lo_i[j]))
        f.write("\n")
    f.close()
    return vals, e_hi, e_lo

if __name__ == "__main__":
    #Running the 0.2 < z < 0.25 test case

    #Do all the initial setup and file reads
    #Setting up the n(z)
    dat = np.loadtxt("/afs/slac.stanford.edu/u/ki/rmredd/code/lambda-mass/dndlnm_z_0.25.dat")
    Mpiv = 10.**14.2
    [A, b1, b2, b3] = fit_plm.nm_approx_third(dat[:,0],dat[:,1],Mpiv)
    myz = 0.25
    B_L = 0.993*np.log10(np.exp(1))
    indir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/dr8_zlambda_temp/"

    #Area of this slice in deg^2
    area = 10405. 

    #Total volume for given range in redshift
    #r = cosmo.rco([0.2, 0.25],H0=100.,omegaM=0.317, omegaL = 0.683)
    volume = cosmo.comoving_volume(0.2,0.25,H0=67.04,omegaM=0.317,omegaL=0.683)*(area/41253.)

    #Start by reading in the biggest clusters list
    lambda_big = np.loadtxt(indir+"lambda_max_list_z_0.2_0.25.dat")

    #Initial parameter guess
    #start  = [0.6399, 0.4359, 0.204, 2.632, 0.7821, 24.384, 0.4229]
    start = [0.46, 0.22, -0.6, 2.37, 0.983, 24.5, 0.5]
    nparam = len(start)
    #Limits on parameter values
    p_limits = [ [0, 1e6], 
                 [0, 1e6],
                 [-1, 1],
                 [-1e6, 1e6],
                 [-1e6, 1e6],
                 [-1e6, 1e6],
                 [-1e6, 1e6] ]

    #Fit test to get x, y values easily
    #[npoints, chi2, res, res_covar, x, y, covar_all] = fit_plm.fit_plm(indir,'0.2','0.25',start,A,b1,b2,b3,Mpiv,myz,B_L,func_fit=fit_plm.func_plm_alt)
    #Construction of the necessary data vectors for the binned part
    [x,y,covar_all] = fit_plm.make_data_vectors(indir,'0.2','0.25',A,b1,b2,b3,Mpiv,
                                                myz,B_L,lm_max=lambda_big[-1],nl_only=False)
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
        [param, p] = take_mcmc_step_tail(param,p_limits,p,covar_start*stepfac**2,fit_plm.plm_likelihood,x,y,covar_all,lambda_big,volume)
        chain = np.append(chain,[param],axis=0)
        p_arr.append(p)
        
    print >> sys.stderr, len(chain)

    #Get a fresh version of covar_start based on these values
    p_covar = get_pcovar(chain[1:1000])

    #Run full chain and print results
    nmcmc = 100000
    f = open("chain.dat",'w')
    for i in range(nmcmc):
        [param, p] = take_mcmc_step_tail(param,p_limits,p,covar_start*stepfac**2,fit_plm.plm_likelihood,x,y,covar_all,lambda_big,volume)
        chain = np.append(chain,[param],axis=0)
        p_arr.append(p)
        print >> sys.stderr, i
        if i>1000:
            p_covar = get_pcovar(chain[1001:])
        print >> f, chain[i+1001,0], chain[i+1001,1], chain[i+1001,2], chain[i+1001,3], chain[i+1001,4], chain[i+1001,5], chain[i+1001,6], p_arr[i+1001]
    f.close()

