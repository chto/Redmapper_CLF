#!/u/ki/dapple/bin/python

import numpy as np
import scipy
import scipy.special
import scipy.optimize
import sys
from glob import glob
import time

import fit_with_covar
import fit_clf
import fit_plm
import mag_convert

#Various functions for working with satellite parameter fits

#Calculation of covariance between phi_* and lambda
def get_r_phi(param,param_sat,lambda_val):
    #For ease of tracking parameters
    sigma_lm = param[0]
    sigma_L = param[1]
    r = param[2]
    lnlm0 = param[3]
    A_lm = param[4]
    lnL0 = param[5]
    A_L = param[6]

    #Satellite parameters
    lnphi0 = param_sat[0]
    A_phi = param_sat[1]
    lnLs0 = param_sat[2]
    A_s = param_sat[3]
    B_s = param_sat[4]

    r_phi = 1/np.sqrt(1+1./lambda_val/sigma_lm**2)

    return r_phi

#Calculation of mean L* at fixed redshift, lambda
def mean_Lst(param,param_sat,lambda_val,z,A,beta1,beta2,beta3):
    #For ease of tracking parameters
    sigma_lm = np.sqrt(param[0]**2.+1./lambda_val)
    sigma_L = param[1]
    r = param[2]
    lnlm0 = param[3]
    A_lm = param[4]
    lnL0 = param[5]
    A_L = param[6]

    #Satellite parameters
    lnphi0 = param_sat[0]
    A_phi = param_sat[1]
    lnLs0 = param_sat[2]
    A_s = param_sat[3]
    B_s = param_sat[4]

    lnLst = lnLs0 + B_s*np.log(1+z)+(A_lm*A_s*(np.log(lambda_val)-lnlm0)-A_s*beta1*sigma_lm**2)/(A_lm**2 + beta2*sigma_lm**2)
    return lnLst/np.log(10.)

#Calculation of mean phi* at fixed redshift, lambda
def mean_phist(param,param_sat,lambda_val,z,A,beta1,beta2,beta3):
    #For ease of tracking parameters
    sigma_lm = np.sqrt(param[0]**2. + 1./lambda_val)
    sigma_L = param[1]
    r = param[2]
    lnlm0 = param[3]
    A_lm = param[4]
    lnL0 = param[5]
    A_L = param[6]

    #Satellite parameters
    lnphi0 = param_sat[0]
    A_phi = param_sat[1]
    lnLs0 = param_sat[2]
    A_s = param_sat[3]
    B_s = param_sat[4]

    r_phi = get_r_phi(param,param_sat,lambda_val)
    lnphi = lnphi0 + ( (A_lm*A_phi + beta2*sigma_lm**2*r_phi)*(np.log(lambda_val)-lnlm0) + A_lm*beta1*r_phi*sigma_lm**2 - A_phi*beta1*sigma_lm**2)/(A_lm**2 + beta2*sigma_lm**2)

    return np.exp(lnphi)

#Calculation of cutoff magnitude m* that Eli uses
def mst_eli(zbins):
    zlist = np.where(zbins <= 0.5)
    alist = np.where(zbins > 0.5)
    mst[zlist] = 22.44 + 3.36*np.log(zbins[zlist]) + 0.273*np.log(zbins[zlist])**2-0.0618*np.log(zbins[zlist])**3 - 0.0227*np.log(zbins[zlist])**4
    mst[alist] = 22.94+3.08*np.log(zbins[alist]) - 11.22*np.log(zbins[alist])**2 -27.11*np.log(zbins[alist])**3-18.02*np.log(zbins[alist])**4
    return mst

#Calculation of Lmin as a function of redshift --
#Note that this just reads in the results already produced in kcorrect at z=0.3 and converts to
#solar luminosities; then it interpolates to get the desired results
def Lmin_eli(z,zvals=[],Lmin_val=[],abs_solar=4.67966):
    if len(zvals) == 0:
        dat = np.loadtxt("mst_eli_table.dat")
        zvals = dat[:,0]
        mst = dat[:,1]

        Lmin_val = np.log10(mag_convert.mag_to_Lsolar(mst,use_des=1,abs_solar=abs_solar))+np.log10(0.2)

    Lmin = np.interp(z,zvals,Lmin_val)

    return Lmin

#Function for making scipy spit out results for gamma(a,x) with a < 0
def gamma_neg(a,x):
    if a > 0:
        gamma = scipy.special.gammaincc(a,x)*scipy.special.gamma(a)
    else:
        gamma = (gamma_neg(a+1,x) - x**a*np.exp(-x))/a
    return gamma

#Function for solving for alpha
def func_alpha(alpha,data):
    Lmin = data[0]
    Lst = data[1]
    phist = data[2]
    lambda_val = data[3]

    fac = 10.**(Lmin-Lst)

    delta = (lambda_val - 1) - phist/np.log(10.)*gamma_neg(alpha+1,fac)
    
    return abs(delta)

#Calculation of alpha from Lmin, L*, phi*, lambda, z
def get_alpha(Lmin, Lst, phist, lambda_val):
    #Need to solve lambda-1 = phist/ln10 Gamma(alpha, Lmin/Lst)
    data = [Lmin, Lst, phist, lambda_val]

    [alpha] = scipy.optimize.fmin(func_alpha,[-0.95],args=([data]),disp=False)

    return alpha
    #return -1.
                                                
#Given input parameters and lambda, z, calculate the parameters describing the
#Satellite CLF schecter function
def get_clf_params(lambda_val,z,param,param_sat,mass_param,
                        zvals=[],Lmin_val=[],fix_alpha=True):
    #note mass_param is A, beta1, beta2, beta3

    Lst = mean_Lst(param,param_sat,lambda_val,z,mass_param[0],mass_param[1],mass_param[2],mass_param[3])
    
    phist = mean_phist(param,param_sat,lambda_val,z,mass_param[0],mass_param[1],mass_param[2],mass_param[3])

    if fix_alpha:
        #Get Lmin for this redshift
        Lmin = Lmin_eli(z,zvals=zvals,Lmin_val=Lmin_val)
        #print Lmin, Lst, phist, lambda_val
    
        #Get alpha given lambda, Lst, phist, Lmin
        alpha = get_alpha(Lmin, Lst, phist, lambda_val)
    else:
        alpha = -1.

    clf_param = np.array([Lst, phist, alpha])

    return clf_param

#chi^2 for a single z, lambda bin, relative to measured parameters
def get_chi2_psat_single(lambda_val,z,param,param_sat,mass_param,y,covar,
                         zvals=[],Lmin_val=[]):
    #note mass_param is A, beta1, beta2, beta3
    y_vec = get_clf_params(lambda_val,z,param,param_sat,mass_param,
                           zvals=zvals,Lmin_val=Lmin_val)
    delta = y_vec - y
    #print y_vec, delta

    chi2 = np.dot(np.dot(delta,np.linalg.inv(covar)),delta)

    return chi2

#chi^2 for a single z, lambda bin, relative to the full CLF
#Note that this assumes the input data is already appropriately trimmed
def get_chi2_psat_single_clf(lambda_val,z,param,param_sat,mass_param,x,y,covar,
                        zvals=[],Lmin_val=[]):
    #First, get the CLF parameters
    clf_param = get_clf_params(lambda_val,z,param,param_sat,mass_param,
                               zvals=zvals,Lmin_val=Lmin_val)

    y_fit = fit_clf.func_sat(clf_param,x)
    
    delta = y_fit - y

    chi2 = np.dot(np.dot(delta,np.linalg.inv(covar)),delta)
    #print chi2
    return chi2

#Chi^2 value for a set of z, lambda satellite CLFs -- uses full CLF data 
#input in x,y,covar parameters
#Currently set for use with one of the fit_with_covar chi^2 minimization schemes
def get_chi2_psat_clf_set(param_sat,lambda_val,z,param,mass_param,x,y,covar,
                          zvals,Lmin_val,ibanderr):
    chi2 = 0.
    if len(ibanderr) == 0:
        ibanderr = np.zeros_like(z)
    
    for i in range(len(x)):
        chi2 = chi2 + get_chi2_psat_single_clf(lambda_val[i],z[i],param,param_sat,mass_param[i],
                                               x[i],y[i],covar[i],zvals=zvals,Lmin_val=Lmin_val+ibanderr[i])

    return np.resize(np.array([chi2]),len(param_sat))

#Function wrapper for performing fits with sets of full satellite CLF data
def fit_psat_wrapper(param_sat,lambda_val,z,param,mass_param,x,y,covar,zvals,Lmin_val):
    [res, cov_x, infodict, msg, ier] = scipy.optimize.leastsq(get_chi2_psat_clf_set,param_sat,
                                                              args=(lambda_val,z,param,mass_param,x,y,covar,zvals,Lmin_val),
                                                              full_output=True)

    print >> sys.stderr, ier, msg
    #Get the chi^2 for this result
    chi2 = get_chi2_psat_clf_set(res,lambda_val,z,param,mass_param,x,y,covar,zvals,Lmin_val)[0]
    
    #Note -- currently not bothering with error estimates
    res_covar = []

    return [chi2, res, res_covar]

#Try to fit this model to the data
#Note that this also reads in and sets up the data being analyzed
def fit_psat_set(infiles,covfiles,zmin,zmax,lmed,param,mass_param,Lcut=9.8,start=[],zvals=[],Lmin_cut=[]):
    if len(infiles) != len(covfiles):
        print >> sys.stderr, "ERROR: Data and covariance files do not match"
        sys.exit(1)

    zmid = (zmin+zmax)/2.
    #Empty data vectors
    x = []
    y = []
    covar_all = []
    Lmin_val = []
    npoints = 0
    chi2 = 0
    lmed_new = []
    for i in range(len(infiles)):
        sat = np.loadtxt(infiles[i])
        covar = np.loadtxt(covfiles[i])

        #Method of getting mean lambda for a bin -- total the galaxies!
        #Note this assumes "extended" galaxies not included
        dL = sat[1,0]-sat[0,0]
        slist = np.where( sat[:,1] > 0)
        lmed_new.append(np.sum(sat[slist,1])*dL)
        #Note that I'm using the Lmin from above to define a cutoff
        Lmin = Lmin_eli(zmax[i],zvals=zvals,Lmin_val=Lmin_cut)
        slist = np.where( (sat[:,1] > 0.01) & (sat[:,0] > Lmin+dL) & (sat[:,0] > Lcut) )[0]
        #slist = slist[2:]
        #print sat[slist[0],0]
        x.append(sat[slist,0])
        y.append(sat[slist,1])
        covar = covar[:,slist]
        covar = covar[slist,:]
        covar_all.append(covar)   

        npoints = npoints + len(slist)

    lmed_new = np.array(lmed_new)
    #Now do the fitting
    if len(start)==0:
        start = np.array([np.log(24.), 0.95, 9.96*np.log(10.),1., 1.])
    
    if len(zvals)==0:
        zvals = np.array(range(1000))*0.001+0.0005
        Lmin_cut = Lmin_eli(zvals,abs_solar=4.71493)

    [chi2, res, res_covar] = fit_psat_wrapper(start,lmed_new+1,zmid,param,mass_param,x,y,covar_all,
                                              zvals,Lmin_cut)
    #if chi2 > npoints-5:
    #    [chi2, res, res_covar] = fit_psat_wrapper(res,lmed,zmid,param,mass_param,x,y,covar_all,
    #                                              zvals,Lmin_val)

    return [npoints, chi2, res, res_covar, x, y, covar_all, lmed_new]
    #return [npoints, chi2, res, res_covar, x, y, covar_all]

#Test fitting of fixed CLFs to data in a single bin
def func_clf_twoparam(param,x):
    Lmin = x[0] #Note that Lmin included w/ data
    lambda_val = x[1]
    
    Lst = param[0]
    phist = param[1]
    
    alpha = get_alpha(Lmin, Lst, phist, lambda_val)
    
    y = fit_clf.func_sat([Lst, phist, alpha],x[2:])
    
    return y

#Fit CLF data in a single bin
def fit_clf_twoparam(infile,covfile,Lmin,lambda_val,start=[]):
    
    #Read input data first
    sat = np.loadtxt(infile)
    covar = np.loadtxt(covfile)

    slist = np.where( (sat[:,0] > Lmin+0.08) & (sat[:,1] > 0.01) )[0]
    slist = slist[3:]
    sat = sat[slist]
    covar = covar[slist,:]
    covar = covar[:,slist]
    #covar = fit_with_covar.regularize(covar)
    #covar = np.diag(np.diag(covar))

    x = np.zeros(len(slist)+2)
    x[0] = Lmin
    x[1] = lambda_val
    x[2:] = sat[:,0]
    y = sat[:,1]
    
    mystart = start
    if len(start) == 0:
        mystart = [10.2, 40]
    if len(start) > 2:
        mystart = start[:2]

    [chi2, res, res_covar] = fit_with_covar.fit_with_covar(mystart,func_clf_twoparam,x,y,covar)
    if chi2 > len(y):
        [chi2, res, res_covar] = fit_with_covar.fit_with_covar(res,func_clf_twoparam,x,y,covar)
    alpha = get_alpha(Lmin, res[0], res[1], lambda_val)

    print len(y), fit_with_covar.get_chisq_with_covar(start,fit_clf.func_sat,x[2:],y,covar)[0], fit_with_covar.get_chisq_with_covar(res,func_clf_twoparam,x,y,covar)[0]

    return [len(y), chi2, res, res_covar, alpha]

#Likelihood for a set of z, lambda binned satellite CLF fits

#Making a full data set for input into the chi2 overall function
def make_data_vector(indir,indir_uber,l_zmin,l_zmax,Lcut=0,glist=[0,1,3,4,5,6,7]):
    #Read in all the relevant data
    satfiles = glob(indir+"clf_sat_z_"+l_zmin+"_"+l_zmax+"*.dat")
    uberfiles = glob(indir_uber+"clf_sat_z_"+l_zmin+"_"+l_zmax+"*.dat")
    covfiles = glob(indir_uber+"clf_sat_covar_z_"+l_zmin+"_"+l_zmax+"*.dat")
    satfiles.sort()
    uberfiles.sort()
    covfiles.sort()
    satfiles=np.array(satfiles)
    uberfiles = np.array(uberfiles)
    covfiles = np.array(covfiles)

    #Set up the output vectors
    x = []
    y = []
    covar = []
    lmed = []

    #Loop over the selected richness ranges
    for i in range(len(glist)):
        sat = np.loadtxt(satfiles[glist[i]])
        sat_uber = np.loadtxt(uberfiles[glist[i]])
        covar_t = np.loadtxt(covfiles[glist[i]])

        slist = np.where(sat[:,1] > 0.01)[0]
        lmed.append(np.sum(sat[slist,1])*(sat[1,0]-sat[0,0])+1.)

        slist = np.where( (sat_uber[:,1] > 0.01) & (sat_uber[:,0] > Lcut) )[0]
        #Always drop the first point -- too close to boundary to include regardless of cuts
        slist = slist[1:]
        x.append(sat_uber[slist,0])
        y.append(sat_uber[slist,1])
        
        covar_t = covar_t[:,slist]
        covar_t = covar_t[slist,:]

        covar.append(covar_t)

    return [x, y, covar, lmed]

#Function for spitting out likelihood -- Note this is just -chi^2/2. for satellite fits
def psat_likelihood(sat_param,lm_param,cen_param,mass_param,x,y,covar,myz,lmed):
    '''
    Note that this is basic version; alpha is currently set to float, not fixed or
    determined by lambda value
    '''
    nz = len(mass_param)
    nbins = len(x)
    nlm = nbins/nz

    Mpiv = lm_param[0]
    A_lm = lm_param[1]
    B_lm = lm_param[2]
    lnlm0 = lm_param[3]

    alpha = sat_param[5]
    if len(sat_param)<=6:
        s = 1.
    else:
        s = sat_param[6] #parameter for scaling covariance matrices
    
    in_param = np.zeros(8)
    in_param[0] = 0.1842 #sigma_lm
    in_param[1] = cen_param[0] #sigma_L
    in_param[2] = cen_param[1] #r
    in_param[3] = lnlm0 + B_lm*np.log(1+0) #log(lm0)
    in_param[4] = A_lm
    in_param[5] = cen_param[2] #ln L0
    in_param[6] = cen_param[3] #A_L
    in_param[7] = cen_param[4] #B_L
    B_L = cen_param[4]

    logp = 0.
    for i in range(nz):
        in_param[3] = lnlm0 + B_lm*np.log(1+myz[i*nlm])
        for j in range(nlm):
            local_sat_param = get_clf_params(lmed[i*nlm+j],myz[i*nlm+j],in_param,sat_param,
                                                      mass_param[i],fix_alpha=False)
            local_sat_param[2] = alpha
            delta = y[nlm*i+j] - fit_clf.func_sat(local_sat_param,x[nlm*i+j])
            chi2 = np.dot( np.dot(delta,np.linalg.inv(covar[nlm*i+j]*s)),delta )
            #print i,nlm*i+j,myz[i*nlm+j],lmed[i*nlm+j], local_sat_param, chi2
            logp = logp + chi2

    #Convert to likelihood
    logp = -logp/2.

    #Account for our covariance matrix scaling
    logp = logp - np.log(s)
    return logp


#Convolution testing

#2D Gaussian for phi-lambda distribution
def p_of_lambda_phi(lm_val, phi, M, lm_param, sigma_lm, sat_param, z):
    #Note pivot and other mass-lambda parameters
    Mpiv = lm_param[0]
    A_lm = lm_param[1]
    B_lm = lm_param[2]
    lnlm0 = lm_param[3]

    #Note that sigma_lm is currently a single input parameter
    my_sigma_lm = sigma_lm + 1./lm_val

    #Relevant satellite parameters
    lnphi0 = sat_param[0]
    A_phi = sat_param[1]

    #Mean value of lambda
    lnlmM = lnlm0 + A_lm*np.log(M/Mpiv)+B_lm*np.log(1+z)

    #Mean value of phi
    lnphi = lnphi0 + A_phi*np.log(M/Mpiv)

    #Correlation coefficient
    r_phi = 1./np.sqrt(1+1./lm_val/sigma_lm**2.)

    #Determinant of covariance matrix
    detC = my_sigma_lm**2*sigma_lm**2*(1-r_phi**2)

    #Getting the value(s) from the 2D gaussian
    p = 1./(2.*np.pi*np.sqrt(detC))*np.exp( -( (np.log(lm_val)-lnlmM )**2/my_sigma_lm**2
                                              -2*r_phi*(np.log(lm_val)-lnlmM)*(np.log(phi)-lnphi)/my_sigma_lm/sigma_lm
                                              +(np.log(phi)-lnphi)**2/sigma_lm**2 )/(2.*(1-r_phi**2)) )

    return p



#Correctly operating convolution function; note we need the desired range in lambda as well
def func_sat_convolved(L,lm_min,lm_max,lm_param,sigma_lm,sat_param,mass_param,z,mval,nval):
    clf = 0*L
    
    #Note the pivot and other mass-lambda parameters
    Mpiv = lm_param[0]
    A_lm = lm_param[1]
    B_lm = lm_param[2]
    lnlm0 = lm_param[3]

    #Note input lambda range
    n_lm_bins = 10
    dlm = np.log(lm_max/lm_min)/n_lm_bins
    lm_bins = lm_min*np.exp( dlm/2. + np.array(range(n_lm_bins))*dlm)
    
    #Note estimated lnphi0 range
    nphi = 20
    dlnphi = np.log(36)/nphi
    phibins = np.log(10.)+np.array(range(nphi))*dlnphi+dlnphi/2.
    #Make the larger phibins matrix
    mat_phibins = np.tile(np.repeat(np.exp(phibins),len(mval)),n_lm_bins).reshape(n_lm_bins,nphi,len(mval))

    #Make the mass function matrices
    mf_matrix = np.tile(nval,n_lm_bins*nphi).reshape(n_lm_bins,nphi,len(mval))
    mat_mval = np.tile(mval,n_lm_bins*nphi).reshape(n_lm_bins,nphi,len(mval))
    #Make the lm_bins matrix
    mat_lm = np.repeat(lm_bins,nphi*len(mval)).reshape(n_lm_bins,nphi,len(mval))

    #Now pick up P(lambda, phi | M)
    plm_phi = p_of_lambda_phi(mat_lm, mat_phibins, mat_mval, lm_param, sigma_lm, sat_param, z)

    #Get the denominator -- this is the same for all L
    p_denom = np.sum( plm_phi*mf_matrix )

    #Get the L* value at the given halo mass (Same for every mass)
    Lst = np.tile( np.exp(sat_param[2] + sat_param[3]*np.log(mval/Mpiv) + sat_param[4]*np.log(1+z)) , n_lm_bins*nphi).reshape(n_lm_bins,nphi,len(mval))
    alpha = sat_param[5]
    #Now, get Phi(L, M, phi) for each value of phi
    #Alternate version -- look Ma, no loops!
    Lst = np.repeat(Lst,len(L)).reshape(n_lm_bins, nphi, len(mval), len(L))
    mat_phibins = np.repeat(mat_phibins, len(L)).reshape(n_lm_bins, nphi, len(mval), len(L))
    L_input = np.tile(10.**L,n_lm_bins*nphi*len(mval)).reshape(n_lm_bins, nphi, len(mval), len(L))
    ratio = (L_input/Lst)
    phi_init = mat_phibins*(ratio)**(alpha+1)*np.exp(-ratio)

    clf = np.zeros_like(L)
    clf = np.sum(np.sum(np.sum(phi_init * np.repeat(plm_phi * mf_matrix, len(L)).reshape(n_lm_bins, nphi, len(mval), len(L)),0),0),0)/p_denom

    #print p_denom
    return clf

def time_test(L,lm_min,lm_max,lm_param,sigma_lm,sat_param,mass_param,z,mval,nval):
    start= time.clock()
    clf = func_sat_convolved_v2(L,lm_min,lm_max,lm_param,sigma_lm,sat_param,mass_param,z,mval,nval)
    print "Time needed was ",time.clock()-start," s, val test: ", clf[0]
    return


#Likelihood version that uses the convolved CLF
#Note that this requires lm_min, lm_max ranges, as well as the halo mass function inputs
def psat_likelihood_conv(sat_param,lm_param,cen_param,mass_param,x,y,covar,myz,args):
    '''
    Note that this is basic version; alpha is currently set to float, not fixed or
    determined by lambda value
    '''
    #Parameter limits check -- note that this is necessary for use with emcee
    if len(sat_param) >= 7:
        if sat_param[6] < 0:
            return -np.inf

    nz = len(mass_param)
    nbins = len(x)
    nlm = nbins/nz

    #Taking apart the lumped argument
    lm_min = args[0]
    lm_max = args[1]
    mf_set = args[2]
    npoints = args[3]

    Mpiv = lm_param[0]
    A_lm = lm_param[1]
    B_lm = lm_param[2]
    lnlm0 = lm_param[3]
    if len(lm_param) > 4:
        sigma_lm = lm_param[4]
    else:
        sigma_lm = 0.1842

    alpha = sat_param[5]
    if len(sat_param)<=6:
        s = 1.
    else:
        s = sat_param[6] #parameter for scaling covariance matrices

    logp = 0.
    for i in range(nz):
        for j in range(nlm):
            #Now do the convolution
            clf = func_sat_convolved(x[nlm*i+j], lm_min[j], lm_max[j], 
                                     lm_param, sigma_lm, sat_param, mass_param[i], 
                                     myz[nlm*i+j], mf_set[i][:,0], mf_set[i][:,1])
            delta = y[nlm*i+j] - clf
            chi2 = np.dot( np.dot(delta,np.linalg.inv(covar[nlm*i+j]*s)),delta )
            #print myz[nlm*i+j], lm_min[j], chi2, len(myz)

            logp = logp + chi2

    #Convert to likelihood
    logp = -logp/2.

    #Account for our covariance matrix scaling
    logp = logp - npoints*np.log(s)/2.
    return logp


#Version of the clf satellite function that includes evolution in alpha with redshift
def func_sat_convolved_a_ev(L,lm_min,lm_max,lm_param,sigma_lm,sat_param,mass_param,z,mval,nval):
    clf = 0*L
    
    #Note the pivot and other mass-lambda parameters
    Mpiv = lm_param[0]
    A_lm = lm_param[1]
    B_lm = lm_param[2]
    lnlm0 = lm_param[3]

    #Note input lambda range
    n_lm_bins = 10
    dlm = np.log(lm_max/lm_min)/n_lm_bins
    lm_bins = lm_min*np.exp( dlm/2. + np.array(range(n_lm_bins))*dlm)
    
    #Note estimated lnphi0 range
    nphi = 20
    dlnphi = np.log(36)/nphi
    phibins = np.log(10.)+np.array(range(nphi))*dlnphi+dlnphi/2.
    #Make the larger phibins matrix
    mat_phibins = np.tile(np.repeat(np.exp(phibins),len(mval)),n_lm_bins).reshape(n_lm_bins,nphi,len(mval))

    #Make the mass function matrices
    mf_matrix = np.tile(nval,n_lm_bins*nphi).reshape(n_lm_bins,nphi,len(mval))
    mat_mval = np.tile(mval,n_lm_bins*nphi).reshape(n_lm_bins,nphi,len(mval))
    #Make the lm_bins matrix
    mat_lm = np.repeat(lm_bins,nphi*len(mval)).reshape(n_lm_bins,nphi,len(mval))

    #Now pick up P(lambda, phi | M)
    plm_phi = p_of_lambda_phi(mat_lm, mat_phibins, mat_mval, lm_param, sigma_lm, sat_param, z)

    #Get the denominator -- this is the same for all L
    p_denom = np.sum( plm_phi*mf_matrix )

    #Get the L* value at the given halo mass (Same for every mass)
    Lst = np.tile( np.exp(sat_param[2] + sat_param[3]*np.log(mval/Mpiv) + sat_param[4]*np.log(1+z)) , n_lm_bins*nphi).reshape(n_lm_bins,nphi,len(mval))
    alpha = sat_param[5]+sat_param[6]*z
    #Now, get Phi(L, M, phi) for each value of phi
    #Alternate version -- look Ma, no loops!
    Lst = np.repeat(Lst,len(L)).reshape(n_lm_bins, nphi, len(mval), len(L))
    mat_phibins = np.repeat(mat_phibins, len(L)).reshape(n_lm_bins, nphi, len(mval), len(L))
    L_input = np.tile(10.**L,n_lm_bins*nphi*len(mval)).reshape(n_lm_bins, nphi, len(mval), len(L))
    ratio = (L_input/Lst)
    phi_init = mat_phibins*(ratio)**(alpha+1)*np.exp(-ratio)

    clf = np.zeros_like(L)
    clf = np.sum(np.sum(np.sum(phi_init * np.repeat(plm_phi * mf_matrix, len(L)).reshape(n_lm_bins, nphi, len(mval), len(L)),0),0),0)/p_denom

    #print p_denom
    return clf

#Version of the likelihood that includes an extra term in sat_param for evolution in alpha
def psat_likelihood_conv_a(sat_param,lm_param,cen_param,mass_param,x,y,covar,myz,args):
    '''
    Note that this is a version that explicitly allows alpha evolution; 
    alpha is currently set to float, not fixed or
    determined by lambda value
    '''
    #Parameter limits check -- note that this is necessary for use with emcee
    if len(sat_param) >= 7:
        if sat_param[-1] < 0:
            return -np.inf

    nz = len(mass_param)
    nbins = len(x)
    nlm = nbins/nz

    #Taking apart the lumped argument
    lm_min = args[0]
    lm_max = args[1]
    mf_set = args[2]
    npoints = args[3]

    Mpiv = lm_param[0]
    A_lm = lm_param[1]
    B_lm = lm_param[2]
    lnlm0 = lm_param[3]

    alpha = sat_param[5]
    if len(sat_param)<=6:
        s = 1.
    else:
        s = sat_param[len(sat_param)-1] #parameter for scaling covariance matrices

    logp = 0.
    for i in range(nz):
        for j in range(nlm):
            #Now do the convolution
            clf = func_sat_convolved_a_ev(x[nlm*i+j], lm_min[j], lm_max[j], 
                                          lm_param, 0.1842, sat_param, mass_param[i], 
                                          myz[nlm*i+j], mf_set[i][:,0], mf_set[i][:,1])
            delta = y[nlm*i+j] - clf
            chi2 = np.dot( np.dot(delta,np.linalg.inv(covar[nlm*i+j]*s)),delta )
            #print myz[nlm*i+j], lm_min[j], chi2, len(myz)

            logp = logp + chi2

    #Convert to likelihood
    logp = -logp/2.

    #Account for our covariance matrix scaling
    logp = logp - npoints*np.log(s)/2.
    return logp

#Version of the clf satellite function that includes evolution in ln phi_0 with redshift
def func_sat_convolved_phi_ev(L,lm_min,lm_max,lm_param,sigma_lm,sat_param,mass_param,z,mval,nval):
    clf = 0*L
    
    #Note the pivot and other mass-lambda parameters
    Mpiv = lm_param[0]
    A_lm = lm_param[1]
    B_lm = lm_param[2]
    lnlm0 = lm_param[3]

    #Note input lambda range
    n_lm_bins = 10
    dlm = np.log(lm_max/lm_min)/n_lm_bins
    lm_bins = lm_min*np.exp( dlm/2. + np.array(range(n_lm_bins))*dlm)
    
    #Note estimated lnphi0 range
    nphi = 20
    dlnphi = np.log(36)/nphi
    phibins = np.log(10.)+np.array(range(nphi))*dlnphi+dlnphi/2.
    #Make the larger phibins matrix
    mat_phibins = np.tile(np.repeat(np.exp(phibins),len(mval)),n_lm_bins).reshape(n_lm_bins,nphi,len(mval))

    #Make the mass function matrices
    mf_matrix = np.tile(nval,n_lm_bins*nphi).reshape(n_lm_bins,nphi,len(mval))
    mat_mval = np.tile(mval,n_lm_bins*nphi).reshape(n_lm_bins,nphi,len(mval))
    #Make the lm_bins matrix
    mat_lm = np.repeat(lm_bins,nphi*len(mval)).reshape(n_lm_bins,nphi,len(mval))

    #Correct lnphi0 for evolution
    sat_param_in = np.copy(sat_param)
    sat_param_in[0] = sat_param[0] + sat_param[6]*np.log(1+z)
    #Now pick up P(lambda, phi | M)
    plm_phi = p_of_lambda_phi(mat_lm, mat_phibins, mat_mval, lm_param, sigma_lm, sat_param_in, z)

    #Get the denominator -- this is the same for all L
    p_denom = np.sum( plm_phi*mf_matrix )

    #Get the L* value at the given halo mass (Same for every mass)
    Lst = np.tile( np.exp(sat_param[2] + sat_param[3]*np.log(mval/Mpiv) + sat_param[4]*np.log(1+z)) , n_lm_bins*nphi).reshape(n_lm_bins,nphi,len(mval))
    alpha = sat_param[5]
    #Now, get Phi(L, M, phi) for each value of phi
    #Alternate version -- look Ma, no loops!
    Lst = np.repeat(Lst,len(L)).reshape(n_lm_bins, nphi, len(mval), len(L))
    mat_phibins = np.repeat(mat_phibins, len(L)).reshape(n_lm_bins, nphi, len(mval), len(L))
    L_input = np.tile(10.**L,n_lm_bins*nphi*len(mval)).reshape(n_lm_bins, nphi, len(mval), len(L))
    ratio = (L_input/Lst)
    phi_init = mat_phibins*(ratio)**(alpha+1)*np.exp(-ratio)

    clf = np.zeros_like(L)
    clf = np.sum(np.sum(np.sum(phi_init * np.repeat(plm_phi * mf_matrix, len(L)).reshape(n_lm_bins, nphi, len(mval), len(L)),0),0),0)/p_denom

    #print p_denom
    return clf

#Version of the likelihood that includes an extra term in sat_param for evolution in phi
def psat_likelihood_conv_phi(sat_param,lm_param,cen_param,mass_param,x,y,covar,myz,args):
    '''
    Note that this is a version that explicitly allows alpha evolution; 
    alpha is currently set to float, not fixed or
    determined by lambda value
    '''
    #Parameter limits check -- note that this is necessary for use with emcee
    if len(sat_param) >= 7:
        if sat_param[-1] < 0:
            return -np.inf

    #Telling everything that super-high normalization fails
    if sat_param[0] > 10:
        return -np.inf
    if sat_param[0] < -4:
        return -np.inf

    nz = len(mass_param)
    nbins = len(x)
    nlm = nbins/nz

    #Taking apart the lumped argument
    lm_min = args[0]
    lm_max = args[1]
    mf_set = args[2]
    npoints = args[3]

    if len(lm_param) == 5:
        Mpiv = lm_param[0]
        A_lm = lm_param[1]
        B_lm = lm_param[2]
        lnlm0 = lm_param[3]
        sigma_lm = lm_param[4]
        lm_param_temp = lm_param
    else:
        #Data array for lm_param --
        #Randomly select from the input lm_param array
        lm_param_temp = lm_param[np.random.randint(len(lm_param))]
        Mpiv = lm_param_temp[0]
        A_lm = lm_param_temp[1]
        B_lm = lm_param_temp[2]
        lnlm0 = lm_param_temp[3]
        sigma_lm = lm_param_temp[4] 
        

    if len(sat_param)<=6:
        s = 1.
    else:
        s = sat_param[len(sat_param)-1] #parameter for scaling covariance matrices

    logp = 0.
    for i in range(nz):
        for j in range(nlm):
            #Now do the convolution
            clf = func_sat_convolved_phi_ev(x[nlm*i+j], lm_min[j], lm_max[j], 
                                            lm_param_temp, sigma_lm, sat_param, mass_param[i], 
                                            myz[nlm*i+j], mf_set[i][:,0], mf_set[i][:,1])
            delta = y[nlm*i+j] - clf
            chi2 = np.dot( np.dot(delta,np.linalg.inv(covar[nlm*i+j]*s)),delta )
            #print myz[nlm*i+j], lm_min[j], chi2, len(myz)

            logp = logp + chi2

    #Convert to likelihood
    logp = -logp/2.

    #Account for our covariance matrix scaling
    logp = logp - npoints*np.log(s)/2.
    return logp



##Version that adds evolution in both phist and alpha
#Likelihood version that uses the convolved CLF
#Note that this requires lm_min, lm_max ranges, as well as the halo mass function inputs
def psat_likelihood_conv_ev_all(sat_param,lm_param,cen_param,mass_param,x,y,covar,myz,args):
    '''
    Note that this is basic version; alpha is currently set to float, not fixed or
    determined by lambda value
    '''
    #Parameter limits check -- note that this is necessary for use with emcee
    if sat_param[-1] < 0:
        return -np.inf

    nz = len(mass_param)
    nbins = len(x)
    nlm = nbins/nz

    #Taking apart the lumped argument
    lm_min = args[0]
    lm_max = args[1]
    mf_set = args[2]
    npoints = args[3]

    if len(lm_param) <= 5:
        Mpiv = lm_param[0]
        A_lm = lm_param[1]
        B_lm = lm_param[2]
        lnlm0 = lm_param[3]
        if len(lm_param) > 4:
            sigma_lm = lm_param[4]
        else:
            sigma_lm = 0.1842
        lm_param_temp = lm_param
    else:
        #Data array for lm_param --
        #Randomly select from the input lm_param array
        lm_param_temp = lm_param[np.random.randint(len(lm_param))]
        Mpiv = lm_param_temp[0]
        A_lm = lm_param_temp[1]
        B_lm = lm_param_temp[2]
        lnlm0 = lm_param_temp[3]
        sigma_lm = lm_param_temp[4] 

    alpha = sat_param[6]
    if len(sat_param)<=8:
        s = 1.
    else:
        s = sat_param[-1] #parameter for scaling covariance matrices

    logp = 0.
    for i in range(nz):
        for j in range(nlm):
            #Set up accounting for redshift evolution
            lnphi0 = sat_param[0] + sat_param[2]*np.log(1+myz[nlm*i+j])
            alpha = sat_param[6] + sat_param[7]*myz[nlm*i+j]
            sat_param_in = [lnphi0, sat_param[1], sat_param[3], sat_param[4], 
                            sat_param[5], alpha,s]
        
            #Now do the convolution
            clf = func_sat_convolved(x[nlm*i+j], lm_min[j], lm_max[j], 
                                            lm_param_temp, sigma_lm, sat_param_in, mass_param[i], 
                                            myz[nlm*i+j], mf_set[i][:,0], mf_set[i][:,1])
            delta = y[nlm*i+j] - clf
            chi2 = np.dot( np.dot(delta,np.linalg.inv(covar[nlm*i+j]*s)),delta )
            #print myz[nlm*i+j], lm_min[j], chi2, len(myz)

            logp = logp + chi2

    #Convert to likelihood
    logp = -logp/2.

    #Account for our covariance matrix scaling
    logp = logp - npoints*np.log(s)/2.
    return logp


#Likelihood version that uses the convolved CLF
#Note that this requires lm_min, lm_max ranges, as well as the halo mass function inputs
def psat_likelihood_conv_noev(sat_param,lm_param,cen_param,mass_param,x,y,covar,myz,args):
    '''
    Note that this is basic version; alpha is currently set to float, not fixed or
    determined by lambda value

    Note that this version removes ALL redshift evolution
    '''
    #Parameter limits check -- note that this is necessary for use with emcee
    if len(sat_param) >= 6:
        if sat_param[5] < 0:
            return -np.inf
    #Telling everything that super-high normalization fails
    if sat_param[0] > 10:
        return -np.inf
    if sat_param[0] < -4:
        return -np.inf

    nz = len(mass_param)
    nbins = len(x)
    nlm = nbins/nz

    #Taking apart the lumped argument
    lm_min = args[0]
    lm_max = args[1]
    mf_set = args[2]
    npoints = args[3]

    if len(lm_param) <= 5:
        Mpiv = lm_param[0]
        A_lm = lm_param[1]
        B_lm = lm_param[2]
        lnlm0 = lm_param[3]
        if len(lm_param) > 4:
            sigma_lm = lm_param[4]
        else:
            sigma_lm = 0.1842
        lm_param_temp = lm_param
    else:
        #Data array for lm_param --
        #Randomly select from the input lm_param array
        lm_param_temp = lm_param[np.random.randint(len(lm_param))]
        Mpiv = lm_param_temp[0]
        A_lm = lm_param_temp[1]
        B_lm = lm_param_temp[2]
        lnlm0 = lm_param_temp[3]
        sigma_lm = lm_param_temp[4] 

    alpha = sat_param[4]
    if len(sat_param)<=5:
        s = 1.
    else:
        s = sat_param[5] #parameter for scaling covariance matrices

    logp = 0.
    #Reset input satellite parameters to remove redshift evolution
    my_sat_param = [sat_param[0], sat_param[1], sat_param[2], sat_param[3], 0., sat_param[4], sat_param[5] ]

    for i in range(nz):
        for j in range(nlm):
            #Now do the convolution
            clf = func_sat_convolved(x[nlm*i+j], lm_min[j], lm_max[j], 
                                     lm_param_temp, sigma_lm, my_sat_param, mass_param[i], 
                                     myz[nlm*i+j], mf_set[i][:,0], mf_set[i][:,1])
            delta = y[nlm*i+j] - clf
            chi2 = np.dot( np.dot(delta,np.linalg.inv(covar[nlm*i+j]*s)),delta )
            #print myz[nlm*i+j], lm_min[j], chi2, len(myz)

            logp = logp + chi2

    #Convert to likelihood
    logp = -logp/2.

    #Account for our covariance matrix scaling
    logp = logp - npoints*np.log(s)/2.

    #Return -np.inf if something went wrong and gave us a NaN
    if np.isnan(logp):
        return -np.inf
    
    return logp



#Functions for running with corrections to the satellite schechter function
def schechter_corr_dr8(L,matrix=False):
    '''
    Input must be log10(L/Lst) for the desired schechter function
    '''
    
    fac = np.zeros_like(L)
    
    x0 = 0.5
    fac = 24*(L-x0)**4+1.
    if matrix:
        xlist = np.where(L < x0)
        fac[xlist] = 1.
        mlist = np.where(fac > 10)
        fac[mlist] = 10.
    else:
        xlist = np.where(L < x0)[0]
        if len(xlist) > 0:
            fac[xlist] = 1.
        mlist = np.where(fac > 10)[0]
        if len(mlist) > 0:
            fac[mlist] = 0*mlist+10.

    return fac

#Provide the corrected CLF, at fixed halo mass and redshift
def func_sat_fixed_mass_phi_ev(L,Mvir,lm_param,sat_param,mass_param,z):
    #Note the pivot and other mass-lambda parameters
    Mpiv = lm_param[0]
    A_lm = lm_param[1]
    B_lm = lm_param[2]
    lnlm0 = lm_param[3]

    #Parameter values
    Lst = sat_param[2] + sat_param[3]*np.log(10.**Mvir/Mpiv) + sat_param[4]*np.log(1+z)
    alpha = sat_param[5]
    phist = np.exp(sat_param[0] + sat_param[1]*np.log(10.**Mvir/Mpiv)+ sat_param[6]*np.log(1+z) )
    ratio = 10.**L/np.exp(Lst)
    clf = phist*ratio**(alpha+1)*np.exp(-ratio)

    #Now get the correction
    fac = schechter_corr_dr8(np.log10(ratio))

    return clf*fac

#Get the corrected CLF, at fixed redshift and a range in richness
def func_sat_conv_sch_corr(L,lm_min,lm_max,lm_param,sigma_lm,sat_param,mass_param,z,mval,nval):

    clf = 0*L
    
    #Note the pivot and other mass-lambda parameters
    Mpiv = lm_param[0]
    A_lm = lm_param[1]
    B_lm = lm_param[2]
    lnlm0 = lm_param[3]

    #Note input lambda range
    n_lm_bins = 10
    dlm = np.log(lm_max/lm_min)/n_lm_bins
    lm_bins = lm_min*np.exp( dlm/2. + np.array(range(n_lm_bins))*dlm)
    
    #Note estimated lnphi0 range
    nphi = 20
    dlnphi = np.log(36)/nphi
    phibins = np.log(10.)+np.array(range(nphi))*dlnphi+dlnphi/2.
    #Make the larger phibins matrix
    mat_phibins = np.tile(np.repeat(np.exp(phibins),len(mval)),n_lm_bins).reshape(n_lm_bins,nphi,len(mval))

    #Make the mass function matrices
    mf_matrix = np.tile(nval,n_lm_bins*nphi).reshape(n_lm_bins,nphi,len(mval))
    mat_mval = np.tile(mval,n_lm_bins*nphi).reshape(n_lm_bins,nphi,len(mval))
    #Make the lm_bins matrix
    mat_lm = np.repeat(lm_bins,nphi*len(mval)).reshape(n_lm_bins,nphi,len(mval))

    #Correct lnphi0 for evolution
    sat_param_in = np.copy(sat_param)
    sat_param_in[0] = sat_param[0] + sat_param[6]*np.log(1+z)
    #Now pick up P(lambda, phi | M)
    plm_phi = p_of_lambda_phi(mat_lm, mat_phibins, mat_mval, lm_param, sigma_lm, sat_param_in, z)

    #Get the denominator -- this is the same for all L
    p_denom = np.sum( plm_phi*mf_matrix )

    #Get the L* value at the given halo mass (Same for every mass)
    Lst = np.tile( np.exp(sat_param[2] + sat_param[3]*np.log(mval/Mpiv) + sat_param[4]*np.log(1+z)) , n_lm_bins*nphi).reshape(n_lm_bins,nphi,len(mval))
    alpha = sat_param[5]
    #Now, get Phi(L, M, phi) for each value of phi
    #Alternate version -- look Ma, no loops!
    Lst = np.repeat(Lst,len(L)).reshape(n_lm_bins, nphi, len(mval), len(L))
    mat_phibins = np.repeat(mat_phibins, len(L)).reshape(n_lm_bins, nphi, len(mval), len(L))
    L_input = np.tile(10.**L,n_lm_bins*nphi*len(mval)).reshape(n_lm_bins, nphi, len(mval), len(L))
    ratio = (L_input/Lst)
    phi_init = mat_phibins*(ratio)**(alpha+1)*np.exp(-ratio)
    #Get the desired correction factor to the schechter function
    #fac = 0*phi_init
    #for i in range(len(fac)):
    #    for j in range(len(fac[i])):
    #        for k in range(len(fac[i,j])):
    #            fac[i,j,k] = schechter_corr_dr8(np.log10(ratio[i,j,k]))
    fac = schechter_corr_dr8(np.log10(ratio),matrix=True)
    phi_init = fac*phi_init

    clf = np.zeros_like(L)
    clf = np.sum(np.sum(np.sum(phi_init * np.repeat(plm_phi * mf_matrix, len(L)).reshape(n_lm_bins, nphi, len(mval), len(L)),0),0),0)/p_denom

    #print p_denom
    return clf


#New likelihood function, using the corrected non-schechter CLF
def psat_likelihood_conv_schcorr(sat_param,lm_param,cen_param,mass_param,x,y,covar,myz,args):
    '''
    Note that this is a version that explicitly allows phi* evolution; 
    empircal correction to schechter function included    
    '''
    #Parameter limits check -- note that this is necessary for use with emcee
    if len(sat_param) >= 7:
        if sat_param[-1] < 0:
            return -np.inf

    nz = len(mass_param)
    nbins = len(x)
    nlm = nbins/nz

    #Taking apart the lumped argument
    lm_min = args[0]
    lm_max = args[1]
    mf_set = args[2]
    npoints = args[3]

    if len(lm_param) <= 5:
        Mpiv = lm_param[0]
        A_lm = lm_param[1]
        B_lm = lm_param[2]
        lnlm0 = lm_param[3]
        sigma_lm = lm_param[4]
        lm_param_temp = lm_param
    else:
        #Data array for lm_param --
        #Randomly select from the input lm_param array
        lm_param_temp = lm_param[np.random.randint(len(lm_param))]
        Mpiv = lm_param_temp[0]
        A_lm = lm_param_temp[1]
        B_lm = lm_param_temp[2]
        lnlm0 = lm_param_temp[3]
        sigma_lm = lm_param_temp[4] 

    if len(sat_param)<=6:
        s = 1.
    else:
        s = sat_param[len(sat_param)-1] #parameter for scaling covariance matrices

    logp = 0.
    for i in range(nz):
        for j in range(nlm):
            #Now do the convolution
            clf = func_sat_conv_sch_corr(x[nlm*i+j], lm_min[j], lm_max[j], 
                                         lm_param_temp, sigma_lm, sat_param, mass_param[i], 
                                         myz[nlm*i+j], mf_set[i][:,0], mf_set[i][:,1])
            delta = y[nlm*i+j] - clf
            chi2 = np.dot( np.dot(delta,np.linalg.inv(covar[nlm*i+j]*s)),delta )
            #print myz[nlm*i+j], lm_min[j], chi2, len(myz)

            logp = logp + chi2

    #Convert to likelihood
    logp = -logp/2.

    #Account for our covariance matrix scaling
    logp = logp - npoints*np.log(s)/2.
    return logp

#Setup for yet another version of the likelihood; this one takes an addition beta
#parameter to handle non-schechter behavior in the satellite CLF at
#the bright end
def func_sat_conv_sch_beta(L,lm_min,lm_max,lm_param,sigma_lm,sat_param,mass_param,z,mval,nval):

    clf = 0*L
    
    #Note the pivot and other mass-lambda parameters
    Mpiv = lm_param[0]
    A_lm = lm_param[1]
    B_lm = lm_param[2]
    lnlm0 = lm_param[3]

    #Note input lambda range
    n_lm_bins = 10
    dlm = np.log(lm_max/lm_min)/n_lm_bins
    lm_bins = lm_min*np.exp( dlm/2. + np.array(range(n_lm_bins))*dlm)
    
    #Note estimated lnphi0 range
    nphi = 20
    dlnphi = np.log(36)/nphi
    phibins = np.log(10.)+np.array(range(nphi))*dlnphi+dlnphi/2.
    #Make the larger phibins matrix
    mat_phibins = np.tile(np.repeat(np.exp(phibins),len(mval)),n_lm_bins).reshape(n_lm_bins,nphi,len(mval))

    #Make the mass function matrices
    mf_matrix = np.tile(nval,n_lm_bins*nphi).reshape(n_lm_bins,nphi,len(mval))
    mat_mval = np.tile(mval,n_lm_bins*nphi).reshape(n_lm_bins,nphi,len(mval))
    #Make the lm_bins matrix
    mat_lm = np.repeat(lm_bins,nphi*len(mval)).reshape(n_lm_bins,nphi,len(mval))

    #Correct lnphi0 for evolution
    sat_param_in = np.copy(sat_param)
    sat_param_in[0] = sat_param[0] + sat_param[6]*np.log(1+z)
    #Now pick up P(lambda, phi | M)
    plm_phi = p_of_lambda_phi(mat_lm, mat_phibins, mat_mval, lm_param, sigma_lm, sat_param_in, z)

    #Get the denominator -- this is the same for all L
    p_denom = np.sum( plm_phi*mf_matrix )

 #Get the L* value at the given halo mass (Same for every mass)
    Lst = np.tile( np.exp(sat_param[2] + sat_param[3]*np.log(mval/Mpiv) + sat_param[4]*np.log(1+z)) , n_lm_bins*nphi).reshape(n_lm_bins,nphi,len(mval))
    alpha = sat_param[5]
    #Now, get Phi(L, M, phi) for each value of phi
    #Alternate version -- look Ma, no loops!
    Lst = np.repeat(Lst,len(L)).reshape(n_lm_bins, nphi, len(mval), len(L))
    mat_phibins = np.repeat(mat_phibins, len(L)).reshape(n_lm_bins, nphi, len(mval), len(L))
    L_input = np.tile(10.**L,n_lm_bins*nphi*len(mval)).reshape(n_lm_bins, nphi, len(mval), len(L))
    ratio = (L_input/Lst)
    #Note extra exponent inside the schechter-like function
    beta = sat_param[7]
    phi_init = mat_phibins*(ratio)**(alpha+1)*np.exp(-ratio**beta)

    clf = np.zeros_like(L)
    clf = np.sum(np.sum(np.sum(phi_init * np.repeat(plm_phi * mf_matrix, len(L)).reshape(n_lm_bins, nphi, len(mval), len(L)),0),0),0)/p_denom

    #print p_denom
    return clf


#And the new likelihood
def psat_likelihood_sch_beta(sat_param,lm_param,cen_param,mass_param,x,y,covar,myz,args):
    '''
    Note that this is a version that explicitly allows phi* evolution; 
    Schechter function is expected by having exp(-ratio**beta) in exponential
    '''
    #Parameter limits check -- note that this is necessary for use with emcee
    if len(sat_param) >= 8:
        if sat_param[-1] < 0:
            return -np.inf
    
        nz = len(mass_param)
    nbins = len(x)
    nlm = nbins/nz

    #Taking apart the lumped argument
    lm_min = args[0]
    lm_max = args[1]
    mf_set = args[2]
    npoints = args[3]

    if len(lm_param) <= 5:
        Mpiv = lm_param[0]
        A_lm = lm_param[1]
        B_lm = lm_param[2]
        lnlm0 = lm_param[3]
        sigma_lm = lm_param[4]
        lm_param_temp = lm_param
    else:
        #Data array for lm_param --
        #Randomly select from the input lm_param array
        lm_param_temp = lm_param[np.random.randint(len(lm_param))]
        Mpiv = lm_param_temp[0]
        A_lm = lm_param_temp[1]
        B_lm = lm_param_temp[2]
        lnlm0 = lm_param_temp[3]
        sigma_lm = lm_param_temp[4] 

    if len(sat_param)<=7:
        s = 1.
    else:
        s = sat_param[len(sat_param)-1] #parameter for scaling covariance matrices

    logp = 0.
    for i in range(nz):
        for j in range(nlm):
            #Now do the convolution
            clf = func_sat_conv_sch_beta(x[nlm*i+j], lm_min[j], lm_max[j], 
                                         lm_param_temp, sigma_lm, sat_param, mass_param[i], 
                                         myz[nlm*i+j], mf_set[i][:,0], mf_set[i][:,1])
            delta = y[nlm*i+j] - clf
            chi2 = np.dot( np.dot(delta,np.linalg.inv(covar[nlm*i+j]*s)),delta )
            #print myz[nlm*i+j], lm_min[j], chi2, len(myz)

            logp = logp + chi2

    #Convert to likelihood
    logp = -logp/2.

    #Account for our covariance matrix scaling
    logp = logp - npoints*np.log(s)/2.
    return logp

#Likelihood using Schechter function, but holding all mass scaling to fixed values
def psat_likelihood_fix_mass(sat_param,lm_param,cen_param,mass_param,x,y,covar,myz,args):
    '''
    Note that this is a version that holds mass dependence fixed
    '''
    #Parameter limits check -- note that this is necessary for use with emcee
    if len(sat_param) >= 6:
        if sat_param[-1] < 0:
            return -np.inf

    #Telling everything that super-high normalization fails
    if sat_param[0] > 10:
        return -np.inf
    if sat_param[0] < -4:
        return -np.inf

    nz = len(mass_param)
    nbins = len(x)
    nlm = nbins/nz

    #Taking apart the lumped argument
    lm_min = args[0]
    lm_max = args[1]
    mf_set = args[2]
    npoints = args[3]

    if len(lm_param) <= 5:
        Mpiv = lm_param[0]
        A_lm = lm_param[1]
        B_lm = lm_param[2]
        lnlm0 = lm_param[3]
        sigma_lm = lm_param[4]
        lm_param_temp = lm_param
    else:
        #Data array for lm_param --
        #Randomly select from the input lm_param array
        lm_param_temp = lm_param[np.random.randint(len(lm_param))]
        Mpiv = lm_param_temp[0]
        A_lm = lm_param_temp[1]
        B_lm = lm_param_temp[2]
        lnlm0 = lm_param_temp[3]
        sigma_lm = lm_param_temp[4] 

    if len(sat_param)<=6:
        s = 1.
    else:
        s = sat_param[len(sat_param)-1] #parameter for scaling covariance matrices

    #Set up the input parameter array -- needs additional values
    sat_param_in = [sat_param[0],0.846,sat_param[1],0.,
                    sat_param[2],sat_param[3],sat_param[4],s]

    logp = 0.
    for i in range(nz):
        for j in range(nlm):
            #Now do the convolution
            clf = func_sat_convolved_phi_ev(x[nlm*i+j], lm_min[j], lm_max[j], 
                                            lm_param_temp, sigma_lm, sat_param_in, mass_param[i], 
                                            myz[nlm*i+j], mf_set[i][:,0], mf_set[i][:,1])
            delta = y[nlm*i+j] - clf
            chi2 = np.dot( np.dot(delta,np.linalg.inv(covar[nlm*i+j]*s)),delta )
            #print myz[nlm*i+j], lm_min[j], chi2, len(myz)

            logp = logp + chi2

    #Convert to likelihood
    logp = -logp/2.

    #Account for our covariance matrix scaling
    logp = logp - npoints*np.log(s)/2.
    
    return logp

#ln(Prior) on lambda-mass relationship -- ND Gaussian
def prior_lm(lm_param,lm_param_mean,lm_cov):
    detC = np.linalg.det(lm_cov)

    p = 1/(2*np.pi)**1.5/np.sqrt(detC)*np.exp( -0.5*np.dot( lm_param-lm_param_mean,np.dot(np.linalg.inv(lm_cov), lm_param-lm_param_mean)) )

    return np.log(p)

