#!/usr/bin/python

import numpy as np
import sys

import fit_plm

#Functions necessary for calcuation of a likelihood given a joint distribution
#of centrals and the clusters' brightest satellite
#Note that these are dependent on the Arnault formalism and approximation
#to the halo mass function


#Calculation of P(Lcen,Lbright|lambda)
def p_cen_bsat(Lcen,Lbright,lm_val,z,mass_param,lm_param,sigma_lm,param):

    #Halo mass parameters
    A = mass_param[0]
    beta1 = mass_param[1]
    beta2 = mass_param[2]
    beta3 = mass_param[3] #Not currently using third order part

    #Mass-richness parameters
    Mpiv = lm_param[0]
    A_lm = lm_param[1]
    B_lm = lm_param[2]
    lnlm0 = lm_param[3]

    #sigma_lambda -- a fixed value, with an additional part to handle richness
    s_lm = np.sqrt(sigma_lm**2 + 1./np.exp(lm_val))

    #The rest of the input parameters
    sigma_c = param[0]
    r = param[1]
    lnLc0 = param[2]
    A_c = param[3]
    B_c = param[4]
    sigma_b = param[5]
    A_sb = param[6]
    r_lmb = param[7]
    r_cb = param[8]
    lnLb0 = param[9]
    A_b = param[10]
    B_b = param[11]

    #Adjust the brightest sat scatter for the dependence on richness
    sigma_b = sigma_b + A_sb*(lm_val - lnlm0)
    if sigma_b < 0.01:
        sigma_b = 0.01

    #Setting up the covariance matrix between observables
    cov = np.zeros([3,3])
    cov[0,0] = s_lm*s_lm
    cov[1,1] = sigma_c*sigma_c
    cov[2,2] = sigma_b*sigma_b
    cov[0,1] = r*s_lm*sigma_c
    cov[1,0] = cov[0,1]
    cov[0,2] = r_lmb*s_lm*sigma_b
    cov[2,0] = cov[0,2]
    cov[1,2] = r_cb*sigma_c*sigma_b
    cov[2,1] = cov[1,2]

    invcov = np.linalg.inv(cov)
    #print "TEST: ",sigma_lm, s_lm, np.linalg.det(cov), (1+2*r*r_lmb*r_cb - r**2 - r_lmb**2 - r_cb**2)
    #print cov
    #print s_lm, sigma_c, sigma_b
    #print r, r_lmb, r_cb
    #print np.linalg.det(cov), (1+2*r*r_lmb*r_cb-r*r-r_lmb*r_lmb-r_cb*r_cb)*s_lm*s_lm*sigma_c*sigma_c*sigma_b*sigma_b
    #print cov
    #print invcov
    #print np.dot(cov,invcov)

    #Observables vector
    xvec = np.zeros(3)
    xvec[0] = lm_val - (lnlm0 + B_lm*np.log(1+z))
    xvec[1] = Lcen - (lnLc0 + B_c*np.log(1+z))
    xvec[2] = Lbright - (lnLb0 + B_b*np.log(1+z))
    #print xvec

    #Slopes vector
    avec = np.zeros(3)
    avec[0] = A_lm
    avec[1] = A_c
    avec[2] = A_b

    #Spread in mass at fixed observables
    #print np.dot(avec,np.dot(invcov,avec)), np.dot(xvec,np.dot(invcov,xvec))
    sigma_1 = 1./np.sqrt(np.dot(avec,np.dot(invcov,avec)))

    #Coefficient before the exponential
    coeff = sigma_1*A_lm/2/np.pi/np.sqrt(abs(np.linalg.det(cov)))*np.sqrt((1+beta2*s_lm*s_lm/A_lm/A_lm)/(1+beta2*sigma_1*sigma_1))
    part1 = np.dot(xvec,np.dot(invcov,xvec))
    part2 = (np.dot( avec, np.dot(invcov, xvec )) - beta1)**2*sigma_1*sigma_1/(1+beta2*sigma_1*sigma_1)
    part3 = xvec[0]*xvec[0]/s_lm/s_lm
    part4 = (A_lm*xvec[0] - beta1*s_lm*s_lm/A_lm/A_lm)**2/(1+beta2*s_lm*s_lm/A_lm/A_lm)/(s_lm*s_lm/A_lm/A_lm)

    #print sigma_1, coeff, part1, part2, part3, part4

    #Convert the leading coefficient to handle dlog10 integration
    coeff = coeff*np.log(10.)**2

    p = coeff*np.exp(-0.5*(part1 - part2 - part3 + part4))
       
    return p


#Calculation of the likelihood for a single redshift and lambda bin
def p_cen_bsat_likelihood_single(param,mass_param,lm_param,sigma_lm,
                                 lm_val,z,Lc,Lb,p_c_b,p_c_b_err):
    '''
    Likelihood estimation for p(Lc,Lb) for a single redshift, lambda bin

    Note that Lc, Lb, p_c_b and p_c_b_err must be formated as 1D arrays

    '''

    p_c_b_calc = 0*Lc
    for i in range(len(Lc)):
        p_c_b_calc[i] = p_cen_bsat(Lc[i]*np.log(10.),Lb[i]*np.log(10.),
                                   np.log(lm_val),z,mass_param,lm_param,sigma_lm,param)

    #Now calculate chi^2 -- note not currently using covariance matrix
    chi2 = np.sum( (p_c_b-p_c_b_calc)*(p_c_b-p_c_b_calc)/p_c_b_err/p_c_b_err )

    logp = -chi2/2.

    return logp

#Calculation of the overall likelihood for a given set of redshift, lambda bins
def p_cen_bsat_likelihood(param,mass_param,lm_param,sigma_lm,Lc,Lb,
                          p_c_b,p_c_b_err,myz,lm_val,npoints):
    #Check the input parameters for values that are outside of allowable ranges
    if( (param[0] < 0) | (param[1] < -1) | (param[1] > 1) | (param[5] < 0) | (param[7] < -1) | (param[7] > 1) | (param[8] < -1) | (param[8] > 1) | (param[12] < 0) ):
        return -np.inf

    #This is a more complex requirement on the relations between the correlation parameters
    #That are allowed mathematically
    if 1 + 2*param[1]*param[7]*param[8] - param[1]*param[1] - param[7]*param[7] - param[8]*param[8] <= 0 :
        return -np.inf

    nz = len(myz)
    nlm = len(lm_val)
    logp = 0.
    
    s = param[-1] #Factor to account for possible underestimation of errors

    for i in range(nz):
        for j in range(nlm):
            my_logp = p_cen_bsat_likelihood_single(param,mass_param[i],lm_param,
                                                   sigma_lm,lm_val[j],myz[i],
                                                   Lc[i][j],Lb[i][j],p_c_b[i][j],
                                                   p_c_b_err[i][j])
            #print i,j,-2*my_logp
            logp = logp + my_logp

    #And add the correction for possible issues with error estimation
    logp = logp - npoints*np.log(s)/2.

    return logp
