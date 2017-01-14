#!/u/ki/dapple/bin/python

import numpy as np
import scipy
import math
import sys

from glob import glob

import fit_with_covar
import cosmo

#Various routines for fitting using the P(lambda, L| M) formalism

#Analytic form of P(lambda, L|M)
def p_of_lambda_lcen(lm_val, L, M, param, Mpiv, z, B_L, extra=True):
    '''
    Analytic form for P(lambda, L|M)
    Inputs:  lambda, L, M, parameters, z, B_L
    L should be in luminosity (no log)
    Parameters should be in a single array, in the order:
    sigma_lambda, sigma_L, r, ln lambda_0, A_lambda, ln L_0, A_L
    Mpiv is a fixed mass scale from the n(M) expression
    The only purpose of z and B_L is to specify the amount of 
    redshift evolution in L
    '''
    
    #For ease of tracking parameters
    if extra:
        sigma_lm = np.sqrt(param[0]**2 + 1./lm_val)
    else:
        sigma_lm = param[0]
    sigma_L = param[1]
    r = param[2]
    lnlm0 = param[3]
    A_lm = param[4]
    lnL0 = param[5]
    A_L = param[6]

    #Mean lambda, L in the 2D Gaussian
    lnlmM = lnlm0 + A_lm*np.log(M/Mpiv)
    lnLM = lnL0 + A_L*np.log(M/Mpiv) + B_L*np.log(1+z)

    #determinance of covariance matrix
    detC = sigma_lm**2*sigma_L**2*(1-r**2)

    p = 1./(2*np.pi*np.sqrt(detC))*np.exp( -( (np.log(lm_val)-lnlmM )**2/sigma_lm**2
                                              -2*r*(np.log(lm_val)-lnlmM)*(np.log(L)-lnLM)/sigma_lm/sigma_L
                                              +(np.log(L)-lnLM)**2/sigma_L**2 )/(2.*(1-r**2)) )

    return p

#Get parameters for a third-order expression of the halo MF, for a given Mpiv
#Requires a dn/dlnM MF as input
def nm_approx_third(mval,nval,Mpiv):
    '''
    Gets parameters for third-order expression of halo MF, given Mpiv
    '''
    
    place = np.argmin( abs(np.log(mval/Mpiv)) )

    #First derivative
    beta1 = -np.log(nval[place+1]/nval[place-1])/np.log( mval[place+1]/mval[place-1] )
    
    #Second derivative
    b1 = np.log(nval[place+1]/nval[place])/np.log( mval[place+1]/mval[place] )
    b2 = np.log(nval[place]/nval[place-1])/np.log( mval[place]/mval[place-1] )
    beta2 = -(b1-b2)/np.log(mval[place+1]/mval[place])

    #Third derivative
    c1 = np.log(nval[place+2]/nval[place+1])/np.log( mval[place+2]/mval[place+1] )
    c2 = np.log(nval[place+1]/nval[place])/np.log( mval[place+1]/mval[place] )
    c3 = np.log(nval[place]/nval[place-1])/np.log( mval[place]/mval[place-1] )
    c4 = np.log(nval[place-1]/nval[place-2])/np.log( mval[place-1]/mval[place-2] )    
    b1 = (c1-c2)/np.log( mval[place+1]/mval[place] )
    b2 = (c3-c4)/np.log( mval[place]/mval[place-1] )
    beta3 = -(b1-b2)/np.log( mval[place+1]/mval[place-1] )

    #Normalization
    if Mpiv == mval[place]:
        A = nval[place]
    else:
        if Mpiv > mval[place]:
            place = place-1
        #Place now marks nearest point < Mpiv; interpolate in log space
        b = np.log(nval[place+1]/nval[place])/np.log(mval[place+1]/mval[place])
        A = nval[place]*(Mpiv/mval[place])**b

    return A, beta1, beta2, beta3

#Third order expression for the halo mass function
def nm_third(mval,A,beta1,beta2,beta3,Mpiv):
    '''
    Third order expression for the halo mass function
    Inputs: mval, A, beta1, beta2, beta3, Mpiv
    '''
    mu = np.log(mval/Mpiv)

    nm = A*np.exp( -beta1*mu -0.5*beta2*mu**2 )*(1-(1./6.)*beta3*mu**3)
    
    return nm

#sigma(M|lambda)
def sigma_m_lm(A,beta1,beta2,beta3,Mpiv,param,lm_val):
    #For ease of tracking parameters
    sigma_lm = np.sqrt(param[0]**2 + 1./lm_val)
    sigma_L = param[1] #Not needed
    r = param[2] #Not needed
    lnlm0 = param[3]
    A_lm = param[4]
    lnL0 = param[5] #Not needed
    A_L = param[6] #Not needed
    
    x_s = 1./(1+beta2*(sigma_lm/A_lm)**2)
    sigma_b2 = x_s*sigma_lm**2/A_lm**2

    return np.sqrt(sigma_b2)

#Mean mass at fixed lambda
def mass_at_lm(A,beta1,beta2,beta3,Mpiv,param,lm_val):
    #For ease of tracking parameters
    sigma_lm = np.sqrt(param[0]**2 + 1./lm_val)
    sigma_L = param[1] #Not needed
    r = param[2] #Not needed
    lnlm0 = param[3]
    A_lm = param[4]
    lnL0 = param[5] #Not needed
    A_L = param[6] #Not needed

    x_s = 1./(1+beta2*(sigma_lm/A_lm)**2)
    mass = (np.log(lm_val) - lnlm0)/A_lm - beta1*sigma_lm/A_lm
    mass = mass*x_s

    #Convert from mu = ln(M/Mpiv) to actual mass
    mass = np.exp(mass)*Mpiv

    return mass

#P(L|lambda)
def p_L(L,lm_val,A,beta1,beta2,beta3,Mpiv,param,z,B_L):
    '''
    Function giving P(L|lambda) given analytic input parameters
    Input: lambda, A, beta1, beta2, beta3, Mpiv, fitted parameters, z, B_L
    '''
    
    #For ease of tracking parameters
    sigma_lm = np.sqrt(param[0]**2 + 1./lm_val)
    #sigma_lm = param[0]
    sigma_L = param[1]
    r = param[2]
    lnlm0 = param[3]
    A_lm = param[4]
    lnL0 = param[5] + B_L*np.log(1+z)
    A_L = param[6]

    #Second-order part is still gaussian
    #Variance
    sigma_pl = ( sigma_lm**2*A_L**2 + sigma_L**2*A_lm**2 - 2*r*sigma_lm*sigma_L*A_lm*A_L + beta2*sigma_lm**2*sigma_L**2*(1-r**2) )/(A_lm**2+beta2*sigma_lm**2) #is sigma^2
    sigma_pl = np.sqrt(sigma_pl)
    
    #Mean as a function of lambda, accounting for passive ev offset
    lnLc0 = lnL0 + (A_lm*beta1*r*sigma_lm*sigma_L - (A_lm*A_L + beta2*r*sigma_lm*sigma_L)*lnlm0 - A_L*beta1*sigma_lm**2)/(A_lm**2 + beta2*sigma_lm**2)
    a_c = (A_lm*A_L+beta2*r*sigma_lm*sigma_L)/(A_lm**2 + beta2*sigma_lm**2)

    #print sigma_pl, lnLc0, a_c

    p2 = 1./np.sqrt(2*np.pi)/sigma_pl*np.exp( -( np.log(L) - (lnLc0 + a_c*np.log(lm_val) ) )**2/sigma_pl**2./2. )

    #Move to third order
    sigma1 = (1-r**2)/(A_lm**2/sigma_lm**2 + A_L**2/sigma_L**2 - 2*r*A_lm*A_L/sigma_lm/sigma_L) #is sigma_mu_b1^2
    mu1 = ( A_lm/sigma_lm**2*(np.log(lm_val)-lnlm0) + A_L/sigma_L**2*(np.log(L) - lnL0 ) - 
            r*A_lm*A_L/sigma_lm/sigma_L*((np.log(lm_val)-lnlm0)/A_lm + (np.log(L)-lnL0)/A_L) 
            )/( A_lm**2/sigma_lm**2 + A_L**2/sigma_L**2 - 2*r*A_lm*A_L/sigma_lm/sigma_L ) - beta1*sigma1 #Mean ln M/Mpiv
    x_s = 1./(1+beta2*sigma1)
    sigma2 = x_s*sigma1 #is sigma_mu_b2^2, for given L and lambda
    mu2 = x_s*mu1 #is mean given L and lambda

    x_s_lm = 1./(1+beta2*sigma_lm**2/A_lm**2) #value for one "signal" (lambda)
    sigma2_lm = (sigma_lm/A_lm)**2*x_s_lm
    mu2_lm = x_s_lm*( (np.log(lm_val) - lnlm0 )/A_lm - beta1*(sigma_lm/A_lm)**2 )

    #p3 = p2*( 1-beta3*( math.gamma(1.5)/np.sqrt(np.pi)*sigma2*mu2 + (1/6.)*mu2**3 ) 
    #          )/(1 - beta3*(math.gamma(1.5)/np.sqrt(np.pi)*sigma2_lm*mu2_lm +(1/6.)*mu2_lm**3 ))

    #Convert from d ln L to d log10 L
    #p3 = p3*np.log(10.)

    #print np.log10(np.exp(mu1)*Mpiv),np.log10(np.exp(mu2_lm)*Mpiv)
    #return p3
    return p2*np.log(10.)

#First-order only P(L|lambda)
def p_L_first(L,lm_val,A,beta1,Mpiv,param,z,B_L):
    '''
    Function giving P(L|lambda) given analytic input parameters
    Input: lambda, A, beta1, beta2, beta3, Mpiv, fitted parameters, z, B_L
    '''
    
    #For ease of tracking parameters
    sigma_lm = np.sqrt(param[0]**2 + 1./lm_val)
    sigma_L = param[1]
    r = param[2]
    lnlm0 = param[3]
    A_lm = param[4]
    lnL0 = param[5] + B_L*np.log(1+z)
    A_L = param[6]

    sigma_pl = A_L**2 * (sigma_L**2/A_L**2 + sigma_lm**2/A_lm**2- 2*r*sigma_L*sigma_lm/A_L/A_lm)
    sigma_pl = np.sqrt(sigma_pl)

    lnLc0 = lnL0 + A_L * ( (np.log(lm_val) - lnlm0)/A_lm + beta1*( r*sigma_lm*sigma_L/A_lm/A_L - sigma_lm**2/A_lm**2) )

    p1 = 1./np.sqrt(2*np.pi)/sigma_pl*np.exp( -( np.log(L) - lnLc0 )**2/sigma_pl**2./2. )

    return p1*np.log(10.)

#n(lambda)
def n_lm(lm_val,A,beta1,beta2,beta3,Mpiv,param):
    '''
    n(lambda) function, in dn/d lambda
    '''

    #For ease of tracking parameters
    #Note that this had been updated to include 1/lambda term in scatter
    sigma_lm = np.sqrt(param[0]**2 + 1./lm_val)
    sigma_L = param[1] #Not needed
    r = param[2] #Not needed
    lnlm0 = param[3]
    A_lm = param[4]
    lnL0 = param[5] #Not needed
    A_L = param[6] #Not needed

    n2 = A/np.sqrt(A_lm**2+beta2*sigma_lm**2)*np.exp(-0.5*( (np.log(lm_val)-lnlm0)**2*(beta2/(A_lm**2 + beta2*sigma_lm**2))
                                                            + (2*beta1*A_lm*(np.log(lm_val)-lnlm0)-beta1**2*sigma_lm**2 )/
                                                            (A_lm**2 + beta2*sigma_lm**2) ) )

    #Additional parts for third order
    x_s = 1./(1+beta2*(sigma_lm/A_lm)**2)
    sigma_b2 = x_s*sigma_lm**2/A_lm**2
    mu_b2 = x_s*( (np.log(lm_val)-lnlm0)/A_lm - beta1*sigma_lm**2/A_lm**2  )

    #For older versions that don't have the right gamma values - Gamma(1.5)
    gamma = 0.86227
    n3 = n2*(1-beta3*(gamma/np.sqrt(np.pi)*sigma_b2*mu_b2 +(1/6.)*mu_b2**3 ) )

    #Convert from dn/d ln lambda to dn/dlambda
    n3 = n3/lm_val

    return n3
    #return n2/lm_val

#Function to use for actually fitting to n(lambda) and Lcen data
#Assumes single redshift fit
def func_plm(param,x):
    '''
    Fitting function for n(lambda) and P(L|lambda)
    input is x, param
    x has the following format:
    A,beta1,beta2,beta3,Mpiv,z,B_L
    number of n(lambda) output data points
    number of P(L|lambda) output data points
    nlm_pts values of lambda; all log values
    nL_pts pairs of (L, lambda) values; all log values

    All fitting assumed to happened in log-log space
    '''
    
    #For ease of tracking parameters
    sigma_lm = param[0]
    sigma_L = param[1]
    r = param[2]
    lnlm0 = param[3]
    A_lm = param[4]
    lnL0 = param[5]
    A_L = param[6]

    #Deal with variable limits
    if(r < -1):
        r = -1.
    if(r > 1):
        r = 1.

    #Extracting values from the x matrix
    #Static parameter values first
    A = x[0]
    beta1 = x[1]
    beta2 = x[2]
    beta3 = x[3]
    Mpiv = x[4]
    z = x[5]
    B_L = x[6]
    
    #Number of n(lambda) data points
    nlm_pts = x[7]
    #Number of CLF data points
    nL_pts = x[8]

    y = np.zeros(nlm_pts+nL_pts)

    count = 9
    #Get the n(lambda) values
    for i in range(nlm_pts):
        y[i] = n_lm(x[count],A,beta1,beta2,beta3,Mpiv,param)
        count = count+1

    #Get the P(L|lambda) values
    for i in range(nL_pts):
        y[nlm_pts+i] = p_L(10.**x[count],x[count+1],A,beta1,beta2,beta3,Mpiv,param,z,B_L)
        count = count+2    

    #|r|>1 is disallowed
    #if(abs(r) > 1):
    #    return 1e9*np.log(y)

    return y

#An alternative version of the model, which replaces sigma_lm with 1/lambda + sigma_lm
#Note that this only
def func_plm_alt(param,x):
    '''
    Fitting function for n(lambda) and P(L|lambda)
    input is x, param
    x has the following format:
    A,beta1,beta2,beta3,Mpiv,z,B_L
    number of n(lambda) output data points
    number of P(L|lambda) output data points
    nlm_pts values of lambda; all log values
    nL_pts pairs of (L, lambda) values; all log values
    '''
    
    #For ease of tracking parameters
    sigma_lm = param[0]
    sigma_L = param[1]
    r = param[2]
    lnlm0 = param[3]
    A_lm = param[4]
    lnL0 = param[5]
    A_L = param[6]

    #Deal with variable limits
    if(r < -1):
        r = -1.
    if(r > 1):
        r = 1.

    #Extracting values from the x matrix
    #Static parameter values first
    A = x[0]
    beta1 = x[1]
    beta2 = x[2]
    beta3 = x[3]
    Mpiv = x[4]
    z = x[5]
    B_L = x[6]
    
    #Number of n(lambda) data points
    nlm_pts = x[7]
    #Number of CLF data points
    nL_pts = x[8]

    y = np.zeros(nlm_pts+nL_pts)

    count = 9
    #Get the n(lambda) values
    for i in range(nlm_pts):
        #Set sigma_lm
        #param[0] = np.sqrt(sigma_lm**2 + 1./x[count])
        y[i] = n_lm(x[count],A,beta1,beta2,beta3,Mpiv,param)
        count = count+1

    #Get the P(L|lambda) values
    for i in range(nL_pts):
        #Set sigma_lm
        #param[0] = np.sqrt(sigma_lm**2 + 1./x[count+1])
        y[nlm_pts+i] = p_L(10.**x[count],x[count+1],A,beta1,beta2,beta3,Mpiv,param,z,B_L)
        count = count+2    

    #And make sure param is reset and not actually changed
    #param[0] = sigma_lm

    #|r|>1 is disallowed
    #if(abs(r) > 1):
    #    return 1e9*np.log(y)

    #return np.log(y)
    return y

#Version that fits to centrals only; assumes that n(lambda) has been dropped from both vectors
def func_plm_cenonly(param,x):
    '''
    Fitting function for n(lambda) and P(L|lambda)
    input is x, param
    x has the following format:
    A,beta1,beta2,beta3,Mpiv,z,B_L
    number of n(lambda) output data points
    number of P(L|lambda) output data points
    nlm_pts values of lambda; all log values
    nL_pts pairs of (L, lambda) values; all log values
    '''
    
    #For ease of tracking parameters
    sigma_lm = param[0]
    sigma_L = param[1]
    r = param[2]
    lnlm0 = param[3]
    A_lm = param[4]
    lnL0 = param[5]
    A_L = param[6]

    #Deal with variable limits
    if(r < -1):
        r = -1.
    if(r > 1):
        r = 1.

    #Extracting values from the x matrix
    #Static parameter values first
    A = x[0]
    beta1 = x[1]
    beta2 = x[2]
    beta3 = x[3]
    Mpiv = x[4]
    z = x[5]
    B_L = x[6]
    
    #Number of CLF data points
    nL_pts = int(x[7])

    y = np.zeros(nL_pts)

    count = 8
    #Get P(L|lambda) values
    for i in range(nL_pts):
        #Set sigma_lm
        #param[0] = np.sqrt(sigma_lm**2 + 1./x[count+1])
        y[i] = p_L(10.**x[count],x[count+1],A,beta1,beta2,beta3,Mpiv,param,z,B_L)
        count = count+2

    return y

#Alternate fitting function which calculates n(lambda) only; should interface okay with 
#other functions, just drops the CLF part
def func_plm_nl(param,x): 
    '''
    Fitting function for n(lambda) and P(L|lambda)
    input is x, param
    x has the following format:
    A,beta1,beta2,beta3,Mpiv,z,B_L
    number of n(lambda) output data points
    number of P(L|lambda) output data points
    nlm_pts values of lambda; all log values
    nL_pts pairs of (L, lambda) values; all log values

    All fitting assumed to happened in log-log space

    Note that this version calculates n(lambda) only
    '''
    
    #For ease of tracking parameters
    sigma_lm = param[0]
    sigma_L = param[1]
    r = param[2]
    lnlm0 = param[3]
    A_lm = param[4]
    lnL0 = param[5]
    A_L = param[6]

    #Deal with variable limits
    if(r < -1):
        r = -1.
    if(r > 1):
        r = 1.

    #Extracting values from the x matrix
    #Static parameter values first
    A = x[0]
    beta1 = x[1]
    beta2 = x[2]
    beta3 = x[3]
    Mpiv = x[4]
    z = x[5]
    B_L = x[6]
    
    #Number of n(lambda) data points
    nlm_pts = x[7]
    #Number of CLF data points
    nL_pts = x[8]

    y = np.zeros(nlm_pts)

    count = 9
    #Get the n(lambda) values
    for i in range(nlm_pts):
        #Set sigma_lm
        #param[0] = np.sqrt(sigma_lm**2 + 1./x[count])
        y[i] = n_lm(x[count],A,beta1,beta2,beta3,Mpiv,param)
        count = count+1

    #Get the P(L|lambda) values
    #for i in range(nL_pts):
    #    #Set sigma_lm
    #    param[0] = np.sqrt(sigma_lm**2 + 1./x[count+1])
    #    y[nlm_pts+i] = p_L(10.**x[count],x[count+1],A,beta1,beta2,beta3,Mpiv,param,z,B_L)
    #    count = count+2    

    #And make sure param is reset and not actually changed
    param[0] = sigma_lm

    return y 


#Running the fit
def fit_plm(indir,l_zmin,l_zmax,
            start,A,beta1,beta2,beta3,Mpiv,z,B_L,
            lm_med=[11.8,17.0,27.2,22.1,27.2,33.9,50.4,46.9,70.7],
            glist = [0,1,3,4,5,7,8],
            func_fit=func_plm):
    '''
    Fitting function for P(lambda,L|M) to central CLF and n(lambda) data

    Inputs:  name of directory containing all files of interest
             l_zmin, l_zmax -- strings given the desired redshift range
             start -- initial parameter guess
             A, beta1, beta2, beta3, Mpiv -- n(M) parameters
             z, B_L -- median redshift and B_L value for centrals
    '''

    #Creat the x array with its first few elements, with two spaces for numbers of points
    x = [A,beta1,beta2,beta3,Mpiv,z,B_L,0,0]
    y = []
    covar_all = []

    #Next step is to read in the n(lambda) data
    nlm_data = np.loadtxt(indir+"nlambda_z_"+l_zmin+"_"+l_zmax+".dat")
    #And construct the first part of the input array
    #and covariance matrix for n(lambda)
    nlist = np.where( nlm_data[:,1] > 0 )[0]
    x[7] = len(nlist)
    for i in range(len(nlist)):
        #x.append(np.log(nlm_data[nlist[i],0]))
        #y.append(np.log(nlm_data[nlist[i],1]))
        #err[i] = nlm_data[nlist[i],2]/nlm_data[nlist[i],1]
        x.append(nlm_data[nlist[i],0])
        y.append(nlm_data[nlist[i],1])

    covar_temp = np.loadtxt(indir+"nlambda_covar_z_"+l_zmin+"_"+l_zmax+".dat")
    covar_temp = covar_temp[:,nlist]
    covar_temp = covar_temp[nlist,:]
    
    #Regularize
    covar_temp = fit_with_covar.regularize(covar_temp)

    covar_all.append(covar_temp)

    #Figure out which CLF files we want to read in
    cenfiles = glob(indir+"clf_cen_z_"+l_zmin+"_"+l_zmax+"*.dat")
    covfiles = glob(indir+"clf_cen_covar_z_"+l_zmin+"_"+l_zmax+"*.dat")
    cenfiles.sort()
    covfiles.sort()
    cenfiles = np.array(cenfiles)
    covfiles = np.array(covfiles)
    
    #Now read in the CLF data and finish constructing the input arrays
    for i in range(len(glist)):
        clf = np.loadtxt(cenfiles[glist[i]])
        mlist = np.where(clf[:,1] > 0)[0]
        clf = clf[mlist,:]

        covar_temp = np.loadtxt(covfiles[glist[i]])
        covar_temp = covar_temp[:,mlist]
        covar_temp = covar_temp[mlist,:]
        
        x[8] = x[8]+len(mlist)

        for j in range(len(clf)):
            #x.append(np.log(10.)*clf[j,0]) #L
            #x.append(np.log(lm_med[glist[i]])) #lambda
            #y.append(np.log(clf[j,1]))
            x.append(clf[j,0]) #L
            x.append(lm_med[glist[i]]) #lambda
            y.append(clf[j,1])

        #Correct the covariance matrix to log values
        for j in range(len(clf)):
            for k in range(len(clf)):
                covar_temp[j,k] = covar_temp[j,k]#/clf[j,1]/clf[k,1]

        covar_all.append(covar_temp)

    #Fitting should happen here, for block-diagonal covariance
    if len(start) == 0:
        start = [0.1, 0.1, 0.99, np.log(20.), 1, 11.*np.log(10.), 1]
    print len(x),x[7],x[8]
    [chi2, res, res_covar] = fit_with_covar.fit_with_block_covar(start,func_fit,x,y,covar_all)

    npoints = len(y)

    return [npoints, chi2, res, res_covar, x, y, covar_all]

#Comparing analytic results with convolution with numerical n(M)
def p_L_convolved(L,lm_val,A,beta1,beta2,beta3,Mpiv,param,z,B_L,mval,nval):
    '''
    Inputs: L,lm_val,A,beta1,beta2,beta3,Mpiv,param,z,B_L,mval,nval
    
    L, lm_val MUST be single values
    '''
    plm = 0*mval
    for i in range(len(mval)):
        plm[i] = p_of_lambda_lcen(lm_val,L,mval[i],param,Mpiv,z,B_L)

    #L values to use in integration
    dL = 0.01
    L_integ = 9+np.array(range(300))*dL
    #print L_integ

    #Numerator -- n(lambda, L)
    p_num = np.sum( nval*plm*np.log(mval[1]/mval[0]) )
    #Denominator -- n(lambda)
    plm_arr = np.zeros([len(mval), len(L_integ)])
    for i in range(len(mval)):
        for j in range(len(L_integ)):
            #print i,j,mval[i],nval[i], L_integ[j],p_of_lambda_lcen(lm_val,10.**L_integ[j],mval[i],param,Mpiv,z,B_L)
            plm_arr[i,j] = nval[i]*np.log(mval[1]/mval[0])*dL*p_of_lambda_lcen(lm_val,10.**L_integ[j],mval[i],param,Mpiv,z,B_L)

    p = p_num/np.sum(plm_arr)
    return p
    #return plm_arr

#Comparing analytic results with convolution with numerical n(M)
def n_lm_convolved(lm_val,A,beta1,beta2,beta3,Mpiv,param,z,B_L,mval,nval,extra=True):
    '''
    Inputs: lm_val,A,beta1,beta2,beta3,Mpiv,param,z,B_L,mval,nval
    
    lm_val MUST be single value
    '''
    plm = 0*mval
    dL = 0.05
    Lbins = 9.+np.array(range(60))*dL

    #Handle the extra scatter
    #sigma_lm = param[0]
    #param[0] = np.sqrt(sigma_lm**2 + 1./lm_val)

    for i in range(len(mval)):
        plm[i] = np.sum(p_of_lambda_lcen(lm_val,10.**Lbins,mval[i],param,Mpiv,z,B_L,extra=extra))*dL*np.log(10.)
    
    #print plm
    #print np.sum(plm)*np.log(mval[1]/mval[0])

    #note conversion to dn/d lambda from dn/d ln lambda
    nlm = np.sum(plm*nval)*np.log(mval[1]/mval[0])/lm_val
    #nlm_alt = np.sum(plm*nm_third(mval,A,beta1,beta2,beta3,Mpiv))*np.log(mval[1]/mval[0])/lm_val

    #Fix the input back to what it started as
    #param[0] = sigma_lm
    
    #return nlm, nlm_alt
    return nlm, 0

#Calculation of interpolated HMF parameters in between two given redshifts
def nm_interpol(zmin,zmax,z,aA, abeta1,abeta2,abeta3, bA, bbeta1, bbeta2, bbeta3):
    iA = np.exp( np.log(aA) + np.log(bA/aA)*np.log((1+z)/(1+zmin))/np.log((1+zmax)/(1+zmin)) )

    ibeta1 = abeta1 - (abeta1 - bbeta1)*(z-zmin)/(zmax-zmin)
    ibeta2 = abeta2 - (abeta2 - bbeta2)*(z-zmin)/(zmax-zmin)
    ibeta3 = abeta3 - (abeta3 - bbeta3)*(z-zmin)/(zmax-zmin)  

    return [iA, ibeta1, ibeta2, ibeta3]

#Calculation of n(lambda) in a redshift bin, given the HMF on the bin edges
def nlm_z_interpol(lm_val,zmin,zmax,mass_param_lo,mass_param_hi,Mpiv,param):
    #Start with fine redshift bins, getting the midpoint redshift
    nzbins = 100
    dz = (zmax-zmin)/nzbins
    zmid = zmin+dz/2.+np.array(range(nzbins))*dz

    nlambda = np.zeros_like(lm_val).astype(float)
    
    #Add n(lambda) for each redshift slice; note that this includes volume weights
    #And DOES NOT includes evolution of lambda-mass with redshift
    for i in range(nzbins):
        mass_param = nm_interpol(zmin,zmax,zmid[i],mass_param_lo[0],mass_param_lo[1],
                                 mass_param_lo[2],mass_param_lo[3],mass_param_hi[0],
                                 mass_param_hi[1],mass_param_hi[2],mass_param_hi[3])
        nlambda = nlambda + n_lm(lm_val,mass_param[0],mass_param[1],mass_param[2],
                                 mass_param[3],Mpiv,param)*cosmo.comoving_volume(zmid[i]-dz/2.,zmid[i]+dz/2.)
        print "Now at ",i,zmid[i],nlambda[0],cosmo.comoving_volume(zmid[i]-dz/2.,zmid[i]+dz/2.)

    #And normalize
    nlambda = nlambda/cosmo.comoving_volume(zmin,zmax)

    return nlambda

#Functions for setup for MCMC code

#Calculate P(L|lambda) properly, using redshift bins -- note that this is based on p_L and n_lm
def p_L_binned(L,lm_val,zmin,zmax,A,beta1,beta2,beta3,Mpiv,param,B_L):
    #Get the binned values in lambda
    dz = 0.01
    zbins = zmin + (zmax-zmin)*dz*np.array(range(100))
    
    
    #Sum up the numerator and denominator arrays
    p_num = np.zeros_like(L)
    p_den = np.zeros_like(L)

    for i in range(len(zbins)):
        #print p_num
        p_num = p_num + p_L(L,lm_val,A,beta1,beta2,beta3,Mpiv,param,zbins[i],B_L)*n_lm(lm_val,A,beta1,beta2,beta3,Mpiv,param)*cosmo.comoving_volume(zbins[i]-dz/2., zbins[i]+dz/2.)
        p_den = p_den + n_lm(lm_val,A,beta1,beta2,beta3,Mpiv,param)*cosmo.comoving_volume(zbins[i]-dz/2., zbins[i]+dz/2.)

    p = p_num/p_den

    return p

#Setup of data vectors for chi^2 calculation
#Includes a lambda cutoff for the n(lambda) section
def make_data_vectors(indir,l_zmin,l_zmax,A,beta1,beta2,beta3,Mpiv,z,B_L,
                      lm_med=[11.8,17.0,27.2,22.1,27.2,33.9,50.4,46.9,70.7],
                      glist=[0,1,3,4,5,7,8],
                      lm_max=80.,lm_min=0, nl_only=False,
                      k=[],Pk=[],volume=1.,
                      H0=67.04,omegaM=0.317,omegaL=0.683):
    x = [A,beta1,beta2,beta3,Mpiv,z,B_L,0,0]
    y = []
    covar_all = []

        #Next step is to read in the n(lambda) data
    nlm_data = np.loadtxt(indir+"nlambda_z_"+l_zmin+"_"+l_zmax+".dat")
    #And construct the first part of the input array
    #and covariance matrix for n(lambda)
    nlist = np.where( (nlm_data[:,1] > 0) & (nlm_data[:,0] < lm_max) & (nlm_data[:,0] > lm_min) )[0]
    x[7] = len(nlist)
    for i in range(len(nlist)):
        x.append(nlm_data[nlist[i],0])
        y.append(nlm_data[nlist[i],1]*cosmo.comoving_volume(float(l_zmin),float(l_zmax))/cosmo.comoving_volume(float(l_zmin),float(l_zmax),H0=H0,omegaM=omegaM,omegaL=omegaL)) #Note cosmology correction to volume

    covar_temp = np.loadtxt(indir+"nlambda_covar_z_"+l_zmin+"_"+l_zmax+".dat")
    covar_temp = covar_temp[:,nlist]
    covar_temp = covar_temp[nlist,:]
    
    #Regularize
    covar_temp = fit_with_covar.regularize(covar_temp)

    covar_all.append(covar_temp)

    if not nl_only:

        #Figure out which CLF files we want to read in
        cenfiles = glob(indir+"clf_cen_z_"+l_zmin+"_"+l_zmax+"*.dat")
        covfiles = glob(indir+"clf_cen_covar_z_"+l_zmin+"_"+l_zmax+"*.dat")
        cenfiles.sort()
        covfiles.sort()
        cenfiles = np.array(cenfiles)
        covfiles = np.array(covfiles)

    #Now read in the CLF data and finish constructing the input arrays
        for i in range(len(glist)):
            clf = np.loadtxt(cenfiles[glist[i]])
            mlist = np.where( (clf[:,1] > 0.01) & (clf[:,0] > 10.))[0]
            clf = clf[mlist,:]

            covar_temp = np.loadtxt(covfiles[glist[i]])
            covar_temp = covar_temp[:,mlist]
            covar_temp = covar_temp[mlist,:]
        
            x[8] = x[8]+len(mlist)

            for j in range(len(clf)):
                x.append(clf[j,0]) #L
            
                x.append(lm_med[glist[i]]) #lambda
                y.append(clf[j,1])

            covar_all.append(covar_temp)
        
    return [x, y, covar_all]

def pl_norm_check(new_param,x):
    #Sanity check addition to chi2 -- require p_L normalized to 1%
    lumbins = np.array(range(40))*0.08+9
    norm_check = (1-0.08*np.sum(p_L(10.**lumbins,10.,x[0],x[1],x[2],x[3],x[4],new_param,x[5],x[6])))**2*1e4
    norm_check += (1-0.08*np.sum(p_L(10.**lumbins,20.,x[0],x[1],x[2],x[3],x[4],new_param,x[5],x[6])))**2*1e4
    norm_check += (1-0.08*np.sum(p_L(10.**lumbins,60.,x[0],x[1],x[2],x[3],x[4],new_param,x[5],x[6])))**2*1e4

    return norm_check

#Calculate the likelihood, including a list of the N most massive clusters
#Actually returns log-likelihood due to individual likelihoods being small
def plm_likelihood(param,x,y,covar_all,lambda_big,volume):
    #print x
    A = x[0]
    beta1 = x[1]
    beta2 = x[2]
    beta3 = x[3]
    Mpiv = x[4]

    #Get the chi^2 value for the binned data first
    chi2 = fit_with_covar.get_chisq_with_block_covar(param,func_plm_alt,x,y,covar_all)[0]
    #print chi2

    #Get our first part of the log-likelihood from the chi2 value
    logp = -chi2/2.

    #Take the remaining part from the N most massive clusters
    ncl = len(lambda_big)
    #Calculate n(lambda) for each of these lambda values
    nlm_big = n_lm(lambda_big,A,beta1,beta2,beta3,Mpiv,param)
    logp = logp + np.sum(np.log(abs(nlm_big)))

    #And give the final "tail" weight, for not having "in-between" clusters
    #Note that this is kind of a pain to handle otherwise
    lm_vals = lambda_big[-1]+np.array(range(500))
    logp = logp - np.sum(n_lm(lm_vals,A,beta1,beta2,beta3,Mpiv,param))*volume
    #print "TEST: ", - np.sum(n_lm(lm_vals,A,beta1,beta2,beta3,Mpiv,param))*volume, np.sum(np.log(abs(nlm_big)))

    #Added check for correct p_L normalization to keep the fit from haring off
    #logp = logp - pl_norm_check(param,x)/2.

    return logp

#Version of individual bin likelihood that ignores all n(lambda) parts
def plm_likelihood_cen_single(param,x,y,covar_all,lambda_big,volume,verbose=False):
    #print x
    A = x[0]
    beta1 = x[1]
    beta2 = x[2]
    beta3 = x[3]
    Mpiv = x[4]

    #Get the chi^2 value for the binned data first

    #Trim x, y vectors, covar_all to relevant data
    xtrim = np.zeros(8+x[8]*2)
    xtrim[0:7] = x[0:7]
    xtrim[7] = x[8]
    xtrim[8:] = x[9 + x[7]:]
    ytrim = y[x[7]:]
    chi2 = fit_with_covar.get_chisq_with_block_covar(param,func_plm_cenonly,xtrim,ytrim,covar_all[1:])[0]
    #print chi2

    #Get our first part of the log-likelihood from the chi2 value
    logp = -chi2/2.

    return logp

#Calculates likelihood, but uses n_lm_convolved for the tail parts
def plm_like_conv(param,x,y,covar_all,lambda_big,volume,mval,nval):
    A = x[0]
    beta1 = x[1]
    beta2 = x[2]
    beta3 = x[3]
    Mpiv = x[4]

    #Get the chi^2 value for the binned data first
    chi2 = fit_with_covar.get_chisq_with_block_covar(param,func_plm_alt,x,y,covar_all)[0]

    #Get our first part of the log-likelihood from the chi2 value
    logp = -chi2/2.

    #Take the remaining part from the N most massive clusters
    ncl = len(lambda_big)

    #Alternate version that uses the convolved n(lambda)
    nlm_big = np.zeros(ncl)
    for i in range(ncl):
        [nlm_big[i], temp] = n_lm_convolved(lambda_big[i],A,beta1,beta2,beta3,Mpiv,param,x[5],x[6],mval,nval)
    logp = logp + np.sum(np.log(nlm_big))

    #And give the final "tail" weight, for not having "in-between" clusters
    #Note that this is kind of a pain to handle otherwise
    lm_vals = lambda_big[-1]+np.array(range(200))
    nlm_vals = 0*lm_vals
    for i in range(len(nlm_vals)):
        [nlm_vals[i], temp] = n_lm_convolved(lm_vals[i],A,beta1,beta2,beta3,Mpiv,param,x[5],x[6],mval,nval)
    logp = logp - np.sum(nlm_vals)*volume

    return logp

#Version using n(lambda) only
def plm_like_nl(param,x,y,covar_all,lambda_big,volume):
    A = x[0]
    beta1 = x[1]
    beta2 = x[2]
    beta3 = x[3]
    Mpiv = x[4]

    #Get the chi^2 value for the binned data first
    chi2 = fit_with_covar.get_chisq_with_block_covar(param,func_plm_nl,x,y,covar_all)[0]

    #Get our first part of the log-likelihood from the chi2 value
    logp = -chi2/2.

    #Take the remaining part from the N most massive clusters
    ncl = len(lambda_big)
    #Calculate n(lambda) for each of these lambda values
    nlm_big = n_lm(lambda_big,A,beta1,beta2,beta3,Mpiv,param)
    logp = logp + np.sum(np.log(nlm_big))

    #And give the final "tail" weight, for not having "in-between" clusters
    #Note that this is kind of a pain to handle otherwise
    lm_vals = lambda_big[-1]+np.array(range(500))
    logp = logp - np.sum(n_lm(lm_vals,A,beta1,beta2,beta3,Mpiv,param))

    return logp

#Version set to run for multiple redshift bins simultaneously
#Each input other than param should now be a list, with one element for each redshift range
def plm_like_multiz(param,x,y,covar_all,myz,lambda_big,volume):
    logp = 0.
    nz = len(x)

    for i in range(nz):
        #Update B_L for this redshift
        x[i][6] = param[7]

        #Now add the log-likelihood for this bin
        logp = logp + plm_likelihood(param,x[i],y[i],covar_all[i],lambda_big[i],volume[i])

    return logp

#Version that returns -logp, for fitting purposes
def plm_like_multiz_fit(param,x,y,covar_all,myz,lambda_big,volume):
    logp = 0.
    nz = len(x)

    for i in range(nz):
        #Update B_L for this redshift
        x[i][6] = param[7]

        #Now add the log-likelihood for this bin
        logp = logp + plm_likelihood(param,x[i],y[i],covar_all[i],lambda_big[i],volume[i])

    return -np.repeat(logp,len(param))

#Version that includes an additional term for evolution in the lambda-mass relationship
def plm_like_multiz_ev(param,x,y,covar_all,myz,lambda_big,volume):
    logp = 0.
    nz = len(x)

    for i in range(nz):
        #Update B_L for this redshift
        x[i][6] = param[7]

        #Add evolution in ln lambda_0 for this redshift
        myparam = np.copy(param[0:7])
        myparam[3] = myparam[3] + param[8]*np.log(1+myz[i])

        #Now add the log-likelihood for this bin
        logp = logp + plm_likelihood(myparam,x[i],y[i],covar_all[i],lambda_big[i],volume[i])
        

    return logp

#Likelihood with lambda-mass evolution given by the changes in halo mass definition and
#background density
def mass_base_true(mnew,z):
    b0 = -0.5318
    c0 = 0.030

    mass = mnew**(1/(c0*(z-0.2)+1))*10.**(14*(1-1./(c0*(z-0.2)+1)))*np.exp(-b0*(z-0.2)/(c0*(z-0.2)+1))

    return mass

#Generation of cosmic variance addition to covariance matrix
def nlm_sample_variance(myparam,x,sigma2_R):
    #First, get the input lambda values and calculate n(lambda)
    lambda_val = x[9:9+x[7]]
    #Now, n(lambda)
    #Don't forget to update the scatter for each 
    npoints = len(lambda_val)
    y = np.zeros_like(lambda_val)
    sigma_lm = myparam[0]
    for i in range(npoints):
        myparam[0] = np.sqrt(sigma_lm**2 + 1./lambda_val[i])
        y[i] = n_lm(lambda_val[i],x[0],x[1],x[2],x[3],x[4],myparam)
    
    cov = np.zeros([npoints,npoints])
    bias = 3.
    for i in range(npoints):
        cov[i,:] = y[i]*y*bias**3*sigma2_R

    return cov

#Version that adds physically motivated evolution in lambda(M)
#Also added option that handles sigma2(R) for cosmic variance to n(lambda) errors
def plm_like_multiz_ev_fix(param,x,y,covar_all,myz,lambda_big,volume,sigma2_R):
    logp = 0.
    nz = len(x)

    #Parameters defining lambda-mass evolution
    b0 = -0.5318
    c0 = 0.030
    h = 0.6704

    for i in range(nz):
        #Update B_L for this redshift
        x[i][6] = param[7]
        Mpiv = x[i][4]

        #Update local parameters for this redshift
        myparam = np.copy(param[0:7])
        #Adjustment to ln lambda_0
        #Note little h correction needed to handle earlier issue
        myparam[3] = param[3] + param[4]*(1/(c0*(myz[i]-0.2)+1)-1)*np.log(Mpiv*h) + param[4]*(1-1/(c0*(myz[i]-0.2)+1))*14.*np.log(10.*h) - b0*(myz[i]-0.2)/(c0*(myz[i]-0.2)+1)
        #Adjustment to A_lambda
        myparam[4] = param[4]/(c0*(myz[i]-0.2)+1)

        #Adjustment to add cosmic variance errors
        my_covar = covar_all[i]
        covar_temp = my_covar[0]
        cov_cosmic = nlm_sample_variance(myparam,x[i],sigma2_R[i])
        my_covar[0] = my_covar[0]+cov_cosmic

        #Now add the log-likelihood for this bin
        logp = logp + plm_likelihood(myparam,x[i],y[i],my_covar,lambda_big[i],volume[i])

    return logp


#Version that adds physically motivated evolution in lambda(M)
#Also added option that handles sigma2(R) for cosmic variance to n(lambda) errors
#Note that this fits to n(lambda) only
def plm_like_multiz_ev_fix_nl(param,x,y,covar_all,myz,lambda_big,volume,sigma2_R):
    logp = 0.
    nz = len(x)

    #Parameters defining lambda-mass evolution
    b0 = -0.5318
    c0 = 0.030
    h = 0.6704

    for i in range(nz):
        #Update B_L for this redshift
        x[i][6] = param[7]
        Mpiv = x[i][4]

        #Update local parameters for this redshift
        myparam = np.copy(param[0:7])
        #Adjustment to ln lambda_0
        #Note little h correction needed to handle earlier issue
        myparam[3] = param[3] + param[4]*(1/(c0*(myz[i]-0.2)+1)-1)*np.log(Mpiv*h) + param[4]*(1-1/(c0*(myz[i]-0.2)+1))*14.*np.log(10.*h) - b0*(myz[i]-0.2)/(c0*(myz[i]-0.2)+1)
        #Adjustment to A_lambda
        myparam[4] = param[4]/(c0*(myz[i]-0.2)+1)

        #Adjustment to add cosmic variance errors
        my_covar = covar_all[i]
        covar_temp = my_covar[0]
        cov_cosmic = nlm_sample_variance(myparam,x[i],sigma2_R[i])
        my_covar[0] = my_covar[0]+cov_cosmic

        #Now add the log-likelihood for this bin
        logp = logp + plm_like_nl(myparam,x[i],y[i],my_covar,lambda_big[i],volume[i])

    return logp

#
#Note other options are not a good idea at the present time
def plm_like_cen_only(cen_param,x,y,covar_all,myz,lambda_big,volume,sigma2_R,npoints,
                      verbose=False):
    #Implementing parameter limits -- note this is necessary to make this work with emcee
    if (cen_param[0] < 0) or (cen_param[5] < 0) or (cen_param[1] < -1) or (cen_param[1] > 1):
        return -np.inf

    logp = 0.
    nz = len(x[1])

    lm_param = x[0]
    
    if verbose:
        print >> sys.stderr, lm_param

    #Note the pivot and other mass-lambda parameters
    #Case where lm_param is a single fixed parameter set
    Mpiv = lm_param[0]
    A_lm = lm_param[1]
    B_lm = lm_param[2]
    lnlm0 = lm_param[3]
    if len(lm_param) == 5:
        sigma_lm = lm_param[4]
    else:
        sigma_lm = 0.1842
        
    if verbose:
        print >> sys.stderr, "lm_param: ",A_lm, B_lm, lnlm0
    h = 0.6704
    for i in range(nz):

        in_param = np.zeros(8)
        in_param[0] = sigma_lm
        in_param[1] = cen_param[0] #sigma_L
        in_param[2] = cen_param[1] #r
        in_param[3] = lnlm0 + B_lm*np.log(1+myz[i]) #
        in_param[4] = A_lm
        in_param[5] = cen_param[2] #ln L0
        in_param[6] = cen_param[3] #A_L
        in_param[7] = cen_param[4] #B_L

        #Make sure to set B_L in the x vector
        x[1][i][6] = in_param[7]

        #This is the scaling for the covariance matrices; note this is NOT
        #applied to the n(lambda) covariance
        s = cen_param[5]
        
        cov_in = np.copy(covar_all[i])
        for j in range(len(cov_in)):
            if j != 0:
                cov_in[j] = cov_in[j]*s
        
        #print in_param
        chi2 = plm_likelihood_cen_single(in_param,x[1][i],y[i],cov_in,lambda_big[i],volume[i],
                                         verbose=verbose)
        logp = logp + chi2
        if verbose:
            print i, nz, chi2*2, x[1][i][8]

    #And add a correcting term for our inflation of the errors
    logp = logp - npoints*np.log(s)/2.

    return logp

#Single-redshift version, for fitting in single redshift slices; no z evolution included
def plm_like_cen_noev(cen_param,x,y,covar_all,myz,lambda_big,volume,sigma2_R,npoints,
                      verbose=False):
    #Implementing parameter limits
    if (cen_param[0] < 0) or (cen_param[4] < 0) or (cen_param[1] < -1) or (cen_param[1] > 1):
        return -np.inf

    logp = 0
    nz = len(x[1])

    lm_param = x[0]

    #Note the pivot and other mass-lambda parameters
    Mpiv = lm_param[0]
    A_lm = lm_param[1]
    B_lm = lm_param[2]
    lnlm0 = lm_param[3]
    if len(lm_param) == 5:
        sigma_lm = lm_param[4]
    else:
        sigma_lm = 0.1842

    h = 0.6704
    for i in range(nz):

        in_param = np.zeros(8)
        in_param[0] = sigma_lm
        in_param[1] = cen_param[0] #sigma_L
        in_param[2] = cen_param[1] #r
        in_param[3] = lnlm0 + B_lm*np.log(1+myz[i]) #lambda point
        in_param[4] = A_lm
        in_param[5] = cen_param[2] #ln L0
        in_param[6] = cen_param[3] #A_L
        in_param[7] = 0 #B_L = 0 (no evolution)
        
        #Make sure to set B_L (to zero) in the x vector
        x[1][i][6] = in_param[7]
        
        #This is the scaling for the covariance matrices
        s = cen_param[4]

        cov_in = np.copy(covar_all[i])
        for j in range(len(cov_in)):
            if j != 0:
                cov_in[j] = cov_in[j]*s
        
        #print in_param
        chi2 = plm_likelihood_cen_single(in_param,x[1][i],y[i],cov_in,lambda_big[i],volume[i],
                                         verbose=verbose)
        logp = logp + chi2
        if verbose:
            print i, nz, chi2*2, x[1][i][8]

    #And add the correcting term for inflation of the errors
    logp = logp - npoints*np.log(s)/2.

    return logp

#ln(Prior) on lambda-mass relationship -- 2D Gaussian
def prior_lm(lm_param,lm_param_mean,lm_cov):
    detC = np.linalg.det(lm_cov)

    p = 1/(2*np.pi)**1.5/np.sqrt(detC)*np.exp( -0.5*np.dot( lm_param-lm_param_mean,np.dot(np.linalg.inv(lm_cov), lm_param-lm_param_mean)) )

    return np.log(p)

#Full probability -- centrals only, with evolution
def plm_like_cen_with_prior(param,lm_param_mean,lm_cov,x,y,covar_all,myz,lambda_big,volume,
                            sigma2_R,npoints):
    lm_param = param[0:3]
    cen_param = param[3:]
    x[0] = [x[0][0],lm_param[0],lm_param[1],lm_param[2], x[0][4]]
    
    p1 = prior_lm(lm_param,lm_param_mean,lm_cov)
    p2 = plm_like_cen_only(cen_param,x,y,covar_all,myz,lambda_big,volume,sigma2_R,npoints)
    #print p1,p2,p1+p2
    return p1+p2

#Full probability -- centrals only, no evolution
def plm_like_cen_noev_with_prior(param,lm_param_mean,lm_cov,x,y,covar_all,myz,lambda_big,volume,
                                 sigma2_R,npoints):
    lm_param = param[0:3]
    cen_param = param[3:]
    x[0] = [x[0][0],lm_param[0],lm_param[1],lm_param[2],x[0][4]]

    return prior_lm(lm_param,lm_param_mean,lm_cov) + plm_like_cen_noev(cen_param,x,y,covar_all,myz,
                                                                       lambda_big,volume,sigma2_R,
                                                                       npoints)
#Probability with an additional prior on A_L
def plm_like_cen_noev_with_AL_prior(param,lm_param_mean,lm_cov,x,y,covar_all,myz,lambda_big,
                                    volume,sigma2_R,npoints):
    lm_param = param[0:3]
    cen_param = param[3:]
    x[0] = [x[0][0],lm_param[0],lm_param[1],lm_param[2],x[0][4]]

    p_AL = 1/np.pi/np.sqrt(2)/0.009*np.exp(-(cen_param[3]-0.359)**2/(2*0.009*0.009))
    print >> sys.stderr,cen_param[3],p_AL,np.log(p_AL)
    
    return prior_lm(lm_param,lm_param_mean,lm_cov) + plm_like_cen_noev(cen_param,x,y,covar_all,myz,
                                                                       lambda_big,volume,sigma2_R,
                                                                       npoints) + np.log(p_AL)

#Full probability -- centrals only, with evolution
def plm_like_cen_with_AL_prior(param,lm_param_mean,lm_cov,x,y,covar_all,myz,lambda_big,volume,
                            sigma2_R,npoints):
    lm_param = param[0:3]
    cen_param = param[3:]
    x[0] = [x[0][0],lm_param[0],lm_param[1],lm_param[2], x[0][4]]
    
    p1 = prior_lm(lm_param,lm_param_mean,lm_cov)
    p2 = plm_like_cen_only(cen_param,x,y,covar_all,myz,lambda_big,volume,sigma2_R,npoints)
    p_AL = 1/np.pi/np.sqrt(2)/0.009*np.exp(-(cen_param[3]-0.359)**2/(2*0.009*0.009))
    #print p1,p2,p1+p2
    return p1+p2+np.log(p_AL)
