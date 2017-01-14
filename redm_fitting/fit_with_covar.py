#!/u/ki/dapple/bin/python

##!/usr/bin/env python

import os
import sys

import numpy as np
import scipy
import scipy.optimize

#General routines for fitting functions including covariance

#Function for regularizing a potentially noisy covariance matrix
def regularize(covar_in,alpha=0.1):
    #[u, s, v] = np.linalg.svd(covar_in)
    
    #s_new = np.copy(s)
    #mylist = np.where(s_new < 1e-3)[0]
    #klist = np.where(s_new >= 1e-3)[0]
    #if len(mylist) > 0:
    #    s_new[mylist] = 0*mylist+1e-3
    
    #SVD version -- is made of epic fail
    #mylist = np.where(s_new > 1e3)[0]
    #klist = np.where(s_new <= 1e3)[0]
    #if (len(mylist)>0) & (s[0] > 10*s[1]):
    #    s_new[mylist] = 0*mylist+np.max(s_new[klist])

    #covar = np.dot(u,np.dot(np.diag(s_new),v))

    #Shrink to uniform on diagonals
    covar = (1-alpha)*covar_in + alpha*np.trace(covar_in)/len(covar_in)*np.identity(len(covar_in))

    #Shrink to diagonals
    mlist = np.array(range(len(covar_in)))
    covar = (1-alpha)*covar_in + alpha*np.diag(covar[mlist,mlist])

    return covar

#General routine for calculating chi^2 for some fitting function
def get_chisq_with_covar(params,func,x,y,covar):

    dx = func(params,x) - y
    
    if np.linalg.det(covar) == 0:
        print >> sys.stderr, "ERROR: Covariance matrix is singular (det=0)"
        sys.exit(1)

    #Multiplying the differences by the inverse matrix,
    # transpose(dx) C^-1 dx
    chisq = np.dot(np.dot(dx,np.linalg.inv(covar)),dx)

    chisq = np.resize(np.array([chisq]),len(params))

    return chisq

def fit_chisq_partial_derivative(i,j,params,func,x,y,covar):
    tol = 0.01

    dparam_i = tol*params[i]
    dparam_j = tol*params[j]

    dpi = np.zeros_like(params)
    dpi[i] = dparam_i/2.
    dpj = np.zeros_like(params)
    dpj[j] = dparam_j/2.

    second_der_start = 0
    second_der = 10

    while abs(second_der - second_der_start)/second_der > 0.001:
        #print second_der_start, second_der
        second_der_start = second_der
        if i == j:
            val_hi = get_chisq_with_covar(params+2*dpi,func,x,y,covar)[0]
            val_mid = get_chisq_with_covar(params,func,x,y,covar)[0]
            val_lo = get_chisq_with_covar(params-2*dpi,func,x,y,covar)[0]
            second_der = (val_hi - 2*val_mid + val_lo)/dparam_i/dparam_i
        else:
            first_hi = (get_chisq_with_covar(params+dpi+dpj,func,x,y,covar)[0] - get_chisq_with_covar(params-dpi+dpj,func,x,y,covar)[0])/dparam_i
            first_lo = (get_chisq_with_covar(params+dpi-dpj,func,x,y,covar)[0] - get_chisq_with_covar(params-dpi-dpj,func,x,y,covar)[0])/dparam_i
    
            second_der = (first_hi-first_lo)/dparam_j
        
        dparam_i = dparam_i/2.
        dparam_j = dparam_j/2.
        dpi = dpi/2.
        dpj = dpj/2.

    return second_der

#Run minimization for data with covariance
#and also estimate covariance of the fit parameters
def fit_with_covar(params,func,x,y,covar):

    [res, cov_x] = scipy.optimize.leastsq(get_chisq_with_covar,
                                     params,args=(func,x,y,covar))
    
    #print res

    #Get the chi^2 for this one
    chi2 = get_chisq_with_covar(res,func,x,y,covar)[0]

    #Calculate matrix of derivatives
    nparam = len(params)
    partials = np.zeros([nparam,nparam])
    
    for i in range(nparam):
        for j in range(nparam):
            if j > i:
                continue
            #print fit_chisq_partial_derivative(i,j,res,func,x,y,covar)
            partials[i,j] = fit_chisq_partial_derivative(i,j,res,func,x,y,covar)
            partials[j,i] = partials[i,j]

    #Divide by two and invert to get the covariance matrix
    if np.linalg.det(partials) == 0:
        print >> sys.stderr, "WARNING:  Hessian det == 0, can't invert"
        res_covar = partials
    else:
        res_covar = np.linalg.inv(0.5*partials)

    return [chi2, res, res_covar]

#Instead of a single matrix, covar is a set of block-diagonal matrices
def get_chisq_with_block_covar(params,func,x,y,covar):
    dx = func(params,x) - y
    
    nblocks = len(covar)
    for i in range(nblocks):
        if np.linalg.det(covar[i]/np.median(np.diag(covar[i]))) == 0:
            print >> sys.stderr, i, covar[i]
            #print >> sys.stderr, x[0:3], y[0]
            print >> sys.stderr, "ERROR: Covariance matrix is singular (det=0)"
            sys.exit(1)

    #Multiplying the differences by the inverse matrix,
    # transpose(dx) C^-1 dx
    count = 0
    chisq = 0
    for i in range(nblocks):
        nvals = len(covar[i])
        chisq_temp = np.dot(np.dot(dx[count:count+nvals],np.linalg.inv(covar[i])),
                        dx[count:count+nvals])
        chisq = chisq + chisq_temp
        count = count + nvals

    chisq = np.resize(np.array([chisq]),len(params))

    return chisq

def fit_chisq_block_partial(i,j,params,func,x,y,covar,tol_i=0.01,tol_j=0.01):
    dparam_i = tol_i*params[i]
    dparam_j = tol_j*params[j]

    dpi = np.zeros_like(params)
    dpi[i] = dparam_i/2.
    dpj = np.zeros_like(params)
    dpj[j] = dparam_j/2.
    
    second_der_start = -1000
    second_der = 1000

    while abs((second_der - second_der_start)/second_der) > 0.0001:
        #print second_der
        second_der_start = second_der
        if i == j:
            val_hi = get_chisq_with_block_covar(params+2*dpi,func,x,y,covar)[0]
            val_mid = get_chisq_with_block_covar(params,func,x,y,covar)[0]
            val_lo = get_chisq_with_block_covar(params-2*dpi,func,x,y,covar)[0]
            second_der = (val_hi - 2*val_mid + val_lo)/dparam_i/dparam_i
        else:
            first_hi = (get_chisq_with_block_covar(params+dpi+dpj,func,x,y,covar)[0] - get_chisq_with_block_covar(params-dpi+dpj,func,x,y,covar)[0])/dparam_i
            first_lo = (get_chisq_with_block_covar(params+dpi-dpj,func,x,y,covar)[0] - get_chisq_with_block_covar(params-dpi-dpj,func,x,y,covar)[0])/dparam_i
            second_der = (first_hi-first_lo)/dparam_j

        dparam_i = dparam_i/2.
        dparam_j = dparam_j/2.
        dpi = dpi/2.
        dpj = dpj/2.
        #print "     ",abs(second_der - second_der_start)/second_der    
    
    return second_der

def fit_chisq_block_partial_analy(i,j,params,func,x,y,covar,tol_i=0.01,tol_j=0.01):
    dparam_i = tol_i*params[i]
    dparam_j = tol_j*params[j]

    dpi = np.zeros_like(params)
    dpi[i] = dparam_i/2.
    dpj = np.zeros_like(params)
    dpj[j] = dparam_j/2.

    cen_chi2 = get_chisq_with_block_covar(params,func,x,y,covar)[0]

    while (get_chisq_with_block_covar(params+dpi+dpj,func,x,y,covar)[0] - cen_chi2) > 0.5:
        dpi = dpi/2.
        dpj = dpj/2.

    if i == j:
        a1 = (get_chisq_with_block_covar(params+dpi,func,x,y,covar)[0] - cen_chi2)/dpi[i]**2
        a2 = (get_chisq_with_block_covar(params-dpi,func,x,y,covar)[0] - cen_chi2)/dpi[i]**2
        second_der = (a1+a2)
    else:
        a1 = get_chisq_with_block_covar(params+dpi+dpj,func,x,y,covar)[0]
        a2 = get_chisq_with_block_covar(params+dpi-dpj,func,x,y,covar)[0]
        a3 = get_chisq_with_block_covar(params-dpi+dpj,func,x,y,covar)[0]
        a4 = get_chisq_with_block_covar(params-dpi-dpj,func,x,y,covar)[0]
        second_der = (a1-a2-a3+a4)/(4.*dpi[i]*dpj[j])

    return second_der

def fit_with_block_covar(params,func,x,y,covar):
    [res, cov_x, infodict, msg, ier] = scipy.optimize.leastsq(get_chisq_with_block_covar,params,
                                                    args=(func,x,y,covar),
                                                   full_output=True)

    print >> sys.stderr, ier, msg
    #Get chi^2 for this result
    chi2 = get_chisq_with_block_covar(res,func,x,y,covar)[0]

    #Calculate the matrix of derivatives
    nparam = len(params)
    partials = np.zeros([nparam,nparam])
    
    for i in range(nparam):
        for j in range(nparam):
            if j > i:
                continue
            #print fit_chisq_partial_derivative(i,j,res,func,x,y,covar)
            partials[i,j] = fit_chisq_block_partial(i,j,res,func,x,y,covar)
            partials[j,i] = partials[i,j]

    #Divide by two and invert to get the covariance matrix
    if np.linalg.det(partials) == 0:
        print "Partials matrix is not invertible"
        return [chi2, res, partials]
    res_covar = np.linalg.inv(0.5*partials)
    #res_covar = partials

    #res_covar = cov_x * chi2/(len(y)-len(params))

    return [chi2, res, res_covar]

def print_fit_res(npoints, chi2, res, res_covar,filename):
    f = open(filename,'w')
    
    for i in range(len(res)):
        f.write(str(res[i])+" ")
    f.write(str(npoints)+" "+str(chi2)+"\n")

    for i in range(len(res_covar)):
        for j in range(len(res_covar)):
            f.write(str(res_covar[i,j])+" ")
        f.write("\n")

    f.close()

    return
