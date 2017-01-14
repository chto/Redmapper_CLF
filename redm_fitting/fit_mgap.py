#!/u/ki/dapple/bin/python

import os
import sys

import numpy as np

import fit_with_covar

#Various funcions for doing fitting to mgap data

def func_gaussian(params,x):
    mu = params[0]
    sigma = params[1]

    y = np.exp(-(x-mu)*(x-mu)/2./sigma/sigma)/sigma/np.sqrt(2*np.pi)
    return y

def fit_mgap_set(infiles,start=[0.2, 0.2]):
    
    for i in range(len(infiles)):
        print infiles[i]

        dat = np.loadtxt(infiles[i])
        x = dat[:,0]
        y = dat[:,1]
        #Dumb poissonian error estimate
        norm = np.sum(y)*(x[1]-x[0])
        clist = np.where(y/norm > 1e-4)[0]
        x = x[clist]
        y = y[clist]
        covar = np.diag(y)
        y = y/norm
        covar = covar/(norm**2)



        [chi2, res, res_covar] = fit_with_covar.fit_with_covar(start,func_gaussian,x,y,covar)
        if chi2 > 0.01:
            [chi2, res, res_covar] = fit_with_covar.fit_with_covar(res,func_gaussian,x,y,covar)
        print i, res, len(clist), chi2

    return
