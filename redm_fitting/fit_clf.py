#!/u/ki/dapple/bin/python

##!/usr/bin/env python

import numpy as np
import scipy

from glob import glob

import fit_with_covar

#Various routines for fitting to CLFs
#Currently set up primarily for fitting using log(L) or log(SM)
#Also set to use form Phi = dn/d log (L)

def func_cen(params,x):
    Lcen = params[0]
    sigma = params[1]

    y = np.exp(-(x-Lcen)*(x-Lcen)/sigma/sigma/2.)/np.sqrt(2*np.pi)/sigma

    return y

def func_sat(params,x):
    Lsat = params[0]
    phist = params[1]
    alpha = params[2]
    
    y = phist*(10**x/10**Lsat)**(alpha+1)*np.exp(-10**x/10**Lsat)

    return y

#Fits a set of centrals
def fit_cen_set(infiles,covarfiles,zmin,zmax,lmin,lmax):
    
    if len(infiles) != len(covarfiles):
        print >> sys.stderr, "ERROR: Data and covariance files do not match"
        sys.exit(1)

    for i in range(len(infiles)):
        cen = np.loadtxt(infiles[i])
        covar_cen = np.loadtxt(covarfiles[i])
        clist = np.where(cen[:,1] > 0)[0]
        #print cen[clist,1]

        x = cen[clist,0]
        y = cen[clist,1]
        covar_cen = covar_cen[clist,:]
        covar_cen = covar_cen[:,clist]
        #print infiles[i],covarfiles[i]

        #start = np.array([10.8, 0.2])
        start = [10.8, 0.2]
        #print start

        [chi2, res, res_covar] = fit_with_covar.fit_with_covar(start,func_cen,x,y,covar_cen)

        if chi2 > len(clist)-2:
            start = res
            [chi2, res, res_covar] = fit_with_covar.fit_with_covar(start,func_cen,x,y,covar_cen)

        print zmin[i], lmin[i], res, len(clist), len(clist)-2, chi2

    return

#Fits a set of satellites
def fit_sat_set(infiles,covarfiles,zmin,zmax,lmin,lmax,minlum=0,do_reg=False):
    
    if len(infiles) != len(covarfiles):
        print >> sys.stderr, "ERROR: Data and covariance files do not match"
        sys.exit(1)

    res_all = []
    res_covar_all = []
    for i in range(len(infiles)):
        sat = np.loadtxt(infiles[i])
        covar_sat = np.loadtxt(covarfiles[i])
        #Remove zero points and points below the fixed threshold
        slist = np.where((sat[:,1] > 0.01) & (sat[:,0] > minlum) & (sat[:,2] > 0) )[0]
        #Lowest satellite point will be spurious due to the mag limit
        slist = slist[1:]
        while sat[slist[1],1] > sat[slist[0],1]:
            slist = slist[1:]

        x = sat[slist,0]
        y = sat[slist,1]
        covar = np.copy(covar_sat[slist,:])
        covar = covar[:,slist]
        if do_reg:
            covar = fit_with_covar.regularize(covar)
        
        start = np.array([10.4+0.2*((zmax[i]+zmin[i])/2.-0.15),40,-1])
        #, 40, -1])
        #print start

        [chi2, res, res_covar] = fit_with_covar.fit_with_covar(start,func_sat,x,y,covar)

        if chi2 > len(slist[1:])-2:
            start = res
            [chi2, res, res_covar] = fit_with_covar.fit_with_covar(start,func_sat,x,y,covar)

        print zmin[i], lmin[i], res, len(slist), len(slist)-3, chi2
        res_all.append(res)
        res_covar_all.append(res_covar)

    return res_all, res_covar_all

#Print out the relevant set of files for a set of satellite fits
#Note this includes both parameters and covariances
def print_sat_set(outdir,zmin,zmax,lmin,lmax,lmed,res,res_covar):
    nfits = len(zmin)

    for i in range(nfits):
        #First, make the output file
        f = open(outdir+"param_sat_single_z_"+str(zmin[i])+"_"+str(zmax[i])+"_lm_"+str(lmin[i])[:5]+
                 "_"+str(lmax[i])[:5]+".dat",'w')
        print >> f, lmed[i], res[i][0], res[i][1],res[i][2]
        
        #Print out the covariance matrix
        for j in range(3):
            print >> f, res_covar[i][j,0], res_covar[i][j,1], res_covar[i][j,2], 0
        f.close()

    return

#Fitting functions for an entire set of related central CLFs -- 
#function of z and lambda, centrals only
def func_cenparam(params,z,richness):
    #Central parameters
    logL0 = params[0]
    acen = params[1]
    bcen = params[2]

    return [logL0 + acen*np.log10(richness/20.) + bcen*np.log10( (1+z)/1.3), params[3] ]

def func_all_cen(params,x):
    npoints = len(x)/3
    lum = np.zeros(npoints)
    z = np.zeros(npoints)
    richness = np.zeros(npoints)
    y = np.zeros(npoints)
    for i in range(npoints):
        lum[i] = x[i*3]
        z[i] = x[i*3+1]
        richness[i] = x[i*3+2]

    [Lcen, sigma] = func_cenparam(params,z,richness)
    y = np.exp(-(lum-Lcen)*(lum-Lcen)/sigma/sigma/2.)/np.sqrt(2*np.pi)/sigma
    
    return y

#Fitting functions which depend on redshift only; usable in fit_all_cen
def func_z_cenparam(params,z,richness):
    #Central parameters
    logL0 = params[0]
    bcen = params[1]
    #No lambda dependence!!!

    return [logL0 + bcen*np.log10( (1+z)/1.3), params[2] ]

def func_all_z_cen(params,x):
    npoints = len(x)/3
    lum = np.zeros(npoints)
    z = np.zeros(npoints)
    richness = np.zeros(npoints)
    y = np.zeros(npoints)
    for i in range(npoints):
        lum[i] = x[i*3]
        z[i] = x[i*3+1]
        richness[i] = x[i*3+2]

    [Lcen, sigma] = func_z_cenparam(params,z,richness)
    y = np.exp(-(lum-Lcen)*(lum-Lcen)/sigma/sigma/2.)/np.sqrt(2*np.pi)/sigma
    
    return y


def fit_all_cen(start,func,pfunc,infiles,covarfiles,zmin,zmax,lmin,lmax,
                diag_only=False,do_reg=False,lmed=[]):
    nfiles = len(infiles)
    if nfiles != len(covarfiles):
        print >> sys.stderr, "ERROR:  Number of data, covariance files do not match"
        sys.exit(1)

    zmean = (zmin+zmax)/2.
    if len(lmed) == 0:
        lmed = np.sqrt(lmin*lmax)

    #Readin and format input
    x = []
    y = []
    covar_all = []
    covar_exp = []
    for i in range(nfiles):
        #Readin
        clf = np.loadtxt(infiles[i])
        covar = np.loadtxt(covarfiles[i])
        
        #Remove zero points
        clist = np.where(clf[:,1] > 0)[0]
        clf = clf[clist]
        #print covar
        covar = covar[clist,:]
        covar = covar[:,clist]
        if do_reg:
            covar = fit_with_covar.regularize(covar)
        covar_exp.append(np.copy(covar))
        for j in range(len(clist)):
            for k in range(len(clist)):
                if diag_only and (j != k):
                    covar[j,k] = 0
                #else:
                #    covar[j,k] = covar[j,k]/clf[j,1]/clf[k,1]

        #print covar
    
        #Add points to the arrays
        for j in range(len(clist)):
            x.append(clf[j,0])
            x.append(zmean[i])
            x.append(lmed[i])
            y.append(clf[j,1])
        covar_all.append(covar)
    #y = np.log(y)

    #Run the fitting routine
    if len(start) == 0:
        start = [10.2, 1., 20., 0.1, 0.2]
    [chi2, res, res_covar] = fit_with_covar.fit_with_block_covar(start,func,x,y,covar_all)
    print len(x)/3

    #Print out chi^2 for each file for testing purposes
    count = 0
    x = np.array(x)
    y = np.array(y)
    for i in range(nfiles):
        lres = pfunc(res,zmean[i],lmed[i])
        nvals = len(covar_all[i])
        vlist = 3*count + 3*np.array(range(nvals),dtype=int)
        #print lres
        #print x[vlist], y[count:count+nvals]
        print nvals,fit_with_covar.get_chisq_with_covar(lres,func_cen,x[vlist],
                                                        y[count:count+nvals],
                                                        covar_all[i])[0],zmin[i],lmin[i]
        count += nvals

    return [chi2, len(y), res, res_covar, x, y, covar_all]


#Equivalent fitting routines for satellites
#Fitting functions which depend on redshift only; usable in fit_all_sat
def func_z_satparam(params,z,richness):
    #Satellite parameters
    logL0 = params[0]
    bsat = params[1]
    #a0 = params[3]
    #ka = params[4]
    #No lambda dependence!!!

    #return [logL0 + bsat*np.log10( (1+z)/1.3), params[2], a0 + ka*np.log10((1+z)/1.3) ]
    return [logL0 + bsat*np.log10( (1+z)/1.3), params[2], params[3] ]

def func_all_z_sat(params,x):
    npoints = len(x)/3
    lum = np.zeros(npoints)
    z = np.zeros(npoints)
    richness = np.zeros(npoints)
    y = np.zeros(npoints)
    for i in range(npoints):
        lum[i] = np.array(x[i*3])
        z[i] = x[i*3+1]
        richness[i] = x[i*3+2]

    [Lsat, phist, alpha] = func_z_satparam(params,z,richness)
    y = phist*(10**lum/10**Lsat)**(alpha+1)*np.exp(-10**lum/10**Lsat)

    mylist = np.where(y < 0)[0]
    if len(mylist) > 0:
        y[mylist] = 0*mylist+1e-10

    return y

def func_satparam(params,z,richness):
    logL0 = params[0]
    asat = params[1]
    bsat = params[2]
    phi_0 = params[3]
    a_phi = params[4]

    return [ logL0 + asat*np.log10(richness/20.)+ bsat*np.log10( (1+z)/1.3), phi_0*(richness/20.)**a_phi , params[5] ]

def func_all_sat(params,x):
    npoints = len(x)/3
    lum = np.zeros(npoints)
    z = np.zeros(npoints)
    richness = np.zeros(npoints)
    y = np.zeros(npoints)
    for i in range(npoints):
        lum[i] = np.array(x[i*3])
        z[i] = x[i*3+1]
        richness[i] = x[i*3+2]

    [Lsat, phist, alpha] = func_satparam(params,z,richness)
    y = phist*(10**lum/10**Lsat)**(alpha+1)*np.exp(-10**lum/10**Lsat)

    mylist = np.where(y < 0)[0]
    if len(mylist) > 0:
        y[mylist] = 0*mylist+1e-10

    return y

def fit_all_sat(start,func,pfunc,infiles,covarfiles,zmin,zmax,lmin,lmax,minlum=9.5,
                diag_only=False,do_reg=False,lmed=[]):
    nfiles = len(infiles)
    if nfiles != len(covarfiles):
        print >> sys.stderr, "ERROR:  Number of data, covariance files do not match"
        sys.exit(1)

    zmean = (zmin+zmax)/2.
    if len(lmed) == 0:
        lmed = np.sqrt(lmin*lmax)

    #Readin and format input
    x = []
    y = []
    covar_all = []
    covar_exp = []
    for i in range(nfiles):
        #Readin
        clf = np.loadtxt(infiles[i])
        covar = np.loadtxt(covarfiles[i])
        
        #Remove zero points
        slist = range(len(clf))
        slist = np.where( (clf[:,1] > 0) & (clf[:,0] > minlum) & (clf[:,2] > 0) )[0]
        #Lowest satellite point will be spurious due to mag limit
        slist = slist[1:]
        while clf[slist[1],1] > clf[slist[0],1]:
            slist = slist[1:]
        clf = clf[slist]
        covar = covar[slist,:]
        covar = covar[:,slist]
        if do_reg:
            covar = fit_with_covar.regularize(covar)
        covar_exp.append(np.copy(covar))
        for j in range(len(slist)):
            for k in range(len(slist)):
                if (j != k) and diag_only:
                    covar[j,k] = 0
                else:
                    covar[j,k] = covar[j,k]#/clf[j,1]/clf[k,1]

        #print i, covar[0,0], clf[0,1], slist[0], covar[0,0]*clf[0,1]**2
    
        #Add points to the arrays
        for j in range(len(clf)):
            x.append(clf[j,0])
            x.append(zmean[i])
            x.append(lmed[i])
            y.append(clf[j,1])
        covar_all.append(covar)
    #y = np.log(y)

    #Run the fitting routine
    if len(start) == 0:
        start = [10., 1., 20., -1., 0.2]
    [chi2, res, res_covar] = fit_with_covar.fit_with_block_covar(start,func,x,y,covar_all)
    print len(x)/3

    #Print out chi^2 for each file for testing purposes
    count = 0
    x = np.array(x)
    y = np.array(y)
    for i in range(nfiles):
        lres = pfunc(res,zmean[i],lmed[i])
        nvals = len(covar_all[i])
        vlist = 3*count + 3*np.array(range(nvals),dtype=int)
        
        print nvals,fit_with_covar.get_chisq_with_covar(lres,func_sat,x[vlist],
                                                        y[count:count+nvals],
                                                        covar_all[i])[0],zmin[i],lmin[i]
        count += nvals

    return [chi2, len(y), res, res_covar, x, y, covar_all]
