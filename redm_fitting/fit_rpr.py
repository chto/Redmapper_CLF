#!/u/ki/dapple/bin/python

import sys
import numpy as np
import scipy

from glob import glob

import fit_with_covar

#Various routines for fitting to radial profiles
#Currently set to fit to projected NFW profile

def func_rpr(params,r):
    rs = params[0]
    rho_0 = params[1]

    x = r/rs
    y = 2*rho_0*rs/(x*x-1)

    alist = np.where(x > 1)[0]
    if len(alist) > 0:
        y[alist] = y[alist]*(1-2/np.sqrt(x[alist]*x[alist]-1)*
                             np.arctan(np.sqrt((x[alist]-1)/(x[alist]+1))) )
    alist = np.where(x < 1)[0]
    if len(alist) > 0:
        y[alist] = y[alist]*(1-2/np.sqrt(1-x[alist]*x[alist])*
                             np.arctanh(np.sqrt((1-x[alist])/(x[alist]+1))) )
    alist = np.where(x == 1)[0]
    if len(alist) > 0:
        y[alist] = alist*0 + 2*rho_0*rs/3.
    
    return y

def fit_rpr_set(infiles, covarfiles, zmin, zmax, lmin, lmax, mmin, outdir, rmin=0,start=[0.2,40],
                diag_only=False,do_reg=False):
    if len(infiles)!=len(covarfiles):
        print >> sys.stderr, "ERROR: data and covariance files do not match"
        sys.exit(1)
        
    params = np.zeros([len(infiles),2])
    for i in range(len(infiles)):
        rpr = np.loadtxt(infiles[i])
        covar = np.loadtxt(covarfiles[i])
        #clist = np.where((rpr[:,1] > 0) & (rpr[:,0] > 0.06))[0]
        clist = np.where( (rpr[:,1] > 0) & (rpr[:,0] > rmin) )[0]
        #Drop last point as well -- hits cutoff
        clist = clist[0:len(clist)-2]
        x = rpr[clist,0]
        y = rpr[clist,1]
        covar = covar[clist,:]
        covar = covar[:,clist]
        
        #Take diagonals only, for testing
        if diag_only:
            covar = np.diag(np.diag(covar))
        if do_reg:
            covar = fit_with_covar.regularize(covar)

        #start = [0.2, 40]

        [chi2, res, res_covar] = fit_with_covar.fit_with_covar(start,func_rpr,x,y,covar)
        #if chi2 > len(clist):
        if chi2 > 0.01:
            [chi2, res, res_covar] = fit_with_covar.fit_with_covar(res,func_rpr,x,y,covar)
        print zmin[i], lmin[i], mmin[i], res, len(clist), len(clist)-2, chi2
        params[i] = res
        f = open(outdir+"rpr_param_z_"+str(zmin[i])+"_"+str(zmax[i])+"_lm_"+str(lmin[i])+"_"+str(lmax[i])
                 +"_"+str(mmin[i])+"_12.dat",'w')
        print >> f, params[i][0],params[i][1], len(clist)-2, chi2
        f.close()

    return params

def func_rpr_param(params,z,richness):
    #r0 = params[0]
    #ar = params[1]
    #br = params[2]
    #kr = params[3]
    #rho_0 = params[4]
    #a_rho = params[5]
    #b_rho = params[6]

    #return [ r0*((1+z)/1.3)**(ar+kr*richness)*(richness/20.)**br , rho_0*((1+z)/1.3)**a_rho*(richness/20.)**b_rho]    
    r0 = params[0]
    br = params[1]
    rho_0 = params[2]
    b_rho = params[3]

    return [ r0*(richness/20.)**br, rho_0*(richness/20.)**b_rho]

def func_all_rpr(params,x_in):
    npoints = len(x_in)/3
    r = np.zeros(npoints)
    z = np.zeros(npoints)
    richness = np.zeros(npoints)
    y = np.zeros(npoints)
    for i in range(npoints):
        r[i] = x_in[i*3]
        z[i] = x_in[i*3+1]
        richness[i] = x_in[i*3+2]

    [rs, rho_0] = func_rpr_param(params,z,richness)
    
    #Now actually calculate the profiles
    x = r/rs
    y = 2*rho_0*rs/(x*x-1)

    alist = np.where(x > 1)[0]
    if len(alist) > 0:
        y[alist] = y[alist]*(1-2/np.sqrt(x[alist]*x[alist]-1)*
                             np.arctan(np.sqrt((x[alist]-1)/(x[alist]+1))) )
    alist = np.where(x < 1)[0]
    if len(alist) > 0:
        y[alist] = y[alist]*(1-2/np.sqrt(1-x[alist]*x[alist])*
                             np.arctanh(np.sqrt((1-x[alist])/(x[alist]+1))) )
    alist = np.where(x == 1)[0]
    if len(alist) > 0:
        y[alist] = alist*0 + 2*rho_0*rs/3.

    #Freak out a bit if rs<0
    alist = np.where(rs < 0)[0]
    if len(alist) > 0:
        y[alist] = 0*alist
        
    return y

def fit_all_rpr(start, func, pfunc, infiles, covarfiles, zmin, zmax, lmin, lmax,
                diag_only=False,do_reg=False,lmed=[],rmin=0):
    
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
        rpr = np.loadtxt(infiles[i])
        covar = np.loadtxt(covarfiles[i])
        
        #Remove zero points
        #Also removes high-r edge for remdapper limits
        clist = np.where( (rpr[:,1] > 0) & (rpr[:,0] > rmin) & 
                          (rpr[:,0] < (lmed[i]/100.)**0.2) )[0]
        clist = clist[0:len(clist)-1]
        #print clist
        rpr = rpr[clist]
        covar = covar[clist,:]
        covar = covar[:,clist]
        if do_reg:
            covar = fit_with_covar.regularize(covar)
        covar_exp.append(np.copy(covar))
        if diag_only:
            covar = np.diag(np.diag(covar))

        #print covar
    
        #Add points to the arrays
        for j in range(len(clist)):
            x.append(rpr[j,0])
            x.append(zmean[i])
            x.append(lmed[i])
            y.append(rpr[j,1])
        covar_all.append(covar)
    #y = np.log(y)

    #Run the fitting routine
    if len(start) == 0:
        start = [0.2, 0.2, 40, -0.17, 0.1]
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
        print nvals,fit_with_covar.get_chisq_with_covar(lres,func_rpr,x[vlist],
                                                        y[count:count+nvals],
                                                        covar_all[i])[0],zmin[i],lmin[i]
        count += nvals

    return [chi2, len(y), res, res_covar, x, y, covar_all]
