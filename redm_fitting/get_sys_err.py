#!/usr/bin/python

import numpy as np
import sys

import scipy.optimize

#Used for estimating systematic error that is needed to make a fit have a chi^2 of 1

def func_true_chisq(syserr, x, x_err, x_true):
    return np.sum((x - x_true)**2/(x_err**2+syserr**2))

def func_chisq_sys(syserr, x, x_err, x_true):
    return np.abs(np.sum((x - x_true)**2/(x_err**2+syserr**2))-len(x))

def get_sys_err(x, x_err, x_true,start=0.1):
    
    res = scipy.optimize.minimize(func_chisq_sys,start,args=(x, x_err, x_true))

    return res.x

def read_param_file(filename):
    dat = np.loadtxt(filename)
    z = dat[1:,0]
    x = dat[1:,1]
    x_err = (dat[1:,2]+dat[1:,3])/2.
    return z, x, x_err

def get_s82_sys_errs(cen_param,sat_param):
    indir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10_v3/"

    #Centrals
    labels = [r'$\sigma_L$','r',r'log $L_{c0}$',r'$A_L$',r'$s_c$']
    pname = ['sigma_L','r','lnLc','A_L','s_c']
    cen_syserr = np.zeros(5)

    print "Getting centrals systematic error..."
    #sigma_L
    z, x, x_err = read_param_file(indir+pname[0]+'.dat')
    x_tr = np.repeat(cen_param[0][0],4)
    cen_syserr[0] = get_sys_err(x,x_err,x_tr,start=0.1)
    print func_chisq_sys(cen_syserr[0],x,x_err,x_tr)

    #r
    z, x, x_err = read_param_file(indir+pname[1]+'.dat')
    x_tr = np.repeat(cen_param[1][0],4)
    cen_syserr[1] = get_sys_err(x,x_err,x_tr)
    print func_chisq_sys(cen_syserr[1],x,x_err,x_tr)

    #lnLc
    z, x, x_err = read_param_file(indir+pname[2]+'.dat')
    x_tr = cen_param[2][0] + cen_param[2][1]*np.log((1+z)/1.3)
    cen_syserr[2] = get_sys_err(x,x_err,x_tr,start=0.07)
    print func_chisq_sys(cen_syserr[2],x,x_err,x_tr), func_true_chisq(0.,x,x_err,x_tr), func_true_chisq(0.07,x,x_err,x_tr), func_true_chisq(cen_syserr[2],x,x_err,x_tr)

    #A_L
    z, x, x_err = read_param_file(indir+pname[3]+'.dat')
    x_tr = np.repeat(cen_param[3][0],4)
    cen_syserr[3] = get_sys_err(x,x_err,x_tr)
    print func_chisq_sys(cen_syserr[3],x,x_err,x_tr), func_true_chisq(0.,x,x_err,x_tr), func_true_chisq(cen_syserr[3],x,x_err,x_tr)

    #s_c
    #Skip

    #Satellites
    labels = [r'ln $\phi_0$',r'$A_\phi$',r'log $L_{s0}$',r'$A_s$',r'$\alpha$',r'$s_s$']
    pname = ['lnphi0','A_phi','lnLs','A_s','alpha','s_s']
    sat_syserr = np.zeros(6)

    #lnphi0
    z, x, x_err = read_param_file(indir+pname[0]+'.dat')
    x_tr = sat_param[0][0] + sat_param[0][1]*np.log((1+z)/1.3)
    sat_syserr[0] = get_sys_err(x,x_err,x_tr,start=0.03)
    print func_chisq_sys(sat_syserr[0],x,x_err,x_tr), func_true_chisq(0.,x,x_err,x_tr), func_true_chisq(sat_syserr[0],x,x_err,x_tr)

    #A_phi
    z, x, x_err = read_param_file(indir+pname[1]+'.dat')
    x_tr = sat_param[1][0] + sat_param[1][1]*np.log((1+z)/1.3)
    sat_syserr[1] = get_sys_err(x,x_err,x_tr,start=0.09)
    print "A_phi: ", func_chisq_sys(sat_syserr[1],x,x_err,x_tr), func_true_chisq(0.09,x,x_err,x_tr), func_true_chisq(sat_syserr[1],x,x_err,x_tr)

    #lnLs
    z, x, x_err = read_param_file(indir+pname[2]+'.dat')
    x_tr = sat_param[2][0] + sat_param[2][1]*np.log((1+z)/1.3)
    sat_syserr[2] = get_sys_err(x,x_err,x_tr,start=0.04)
    print "lnLs:", func_chisq_sys(sat_syserr[2],x,x_err,x_tr), func_true_chisq(0.04,x,x_err,x_tr), func_true_chisq(sat_syserr[2],x,x_err,x_tr)

    #A_s
    z, x, x_err = read_param_file(indir+pname[3]+'.dat')
    x_tr = sat_param[3][0] + sat_param[3][1]*np.log((1+z)/1.3)
    sat_syserr[3] = get_sys_err(x,x_err,x_tr)
    print func_chisq_sys(sat_syserr[3],x,x_err,x_tr), func_true_chisq(0.,x,x_err,x_tr), func_true_chisq(sat_syserr[3],x,x_err,x_tr)

    #alpha
    z, x, x_err = read_param_file(indir+pname[4]+'.dat')
    x_tr = sat_param[4][0] + sat_param[4][1]*np.log((1+z)/1.3)
    sat_syserr[4] = get_sys_err(x,x_err,x_tr)
    print func_chisq_sys(sat_syserr[4],x,x_err,x_tr), func_true_chisq(0.,x,x_err,x_tr), func_true_chisq(sat_syserr[4],x,x_err,x_tr)

    #Skipping s_s for now
    
    return cen_syserr, sat_syserr
