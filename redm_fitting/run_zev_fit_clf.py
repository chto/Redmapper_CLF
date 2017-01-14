#!/usr/bin/python

import numpy as np
import sys
import matplotlib.pyplot as pyplot

import run_zev_fit as run_zev

#General function for making individual parameter output files
def print_param_zev_basic(files,zmin,zmax,indir,sat=False):
    nz = len(zmin)
    nparam = len(files)

    #Opening the output files to clear them
    for i in range(nparam):
        f = open(files[i],'w')
        f.close()

    for i in range(nz):
        if sat:
            chain = np.loadtxt(indir+"chain_sat_z_"+str(zmin[i])+"_"+str(zmax[i])+".dat")
        else:
            chain = np.loadtxt(indir+"chain_cen_z_"+str(zmin[i])+"_"+str(zmax[i])+".dat")
        param_best = chain[np.argmax(chain[:,-1])]
        param = np.median(chain[10000:],axis=0)
        ehi = np.percentile(chain[10000:],84,axis=0)-param
        elo = param - np.percentile(chain[10000:],16,axis=0)

        zmid = (zmin[i]+zmax[i])/2.
        for j in range(nparam):
            f = open(files[j],'a')
            print >> f, zmid, param[j], ehi[j], elo[j], param_best[j]
            f.close()

    return

#Main function for generating the parameters(z) and errors
def make_param_zev(indir,zmin,zmax):
    nz = len(zmin)
    
    #Centrals first
    #List of filenames
    files = np.array([indir+"sigma_L.dat",indir+"r.dat",indir+"lnLc.dat",indir+"A_L.dat",indir+"s_c.dat"])
        
    print_param_zev_basic(files,zmin,zmax,indir)

    #Now satellites
    files = np.array([indir+"lnphi0.dat",indir+"A_phi.dat",indir+"lnLs.dat",indir+"A_s.dat",indir+"alpha.dat",indir+"s_s.dat"])    
    
    print_param_zev_basic(files,zmin,zmax,indir,sat=True)

    return

#Format and set up fit for a single parameter across several surveys
def fit_across_surveys_single(dirlist,zmin,pname,
                              start=[],fit_type='power',syserr=0):
    nsets = len(dirlist)
    #data readin
    data = []
    npoints = 0
    for i in range(nsets):
        data.append(np.loadtxt(dirlist[i]+pname+".dat"))
        xlist = np.where(data[i][:,0] > zmin[i])[0]
        data[i] = data[i][xlist]
        npoints = npoints + len(data[i])
    
    #Reformat for fitting
    z = np.zeros(npoints)
    x = np.zeros(npoints)
    err = np.zeros(npoints)

    count = 0
    for i in range(nsets):
        for j in range(len(data[i])):
            z[count] = data[i][j,0]
            x[count] = data[i][j,1]
            err[count] = (data[i][j,2]+data[i][j,3])/2.
            count = count+1
    
    #Run the fit
    param, ehi, elo, chain, prob = run_zev.general_zev_fit(z,x,err,start=start,fit_type=fit_type)    

    return data, param, ehi, elo

#Loops over all surveys.  Spits out desired parameters
def fit_across_surveys_all(dirlist,zmin,syserr_cen=[],syserr_sat=[]):
    fit_cen = np.zeros([5,2])
    err_cen = np.zeros([5,2,2])
    pname_cen = ['sigma_L','r','lnLc','A_L','s_c']

    if len(syserr_cen) == 0:
        syserr_cen = np.zeros(5)
    
    #sigma_L
    dat_t, param_t, ehi_t, elo_t = fit_across_surveys_single(dirlist,zmin,'sigma_L',
                                                             start=[0.3,1.],fit_type='constant')
    fit_cen[0,0] = param_t[0]
    err_cen[0,0,0] = ehi_t[0]
    err_cen[0,0,1] = elo_t[0]

    #r
    dat_t, param_t, ehi_t, elo_t = fit_across_surveys_single(dirlist,zmin,'r',
                                                             start=[-0.5,1.],fit_type='constant')
    fit_cen[1,0] = param_t[0]
    err_cen[1,0,0] = ehi_t[0]
    err_cen[1,0,1] = elo_t[0]

    #lnLc
    dat_t, param_t, ehi_t, elo_t = fit_across_surveys_single(dirlist,zmin,'lnLc',
                                                             start=[25.,1.2,1.],fit_type='power_log')
    fit_cen[2] = param_t[0:2]
    err_cen[2,:,0] = ehi_t[0:2]
    err_cen[2,:,1] = elo_t[0:2]

    #A_L
    dat_t, param_t, ehi_t, elo_t = fit_across_surveys_single(dirlist,zmin,'A_L',
                                                             start=[0.3,1.],fit_type='constant')
    fit_cen[3,0] = param_t[0]
    err_cen[3,0,0] = ehi_t[0]
    err_cen[3,0,1] = elo_t[0]

    #s_c
    dat_t, param_t, ehi_t, elo_t = fit_across_surveys_single(dirlist,zmin,'s_c',
                                                             start=[1.,1.],fit_type='constant')
    fit_cen[4][0] = param_t[0]
    err_cen[4,0,0] = ehi_t[0]
    err_cen[4,0,1] = elo_t[0]

    fit_sat = np.zeros([6,2])
    err_sat = np.zeros([6,2,2])
    pname_sat = ['lnphi0','A_phi','lnLs','A_s','alpha','s_s']

    #lnphi0
    dat_t, param_t, ehi_t, elo_t = fit_across_surveys_single(dirlist,zmin,'lnphi0',
                                                             start=[3.8,0.8,1.],fit_type='power_log')
    fit_sat[0] = param_t[0:2]
    err_sat[0,:,0] = ehi_t[0:2]
    err_sat[0,:,1] = elo_t[0:2]

    #A_phi
    dat_t, param_t, ehi_t, elo_t = fit_across_surveys_single(dirlist,zmin,'A_phi',
                                                             start=[0.8,0.,1.],fit_type='power_log')
    fit_sat[1] = param_t[0:2]
    err_sat[1,:,0] = ehi_t[0:2]
    err_sat[1,:,1] = elo_t[0:2]

    #lnLs
    dat_t, param_t, ehi_t, elo_t = fit_across_surveys_single(dirlist,zmin,'lnLs',
                                                             start=[23.5,1.5,1.],fit_type='power_log')
    fit_sat[2] = param_t[0:2]
    err_sat[2,:,0] = ehi_t[0:2]
    err_sat[2,:,1] = elo_t[0:2]
    
    #A_s
    dat_t, param_t, ehi_t, elo_t = fit_across_surveys_single(dirlist,zmin,'A_s',
                                                             start=[0.,0.,1.],fit_type='power_log')
    fit_sat[3] = param_t[0:2]
    err_sat[3,:,0] = ehi_t[0:2]
    err_sat[3,:,1] = elo_t[0:2]
    
    #alpha
    dat_t, param_t, ehi_t, elo_t = fit_across_surveys_single(dirlist,zmin,'alpha',
                                                             start=[-0.8,0.,1.],fit_type='power_log')
    fit_sat[4] = param_t[0:2]
    err_sat[4,:,0] = ehi_t[0:2]
    err_sat[4,:,1] = elo_t[0:2]
    
    #s_s
    dat_t, param_t, ehi_t, elo_t = fit_across_surveys_single(dirlist,zmin,'s_s',
                                                             start=[1.,1.],fit_type='constant')
    fit_sat[5][0] = param_t[0]
    err_sat[5,0,0] = ehi_t[0]
    err_sat[5,0,1] = elo_t[0]
    
    return fit_cen, err_cen, fit_sat, err_sat

#Plot the fitting results
def plot_zev_fit_cen(dirlist,zmin,fit_cen,err_cen,outfile,zlim=[0.1,0.6],
                     fit_cen_alt=[],err_cen_alt=[],syserr=[]):
    #labeling
    labels = [r'$\sigma_L$','r',r'log $L_{c0}$',r'$A_L$',r'$s_c$']
    pname = ['sigma_L','r','lnLc','A_L','s_c']
    fac = [np.log(10), 1., np.log(10), 1., 1., 1.]
    nparams = len(labels)
    zvals = np.array(range(1000))*0.001

    pyplot.figure(1,[11,8.5])
    pyplot.subplots_adjust(wspace=0.4)
    colors = ['k','b','r','g']
    symbols = ['o','^','x']
    for i in range(nparams):
        pyplot.subplot(2,3,i+1)
        for j in range(len(dirlist)):
            dat = np.loadtxt(dirlist[j]+pname[i]+".dat")
            dat = dat[np.where(dat[:,0] > zmin[j])[0]]
            if j == 1 and len(syserr) > 0 and i != nparams-1:
                pyplot.errorbar(dat[:,0],dat[:,1]/fac[i],yerr=[np.sqrt(dat[:,2]**2+syserr[i]**2)/fac[i],np.sqrt(dat[:,3]**2+syserr[i]**2)/fac[i]],ecolor=colors[j+1],fmt=None)
            pyplot.plot(dat[:,0],dat[:,1]/fac[i],colors[j]+symbols[j])
            pyplot.errorbar(dat[:,0],dat[:,1]/fac[i],yerr=[dat[:,2]/fac[i],dat[:,3]/fac[i]],ecolor=colors[j],fmt=None)

        if i < nparams-1:
            pyplot.plot(zvals, (fit_cen[i][0] + fit_cen[i][1]*np.log((1+zvals)/1.3))/fac[i],'k')
            pyplot.plot(zvals, (fit_cen[i][0] + err_cen[i][0,1] + fit_cen[i][1]*np.log((1+zvals)/1.3))/fac[i],'k--')
            pyplot.plot(zvals, (fit_cen[i][0] - err_cen[i][0,0] + fit_cen[i][1]*np.log((1+zvals)/1.3))/fac[i],'k--') 
        if len(fit_cen_alt)>0:
            pyplot.plot(zvals, (fit_cen_alt[i][0] + fit_cen_alt[i][1]*np.log((1+zvals)/1.3))/fac[i],'r')
            pyplot.plot(zvals, (fit_cen_alt[i][0] + err_cen_alt[i][0,1] + fit_cen_alt[i][1]*np.log((1+zvals)/1.3))/fac[i],'r--')
            pyplot.plot(zvals, (fit_cen_alt[i][0] - err_cen_alt[i][0,0] + fit_cen_alt[i][1]*np.log((1+zvals)/1.3))/fac[i],'r--')
            
        
        if i == 1:
            pyplot.ylim([-1.,0.7])

        pyplot.xlim(zlim)
        pyplot.ylabel(labels[i], fontsize=18)
        pyplot.xlabel('z', fontsize=18)
            
    pyplot.margins(0,0)
    pyplot.savefig(outfile,orientation='landscape')
    pyplot.clf()
    return

def plot_zev_fit_sat(dirlist,zmin,fit_sat,err_sat,outfile,zlim=[0.1,0.6],
                     fit_sat_alt=[],err_sat_alt=[],syserr=[]):
    #labeling
    labels = [r'ln $\phi_0$',r'$A_\phi$',r'log $L_{s0}$',r'$A_s$',r'$\alpha$',r'$s_s$']
    pname = ['lnphi0','A_phi','lnLs','A_s','alpha','s_s']
    fac = [1., 1., np.log(10), 1., 1., 1.]
    nparams = len(labels)
    zvals = np.array(range(1000))*0.001

    pyplot.figure(1,[11,8.5])
    pyplot.subplots_adjust(wspace=0.4)
    colors = ['k','b','r','g']
    symbols = ['o','^','x']
    for i in range(nparams):
        pyplot.subplot(2,3,i+1)
        for j in range(len(dirlist)):
            dat = np.loadtxt(dirlist[j]+pname[i]+".dat")
            dat = dat[np.where(dat[:,0] > zmin[j])[0]]
            if j == 1 and len(syserr)>0 and i != nparams-1:
                pyplot.errorbar(dat[:,0],dat[:,1]/fac[i],yerr=[np.sqrt(dat[:,2]**2+syserr[i]**2)/fac[i],np.sqrt(dat[:,3]**2+syserr[i]**2)/fac[i]],ecolor=colors[j+1],fmt=None)
            pyplot.plot(dat[:,0],dat[:,1]/fac[i],colors[j]+symbols[j])
            pyplot.errorbar(dat[:,0],dat[:,1]/fac[i],yerr=[dat[:,2]/fac[i],dat[:,3]/fac[i]],ecolor=colors[j],fmt=None)

        if i < nparams-1:
            pyplot.plot(zvals, (fit_sat[i][0] + fit_sat[i][1]*np.log((1+zvals)/1.3))/fac[i],'k')
            pyplot.plot(zvals, (fit_sat[i][0] + err_sat[i][0,1] + fit_sat[i][1]*np.log((1+zvals)/1.3))/fac[i],'k--')
            pyplot.plot(zvals, (fit_sat[i][0] - err_sat[i][0,0] + fit_sat[i][1]*np.log((1+zvals)/1.3))/fac[i],'k--')
        
        if len(fit_sat_alt)>0:
            pyplot.plot(zvals, (fit_sat_alt[i][0] + fit_sat_alt[i][1]*np.log((1+zvals)/1.3))/fac[i],'r')
            pyplot.plot(zvals, (fit_sat_alt[i][0] + err_sat_alt[i][0,1] + fit_sat_alt[i][1]*np.log((1+zvals)/1.3))/fac[i],'r--')
            pyplot.plot(zvals, (fit_sat_alt[i][0] - err_sat_alt[i][0,0] + fit_sat_alt[i][1]*np.log((1+zvals)/1.3))/fac[i],'r--')

        pyplot.xlim(zlim)
        pyplot.ylabel(labels[i], fontsize=18)
        pyplot.xlabel('z', fontsize=18)
            
    pyplot.margins(0,0)
    pyplot.savefig(outfile,orientation='landscape')
    pyplot.clf()
        
    return
