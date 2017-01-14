#/usr/bin/python

import numpy as np
import matplotlib.pyplot as pyplot
from glob import glob

import fit_plm
import fit_psat
import fit_cen_bsat

#Scripts for plotting the central/brightest satellite distribution

def plot_cen_bsat_set(indir,lm_param, param,
                      lm_min,lm_max,lm_med,mass_param, zmin, zmax, 
                      mf_all, outfile, param_alt=[],lbin=20):

    zmid = (zmin+zmax)/2.
    nz = len(zmin)
    nlm = len(lm_min)

    sigma_lm = lm_param[-1]

    pyplot.figure(1,[11.,8.5])
    for i in range(nlm):
        for j in range(nz):
            #Read in the distribution data
            dist = np.loadtxt(indir+"dist_cen_sat_bright_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            dist_err = np.loadtxt(indir+"dist_err_cen_sat_bright_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            
            lumbins = dist[:,0]
            dist = dist[:,1:]
            dist_err = np.sqrt(dist_err[:,1:])
            
            pyplot.subplot(nlm,nz,i*nz+j+1)
            #Currently only plotting one slice at fixed Lcen
            pyplot.semilogy(lumbins,dist[lbin,:],'ko')
            pyplot.errorbar(lumbins,dist[lbin,:],dist_err[lbin,:],fmt=None,ecolor='k')

            #Plot the analytic fit
            fit = 0*lumbins
            for k in range(len(lumbins)):
                fit[k] = fit_cen_bsat.p_cen_bsat(lumbins[lbin]*np.log(10.),
                                                 lumbins[k]*np.log(10.),
                                                 np.log(lm_med[i]),
                                                 zmid[j],mass_param[j],lm_param,
                                                 sigma_lm,param)
            pyplot.plot(lumbins,fit,'b')
            
            if(len(param_alt) > 0):                
                for k in range(len(lumbins)):
                    fit[k] = fit_cen_bsat.p_cen_bsat(lumbins[lbin]*np.log(10.),
                                                     lumbins[k]*np.log(10.),
                                                     np.log(lm_med[i]),
                                                     zmid[j],mass_param[j],
                                                     lm_param,
                                                     sigma_lm,param_alt)
                pyplot.plot(lumbins,fit,'g')

            #And label some things
            if i == nlm-1:
                pyplot.xlabel(r'$L_i$ [$L_\odot/h^2$]')
            if j == 0:
                pyplot.ylabel(r'$\Phi$ $[(d log L)^{-1}]$')
                pyplot.text(9.7,20,str(lm_min[i])[0:4]+r'$<\lambda<$'+str(lm_max[i])[0:4])
            if i == 0:
                pyplot.title(str(zmin[j])+'<z<'+str(zmax[j]))
            pyplot.xlim(9.5,11.5)
            pyplot.ylim(1e-3,100)
                

    pyplot.savefig(outfile,orientation='landscape')
    pyplot.clf()

    return

#Orthogonal plotting -- Lcen at fixed Lbrightest
def plot_cen_bsat_set_alt(indir,lm_param, param,
                      lm_min,lm_max,lm_med,mass_param, zmin, zmax, 
                      mf_all, outfile, param_alt=[],lbin=20):

    zmid = (zmin+zmax)/2.
    nz = len(zmin)
    nlm = len(lm_min)

    sigma_lm = lm_param[-1]

    pyplot.figure(1,[11.,8.5])
    for i in range(nlm):
        for j in range(nz):
            #Read in the distribution data
            dist = np.loadtxt(indir+"dist_cen_sat_bright_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            dist_err = np.loadtxt(indir+"dist_err_cen_sat_bright_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            
            lumbins = dist[:,0]
            dist = dist[:,1:]
            dist_err = np.sqrt(dist_err[:,1:])
            
            pyplot.subplot(nlm,nz,i*nz+j+1)
            #Currently only plotting one slice at fixed Lcen
            pyplot.semilogy(lumbins,dist[:,lbin],'ko')
            pyplot.errorbar(lumbins,dist[:,lbin],dist_err[lbin,:],fmt=None,ecolor='k')

            #Plot the analytic fit
            fit = 0*lumbins
            for k in range(len(lumbins)):
                fit[k] = fit_cen_bsat.p_cen_bsat(lumbins[k]*np.log(10.),
                                                 lumbins[lbin]*np.log(10.),
                                                 np.log(lm_med[i]),
                                                 zmid[j],mass_param[j],lm_param,
                                                 sigma_lm,param)
            pyplot.plot(lumbins,fit,'b')
            
            if(len(param_alt) > 0):                
                for k in range(len(lumbins)):
                    fit[k] = fit_cen_bsat.p_cen_bsat(lumbins[k]*np.log(10.),
                                                     lumbins[lbin]*np.log(10.),
                                                     np.log(lm_med[i]),
                                                     zmid[j],mass_param[j],
                                                     lm_param,
                                                     sigma_lm,param_alt)
                pyplot.plot(lumbins,fit,'g')

            #And label some things
            if i == nlm-1:
                pyplot.xlabel(r'$L_i$ [$L_\odot/h^2$]')
            if j == 0:
                pyplot.ylabel(r'$\Phi$ $[(d log L)^{-1}]$')
                pyplot.text(9.7,20,str(lm_min[i])[0:4]+r'$<\lambda<$'+str(lm_max[i])[0:4])
            if i == 0:
                pyplot.title(str(zmin[j])+'<z<'+str(zmax[j]))
            pyplot.xlim(9.5,11.5)
            pyplot.ylim(1e-3,100)
                

    pyplot.savefig(outfile,orientation='landscape')
    pyplot.clf()

    return

#Wrapper for quickly plotting the S82 data
def plot_s82_cbsat(outdir):
    h = 0.6704
    Mpiv = 2.35e14

    #New mass-lambda parameters
    lm_param = [Mpiv, 0.857, 1.547, 2.7226, 0.1842]

    #Set up the input directories
    indir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/s82_v5.10_uber/"

    #Set up the redshift ranges
    zmin = [0.1, 0.2, 0.3, 0.4, 0.5]
    zmin = np.array(zmin)
    zmax = zmin + 0.1
    
    #Set up lambda ranges
    lm_min = [10., 20., 40.]
    lm_max = [20., 100., 100.]
    lm_min = np.array(lm_min)
    lm_max = np.array(lm_max)
    lm_med = [13., 27.4, 47.5]

    #Get file list for halo masses
    mf_files = glob(indir+"../mass_functions_planck/mf_planck_*.dat")
    mf_files = np.array(mf_files)
    mf_files.sort()
    mf_files = mf_files[[2,6,12,14,16]]

    #Make the mass_param list for the CLFs
    mf_all = []
    mass_param = []
    for i in range(len(mf_files)):
        mf = np.loadtxt(mf_files[i])
        mf[:,0] = mf[:,0]/h
        mf[:,1] = mf[:,1]*h**3
        mf_all.append(mf)
        mass_param.append(fit_plm.nm_approx_third(mf[:,0], mf[:,1], Mpiv))
    mass_param = np.array(mass_param)
    mf_all = np.array(mf_all)

    param = [0.36, -0.9, 24.74, 0.40, 1.21, 0.3, -0.05, 0., 0., 24.34, 0.4, 1.2, 2.]
    #param_alt = [0.36, -0.9, 24.74, 0.40, 1.21, 0.3, -0.05, -.9, 0.9, 24.34, 0.4, 1.2, 2.]
    param_alt = [0.2765, 0.76, 24.6165, 0.4305, 1.465, 0.2234, -0.1733, -0.3364, -0.3128, 24.353, 0.3373, 1.523, 4.03]

    #Plot the distribution
    plot_cen_bsat_set(indir,lm_param, param,
                      lm_min,lm_max,lm_med,mass_param[[1,2,4]], 
                      zmin[[1,2,4]], zmax[[1,2,4]], 
                      mf_all[[1,2,4]], outdir+"dist_cen_bsat_m10.6.ps",
                      param_alt=param_alt,lbin=20)

    #Plot the orthogonal distribution
    plot_cen_bsat_set_alt(indir,lm_param, param,
                          lm_min,lm_max,lm_med,mass_param[[1,2,4]], 
                          zmin[[1,2,4]], zmax[[1,2,4]], 
                          mf_all[[1,2,4]], outdir+"dist_cen_bsat_alt_m10.6.ps",
                          param_alt=param_alt,lbin=20)

    #Plot the distribution
    plot_cen_bsat_set(indir,lm_param, param,
                      lm_min,lm_max,lm_med,mass_param[[1,2,4]], 
                      zmin[[1,2,4]], zmax[[1,2,4]], 
                      mf_all[[1,2,4]], outdir+"dist_cen_bsat_m11.0.ps",
                      param_alt=param_alt,lbin=25)

    #Plot the orthogonal distribution
    plot_cen_bsat_set_alt(indir,lm_param, param,
                          lm_min,lm_max,lm_med,mass_param[[1,2,4]], 
                          zmin[[1,2,4]], zmax[[1,2,4]], 
                          mf_all[[1,2,4]], outdir+"dist_cen_bsat_alt_m11.0.ps",
                          param_alt=param_alt,lbin=25)

    return
