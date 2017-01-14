#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.ticker as ticker
import matplotlib.axes as axes
import sys
from glob import glob

import cosmo
import mcmc_errors as mcmc

#Various functions for making main evolution comparison plots

#Reads in, converts values to log10, and reorders parameters appropriately
def read_sat_chain(filename):
    chain = np.loadtxt(filename)

    chain = chain[:,[0,1,6,2,3,4,5,7,8]]
    chain[:,3] = chain[:,3]/np.log(10.)

    return chain

def plot_ev_test(vals_data,outdir,
                 vals_data_dr8=[]):
    #labeling
    labels = [r'ln $\phi_0$',r'$A_\phi$',r'log $L_{s0}$',r'$A_s$',r'$\alpha$',r'$s_s$']
    
    nparams = len(labels)

    #Read in the S82 data
    chain_s82 = read_sat_chain("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10/chain_sat_ev_all.dat")
    vals_truth, vals_ehi, vals_elo = mcmc.get_errors(chain_s82[20000:,:])
    vals_slope = np.zeros(6)
    print vals_truth
    vals_slope[0] = vals_truth[2]
    vals_slope[2] = vals_truth[5]
    vals_truth = vals_truth[[0,1,3,4,6,7]]
    vals_ehi = vals_ehi[[0,1,3,4,6,7]]
    vals_elo = vals_elo[[0,1,3,4,6,7]]
    
    #DR8 data
    chain_dr8 = read_sat_chain("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10_cencorr/chain_sat_ev_all.dat")
    vals_truth_dr8, vals_ehi_dr8, vals_elo_dr8 = mcmc.get_errors(chain_dr8[20000:,:])
    print vals_truth_dr8
    vals_slope_dr8 = np.zeros(6)
    vals_slope_dr8[0] = vals_truth_dr8[2]
    vals_slope_dr8[2] = vals_truth_dr8[5]
    vals_truth_dr8 = vals_truth_dr8[[0,1,3,4,6,7]]
    vals_ehi_dr8 = vals_ehi_dr8[[0,1,3,4,6,7]]
    vals_elo_dr8 = vals_elo_dr8[[0,1,3,4,6,7]]

    #vals_truth_dr8 = [3.91, 0.78, 23.105/np.log(10), 0.045, -0.85, 1.5]
    #vals_slope_dr8 = [0.6, 0., 1.5, 0., 0., 0.]

    zvals = np.array(range(500))*0.001+0.1
    fac = [1., 1., np.log(10), 1., 1., 1.]
    pyplot.figure(1,[11.,8.5])
    pyplot.subplots_adjust(wspace = 0.4)
    for i in range(len(labels)):
        print i, vals_truth[i], vals_slope[i]
        pyplot.subplot(2,3,i+1)
        #DR8 part
        pyplot.plot(zvals,(vals_truth_dr8[i]+vals_slope_dr8[i]*np.log10(1+zvals)),'k')
        pyplot.plot(zvals,(vals_truth_dr8[i]+vals_ehi_dr8[i]+vals_slope_dr8[i]*np.log10(1+zvals)),'k--')
        pyplot.plot(zvals,(vals_truth_dr8[i]-vals_elo_dr8[i]+vals_slope_dr8[i]*np.log10(1+zvals)),'k--')

        if len(vals_data_dr8) > 0:
            pyplot.plot(vals_data_dr8[:,0],vals_data_dr8[:,1+3*i]/fac[i],'ko')
            pyplot.errorbar(vals_data_dr8[:,0],vals_data_dr8[:,1+3*i]/fac[i],yerr=[vals_data_dr8[:,3+3*i]/fac[i],vals_data_dr8[:,2+3*i]/fac[i]],fmt=None,ecolor='k')

        #S82 parts
        pyplot.plot(zvals,(vals_truth[i]+vals_slope[i]*np.log10(1+zvals)),'b')
        pyplot.plot(zvals,(vals_truth[i]+vals_ehi[i]+vals_slope[i]*np.log10(1+zvals)),'b--')
        pyplot.plot(zvals,(vals_truth[i]-vals_elo[i]+vals_slope[i]*np.log10(1+zvals)),'b--')

        pyplot.plot(vals_data[1:,0],vals_data[1:,1+3*i]/fac[i],'bo')

        pyplot.errorbar(vals_data[1:,0],vals_data[1:,1+3*i]/fac[i],yerr=[vals_data[1:,3+3*i]/fac[i],vals_data[1:,2+3*i]/fac[i]],fmt=None,ecolor='b')
        pyplot.ylabel(labels[i])
        pyplot.xlabel('z')

    #pyplot.tight_layout()
    pyplot.margins(0,0)
    pyplot.savefig(outdir+"pev_all.ps",orientation='landscape')
    pyplot.clf()

    return

#Reads in, converts values to log10, and reorders parameters appropriately
def read_sat_chain(filename):
    chain = np.loadtxt(filename)

    chain = chain[:,[0,1,6,2,3,4,5,7,8]]
    chain[:,3] = chain[:,3]/np.log(10.)

    return chain

#Adds in the SVA1 comparison
def plot_ev_test_sva(vals_data,outdir,
                     vals_data_dr8=[],vals_sva=[]):
    #labeling
    labels = [r'ln $\phi_0$',r'$A_\phi$',r'log $L_{s0}$',r'$A_s$',r'$\alpha$',r'$s_s$']
    
    nparams = len(labels)

    #Read in the S82 data
    chain_s82 = read_sat_chain("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10/chain_sat_ev_all.dat")
    vals_truth, vals_ehi, vals_elo = mcmc.get_errors(chain_s82[20000:,:])
    vals_slope = np.zeros(6)
    print vals_truth
    vals_slope[0] = vals_truth[2]
    vals_slope[2] = vals_truth[5]
    vals_truth = vals_truth[[0,1,3,4,6,7]]
    vals_ehi = vals_ehi[[0,1,3,4,6,7]]
    vals_elo = vals_elo[[0,1,3,4,6,7]]
    
    #DR8 data
    chain_dr8 = read_sat_chain("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10_cencorr/chain_sat_ev_all.dat")
    vals_truth_dr8, vals_ehi_dr8, vals_elo_dr8 = mcmc.get_errors(chain_dr8[20000:,:])
    print vals_truth_dr8
    vals_slope_dr8 = np.zeros(6)
    vals_slope_dr8[0] = vals_truth_dr8[2]
    vals_slope_dr8[2] = vals_truth_dr8[5]
    vals_truth_dr8 = vals_truth_dr8[[0,1,3,4,6,7]]
    vals_ehi_dr8 = vals_ehi_dr8[[0,1,3,4,6,7]]
    vals_elo_dr8 = vals_elo_dr8[[0,1,3,4,6,7]]

    #SVA1 data
    chain_sv = read_sat_chain("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_sva1_v6.1.3/chain_sat_ev_all.dat")
    vals_truth_sva, vals_ehi_sva, vals_elo_sva = mcmc.get_errors(chain_sv[20000:,:])
    vals_slope_sva = np.zeros(6)
    vals_slope_sva[0] = vals_truth_sva[2]
    vals_slope_sva[2] = vals_truth_sva[5]
    vals_truth_sva = vals_truth_sva[[0,1,3,4,6,7]]
    vals_ehi_sva = vals_ehi_sva[[0,1,3,4,6,7]]
    vals_elo_sva = vals_elo_sva[[0,1,3,4,6,7]]

    #vals_truth_dr8 = [3.91, 0.78, 23.105/np.log(10), 0.045, -0.85, 1.5]
    #vals_slope_dr8 = [0.6, 0., 1.5, 0., 0., 0.]

    zvals = np.array(range(800))*0.001+0.1
    fac = [1., 1., np.log(10), 1., 1., 1.]
    pyplot.figure(1,[11.,8.5])
    pyplot.subplots_adjust(wspace = 0.4)
    for i in range(len(labels)):
        print i, vals_truth[i], vals_slope[i]
        pyplot.subplot(2,3,i+1)
        #DR8 part
        pyplot.plot(zvals,(vals_truth_dr8[i]+vals_slope_dr8[i]*np.log10(1+zvals)),'k')
        pyplot.plot(zvals,(vals_truth_dr8[i]+vals_ehi_dr8[i]+vals_slope_dr8[i]*np.log10(1+zvals)),'k--')
        pyplot.plot(zvals,(vals_truth_dr8[i]-vals_elo_dr8[i]+vals_slope_dr8[i]*np.log10(1+zvals)),'k--')

        if len(vals_data_dr8) > 0:
            pyplot.plot(vals_data_dr8[:,0],vals_data_dr8[:,1+3*i]/fac[i],'ko')
            pyplot.errorbar(vals_data_dr8[:,0],vals_data_dr8[:,1+3*i]/fac[i],yerr=[vals_data_dr8[:,3+3*i]/fac[i],vals_data_dr8[:,2+3*i]/fac[i]],fmt=None,ecolor='k')

        #S82 parts
        pyplot.plot(zvals,(vals_truth[i]+vals_slope[i]*np.log10(1+zvals)),'b')
        pyplot.plot(zvals,(vals_truth[i]+vals_ehi[i]+vals_slope[i]*np.log10(1+zvals)),'b--')
        pyplot.plot(zvals,(vals_truth[i]-vals_elo[i]+vals_slope[i]*np.log10(1+zvals)),'b--')

        pyplot.plot(vals_data[1:,0],vals_data[1:,1+3*i]/fac[i],'bo')

        pyplot.errorbar(vals_data[1:,0],vals_data[1:,1+3*i]/fac[i],yerr=[vals_data[1:,3+3*i]/fac[i],vals_data[1:,2+3*i]/fac[i]],fmt=None,ecolor='b')

        #SVA1 part
        pyplot.plot(zvals,(vals_truth_sva[i]+vals_slope_sva[i]*np.log10(1+zvals)),'r')
        pyplot.plot(zvals,(vals_truth_sva[i]+vals_ehi_sva[i]+vals_slope_sva[i]*np.log10(1+zvals)),'r--')
        pyplot.plot(zvals,(vals_truth_sva[i]-vals_elo_sva[i]+vals_slope_sva[i]*np.log10(1+zvals)),'r--')

        if len(vals_sva) > 0:
            pyplot.plot(vals_sva[1:,0],vals_sva[1:,1+3*i]/fac[i],'ro')
            pyplot.errorbar(vals_sva[1:,0],vals_sva[1:,1+3*i]/fac[i],yerr=[vals_sva[1:,3+3*i]/fac[i],vals_sva[1:,2+3*i]/fac[i]],fmt=None,ecolor='r')

        pyplot.ylabel(labels[i])
        pyplot.xlabel('z')

    #pyplot.tight_layout()
    pyplot.margins(0,0)
    pyplot.savefig(outdir+"pev_all.ps",orientation='landscape')
    pyplot.clf()

    return


#Plotting the centrals part of the evolution test
def plot_ev_test_cen(vals_data,outdir,
                     vals_data_dr8=[]):
    #labeling
    labels = [r'$\sigma_L$','r',r'log $L_{c0}$',r'$A_L$',r'$s_c$']    
    nparams = len(labels)

    #S82 overall fit
    #chain_s82 = read_cen_chain("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10/chain_cen_all.dat")
    chain_s82 = read_cen_chain("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10_v2/chain_cen_all.dat")
    vals_truth, vals_ehi, vals_elo = mcmc.get_errors(chain_s82[8000:,:])
    vals_slope = np.zeros(5)
    vals_slope[2] = vals_truth[4]
    vals_truth = vals_truth[[0,1,2,3,5]]
    vals_ehi = vals_ehi[[0,1,2,3,5]]
    vals_elo = vals_elo[[0,1,2,3,5]]

    #DR8 overall fit
    chain_dr8 = read_cen_chain("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10_cencorr/chain_cen_all.dat")
    vals_truth_dr8, vals_ehi_dr8, vals_elo_dr8 = mcmc.get_errors(chain_dr8[8000:,:])
    vals_slope_dr8 = np.zeros(5)
    vals_slope_dr8[2] = vals_truth_dr8[4]
    vals_truth_dr8 = vals_truth_dr8[[0,1,2,3,5]]
    vals_ehi_dr8 = vals_ehi_dr8[[0,1,2,3,5]]
    vals_elo_dr8 = vals_elo_dr8[[0,1,2,3,5]]

    zvals = np.array(range(500))*0.001+0.1
    fac = [np.log(10.), 1., np.log(10.), 1., 1., 1.]
    pyplot.figure(1,[11.,8.5])
    pyplot.subplots_adjust(wspace = 0.35)
    for i in range(len(labels)):
        print i, vals_truth[i], vals_slope[i]
        pyplot.subplot(2,3,i+1)
        #DR8 part
        pyplot.plot(zvals,(vals_truth_dr8[i]+vals_slope_dr8[i]*np.log10(1+zvals)),'k')
        pyplot.plot(zvals,(vals_truth_dr8[i]+vals_ehi_dr8[i]+vals_slope_dr8[i]*np.log10(1+zvals)),'k--')
        pyplot.plot(zvals,(vals_truth_dr8[i]-vals_elo_dr8[i]+vals_slope_dr8[i]*np.log10(1+zvals)),'k--')

        if len(vals_data_dr8) > 0:
            pyplot.plot(vals_data_dr8[:,0],vals_data_dr8[:,1+3*i]/fac[i],'ko')
            pyplot.errorbar(vals_data_dr8[:,0],vals_data_dr8[:,1+3*i]/fac[i],yerr=[vals_data_dr8[:,3+3*i]/fac[i],vals_data_dr8[:,2+3*i]/fac[i]],fmt=None,ecolor='k')

        #S82 parts
        pyplot.plot(zvals,(vals_truth[i]+vals_slope[i]*np.log10(1+zvals)),'b')
        pyplot.plot(zvals,(vals_truth[i]+vals_ehi[i]+vals_slope[i]*np.log10(1+zvals)),'b--')
        pyplot.plot(zvals,(vals_truth[i]-vals_elo[i]+vals_slope[i]*np.log10(1+zvals)),'b--')

        pyplot.plot(vals_data[1:,0],vals_data[1:,1+3*i]/fac[i],'bo')

        pyplot.errorbar(vals_data[1:,0],vals_data[1:,1+3*i]/fac[i],yerr=[vals_data[1:,3+3*i]/fac[i],vals_data[1:,2+3*i]/fac[i]],fmt=None,ecolor='b')
        pyplot.ylabel(labels[i])
        pyplot.xlabel('z')

    #pyplot.tight_layout()
    pyplot.margins(0,0)
    pyplot.savefig(outdir+"pev_cen_all.ps",orientation='landscape')
    pyplot.clf()

    return

def read_cen_chain(filename):
    chain = np.loadtxt(filename)
    chain[:,0] = chain[:,0]/np.log(10)
    chain[:,2] = chain[:,2]/np.log(10)

    return chain

def read_cen_chain_prior(filename):
    chain = np.loadtxt(filename)
    chain = chain[:,3:]
    chain[:,0] = chain[:,0]/np.log(10)
    chain[:,2] = chain[:,2]/np.log(10)

    return chain

#Plotting the centrals part of the evolution test
def plot_ev_test_cen_sva(vals_data,outdir,
                         vals_data_dr8=[],vals_sva=[]):
    #labeling
    labels = [r'$\sigma_L$','r',r'log $L_{c0}$',r'$A_L$',r'$s_c$']    
    nparams = len(labels)

    #S82 overall fit
    #chain_s82 = read_cen_chain("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10/chain_cen_all.dat")
    chain_s82 = read_cen_chain("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10_v2/chain_cen_all.dat")
    vals_truth, vals_ehi, vals_elo = mcmc.get_errors(chain_s82[8000:,:])
    vals_slope = np.zeros(5)
    vals_slope[2] = vals_truth[4]
    vals_truth = vals_truth[[0,1,2,3,5]]
    vals_ehi = vals_ehi[[0,1,2,3,5]]
    vals_elo = vals_elo[[0,1,2,3,5]]

    #DR8 overall fit
    chain_dr8 = read_cen_chain("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10_cencorr/chain_cen_all.dat")
    vals_truth_dr8, vals_ehi_dr8, vals_elo_dr8 = mcmc.get_errors(chain_dr8[8000:,:])
    vals_slope_dr8 = np.zeros(5)
    vals_slope_dr8[2] = vals_truth_dr8[4]
    vals_truth_dr8 = vals_truth_dr8[[0,1,2,3,5]]
    vals_ehi_dr8 = vals_ehi_dr8[[0,1,2,3,5]]
    vals_elo_dr8 = vals_elo_dr8[[0,1,2,3,5]]

    #SVA overall fit
    chain_sv = read_cen_chain("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_sva1_v6.1.3/chain_cen_all.dat")
    vals_truth_sva, vals_ehi_sva, vals_elo_sva = mcmc.get_errors(chain_sv[8000:,:])
    vals_slope_sva = np.zeros(5)
    vals_slope_sva[2] = vals_truth_sva[4]
    vals_truth_sva = vals_truth_sva[[0,1,2,3,5]]
    vals_ehi_sva = vals_ehi_sva[[0,1,2,3,5]]
    vals_elo_sva = vals_elo_sva[[0,1,2,3,5]]

    zvals = np.array(range(800))*0.001+0.1
    fac = [np.log(10.), 1., np.log(10.), 1., 1., 1.]
    pyplot.figure(1,[11.,8.5])
    pyplot.subplots_adjust(wspace = 0.35)
    for i in range(len(labels)):
        print i, vals_truth[i], vals_slope[i]
        pyplot.subplot(2,3,i+1)
        #DR8 part
        pyplot.plot(zvals,(vals_truth_dr8[i]+vals_slope_dr8[i]*np.log10(1+zvals)),'k')
        pyplot.plot(zvals,(vals_truth_dr8[i]+vals_ehi_dr8[i]+vals_slope_dr8[i]*np.log10(1+zvals)),'k--')
        pyplot.plot(zvals,(vals_truth_dr8[i]-vals_elo_dr8[i]+vals_slope_dr8[i]*np.log10(1+zvals)),'k--')

        if len(vals_data_dr8) > 0:
            pyplot.plot(vals_data_dr8[:,0],vals_data_dr8[:,1+3*i]/fac[i],'ko')
            pyplot.errorbar(vals_data_dr8[:,0],vals_data_dr8[:,1+3*i]/fac[i],yerr=[vals_data_dr8[:,3+3*i]/fac[i],vals_data_dr8[:,2+3*i]/fac[i]],fmt=None,ecolor='k')

        #S82 parts
        pyplot.plot(zvals,(vals_truth[i]+vals_slope[i]*np.log10(1+zvals)),'b')
        pyplot.plot(zvals,(vals_truth[i]+vals_ehi[i]+vals_slope[i]*np.log10(1+zvals)),'b--')
        pyplot.plot(zvals,(vals_truth[i]-vals_elo[i]+vals_slope[i]*np.log10(1+zvals)),'b--')

        pyplot.plot(vals_data[1:,0],vals_data[1:,1+3*i]/fac[i],'bo')

        pyplot.errorbar(vals_data[1:,0],vals_data[1:,1+3*i]/fac[i],yerr=[vals_data[1:,3+3*i]/fac[i],vals_data[1:,2+3*i]/fac[i]],fmt=None,ecolor='b')

        #SVA part
        pyplot.plot(zvals,(vals_truth_sva[i]+vals_slope_sva[i]*np.log10(1+zvals)),'r')
        pyplot.plot(zvals,(vals_truth_sva[i]+vals_ehi_sva[i]+vals_slope_sva[i]*np.log10(1+zvals)),'r--')
        pyplot.plot(zvals,(vals_truth_sva[i]-vals_elo_sva[i]+vals_slope_sva[i]*np.log10(1+zvals)),'r--')

        if len(vals_sva)>0:
            pyplot.plot(vals_sva[1:,0],vals_sva[1:,1+3*i]/fac[i],'ro')
            pyplot.errorbar(vals_sva[1:,0],vals_sva[1:,1+3*i]/fac[i],yerr=[vals_sva[1:,3+3*i]/fac[i],vals_sva[1:,2+3*i]/fac[i]],fmt=None,ecolor='r')
        

        pyplot.ylabel(labels[i])
        pyplot.xlabel('z')

    #pyplot.tight_layout()
    pyplot.margins(0,0)
    pyplot.savefig(outdir+"pev_cen_all.ps",orientation='landscape')
    pyplot.clf()

    return

#Plot of Lcen mass dependence vs Hansen 2009, other literature references
def plot_Lcen_mass(outdir):
    h = 0.6704
    Mpiv = 2.35e14
    zval = 0.25

    #Mass bins
    mbins = np.array(range(201))*0.01+13

    #Hansen values, converted to virial mass
    Mpiv_hansen = 1.66e14/h

    #Getting the central luminosities
    Lcen_hansen = 6e10*(10.**mbins/Mpiv_hansen)**0.3
    Lcen_h_hi = 8e10*(10.**mbins/Mpiv_hansen)**0.3
    Lcen_h_lo = 4e10*(10.**mbins/Mpiv_hansen)**0.3

    #Getting DR8 central luminosities at z=0.25
    fits_dir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10/"
    chain = np.loadtxt(fits_dir + "chain_cen_all.dat")
    cen_param_dr8 = chain[np.argmax(chain[:,-1])]

    #Getting S82 central luminosities at z=0.25
    fits_dir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10/"
    chain = np.loadtxt(fits_dir + "chain_cen_all.dat")
    cen_param_s82 = chain[np.argmax(chain[:,-1])]

    pyplot.figure(1,[11.,8.5])

    pyplot.plot(mbins,np.log10(Lcen_hansen),'k',label='Hansen 09')
    pyplot.plot(mbins,np.log10(Lcen_h_hi),'k--',label='_nolegend_')
    pyplot.plot(mbins,np.log10(Lcen_h_lo),'k--',label='_nolegend_')

    pyplot.plot(mbins,(cen_param_dr8[2]+cen_param_dr8[3]*(mbins*np.log(10.)-np.log(Mpiv)) + cen_param_dr8[4]*np.log(1+zval))/np.log(10.),'b-' ,label='DR8')
    pyplot.plot(mbins,(cen_param_s82[2]+cen_param_s82[3]*(mbins*np.log(10.)-np.log(Mpiv)) + cen_param_s82[4]*np.log(1+zval))/np.log(10.),'r-' ,label='S82')

    pyplot.xlabel(r'$log(M_{vir})$ $[M_\odot]$')
    pyplot.ylabel(r'$log(L_{cen})$ $[L_\odot/h^2]$')

    pyplot.legend(loc='upper left')

    pyplot.savefig(outdir+"Lcen_lit_comp.ps",orientation='landscape')
    pyplot.clf()

    return
