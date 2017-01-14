#/usr/bin/python

#Final plotting calls for contours

import numpy as np
import plot_test
import matplotlib.pyplot as pyplot

def plot_final_contour_cen(outdir):
    indir_dr8 = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10_v3/"
    indir_s82 = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10_v3/"

    dat_cen_dr8 = np.loadtxt(indir_dr8 + "chain_cen_z_0.3_0.33.dat")
    dat_cen_s82 = np.loadtxt(indir_s82 + "chain_cen_z_0.3_0.4.dat")
    
    cen_syserr = [0., 0., 0.0775, 0.08602]

    labels_cen = [r'$\sigma_L$','r',r'log $L_0$',r'$A_L$']

    #Creating the systematics contours
    dat_cen_s82_sys = np.copy(dat_cen_s82)
    for i in range(len(cen_syserr)):
        if cen_syserr[i] == 0:
            continue
        dat_cen_s82_sys[:,i] = dat_cen_s82[:,i] + np.random.normal(0., cen_syserr[i], len(dat_cen_s82) )

    #Fix things to log 10 base
    dat_cen_dr8[:,0] = dat_cen_dr8[:,0]/np.log(10.)
    dat_cen_dr8[:,2] = dat_cen_dr8[:,2]/np.log(10.)
    dat_cen_s82[:,0] = dat_cen_s82[:,0]/np.log(10.)
    dat_cen_s82[:,2] = dat_cen_s82[:,2]/np.log(10.)
    dat_cen_s82_sys[:,0] = dat_cen_s82_sys[:,0]/np.log(10.)
    dat_cen_s82_sys[:,2] = dat_cen_s82_sys[:,2]/np.log(10.)

    #Make the plot
    plot_test.plot_many_contours(np.array([dat_cen_dr8[:,:5],dat_cen_s82[:,:5],dat_cen_s82_sys[:,:5]]),5000,labels_cen)


    #Save the plot
    pyplot.savefig(outdir+"contour_cen_sys.ps",orientation='landscape')

    return

def plot_final_contour_sat(outdir):
    indir_dr8 = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10_v3/"
    indir_s82 = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10_v3/"

    dat_sat_dr8 = np.loadtxt(indir_dr8 + "chain_sat_z_0.3_0.33.dat")
    dat_sat_s82 = np.loadtxt(indir_s82 + "chain_sat_z_0.3_0.4.dat")

    sat_syserr = [0.0326, 0.09431, 0.04446, 0.04247, 0.1235]

    #Creating the systematics contours
    dat_sat_s82_sys = np.copy(dat_sat_s82)
    for i in range(len(sat_syserr)):
        if sat_syserr[i] == 0:
            continue
        dat_sat_s82_sys[:,i] = dat_sat_s82[:,i] + np.random.normal(0., sat_syserr[i], len(dat_sat_s82) )

    dat_sat_dr8[:,2] = dat_sat_dr8[:,2]/np.log(10.)
    dat_sat_s82[:,2] = dat_sat_s82[:,2]/np.log(10.)
    dat_sat_s82_sys[:,2] = dat_sat_s82_sys[:,2]/np.log(10.)

    labels_sat = [r'ln $\phi_0$',r'$A_\phi$',r'log $L_{s0}$',r'$A_s$',r'$\alpha$']

    plot_test.plot_many_contours(np.array([dat_sat_dr8[:,:6],dat_sat_s82[:,:6],dat_sat_s82_sys[:,:6]]),10000,labels_sat)  
    
    #Save the plot
    pyplot.savefig(outdir+"contour_sat_sys.ps",orientation='landscape')
    pyplot.clf()

    return
