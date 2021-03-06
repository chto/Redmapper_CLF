#/usr/bin/python

#This has a single "plot summary" function, which runs most of the plots 
#for the DR8/S82 CLF paper
#It calls lots of other things, and I've left a few notes about what 
#each set of functions makes

import numpy as np
import matplotlib
#import matplotlib.rc
import matplotlib.pyplot as pyplot

import plot_nz
import plot_test
import plot_ev_test
import get_p_bcg_cen

def run_plot_summary(outdir):
    #All the directories with input data
    #CLFs
    indir_dr8 = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/dr8_zlambda_v5.10/"
    indir_s82 = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/s82_v5.10_uber/"
    
    #MCMC and fit data
    indir_dr8_fit = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10_v3/"
    indir_s82_fit = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10_v3/"
    
    #Increase tick label sizes
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)

    #Figure 1
    #n(z) plot
    nz_dr8 = np.loadtxt(indir_dr8+'nz_lm_20_200.dat')
    nz_s82 = np.loadtxt(indir_s82+'nz_lm_20_200.dat')
    #Makes appropriate conversions for volume, and outputs values
    z_dr8, val_dr8 = plot_nz.nz_plot_conversion(nz_dr8)
    z_s82, val_s82 = plot_nz.nz_plot_conversion(nz_s82)

    #Upper limit on data 
    zcut = np.where(z_dr8 <= 0.33)[0]

    #Clear, just in case
    #Let's plot things
    pyplot.plot(z_dr8[zcut[:-1]], val_dr8[zcut[:-1]]*1e6,'k-',linewidth=2.0)
    pyplot.plot(z_s82, val_s82*1e6,'b--',linewidth=2.0)
    ztmp = [0.33, 0.33]
    match = [0, 20]
    pyplot.plot(ztmp, match,'k', linestyle='dotted', linewidth=2.0)

    pyplot.xlim([0.1,0.6])
    pyplot.ylim([6.1,15])
    pyplot.xlabel('z',fontsize=18)
    pyplot.ylabel('n(z) '+r'$[10^{-6}(Mpc/h)^{-3}]$',fontsize=18)
    pyplot.savefig(outdir+'nz_comp_both.pdf',orientation='landscape')
    pyplot.clf()
    
    #Restore tick labels, etc., to defaults
    matplotlib.rcdefaults()

    #Making the contour plots
    print "Making contour plots..."
    chain_dr8_cen = np.loadtxt(indir_dr8_fit + "chain_cen_z_0.25_0.3.dat")
    chain_s82_cen = np.loadtxt(indir_s82_fit + "chain_cen_z_0.2_0.3.dat")

    chain_dr8_sat = np.loadtxt(indir_dr8_fit + "chain_sat_z_0.25_0.3.dat")
    chain_s82_sat = np.loadtxt(indir_s82_fit + "chain_sat_z_0.2_0.3.dat")

    #Centrals
    censys = [0., 0., 0.0775, 0.08602]
    chain_s82_cen_sys = np.copy(chain_s82_cen)
    for i in range(len(censys)):
        if censys[i] == 0:
            continue
        chain_s82_cen_sys[:,i] += np.random.normal(0.,censys[i],len(chain_s82_cen))
        
    #Fix ln->log base 10
    chain_dr8_cen[:,0] /= np.log(10.)
    chain_s82_cen[:,0] /= np.log(10.)
    chain_s82_cen_sys[:,0] /= np.log(10.)
    chain_dr8_cen[:,2] /= np.log(10.)
    chain_s82_cen[:,2] /= np.log(10.)
    chain_s82_cen_sys[:,2] /= np.log(10.)

    labels_cen = [r'$\sigma_L$','r',r'log $L_0$',r'$A_L$']
    plot_test.plot_many_contours(np.array([chain_dr8_cen, chain_s82_cen, chain_s82_cen_sys]),5000,labels_cen, rotation=45)
    pyplot.savefig(outdir+'contour_cen_sys.pdf',orientation='landscape')
    pyplot.clf()

    #Satellites
    satsys = [0.0326, 0.0943, 0.04446, 0.04272, 0.1235]
    chain_s82_sat_sys = np.copy(chain_s82_sat)
    for i in range(len(satsys)):
        chain_s82_sat_sys[:,i] += np.random.normal(0.,satsys[i],len(chain_s82_sat))

    #Fix ln->log base 10
    chain_dr8_sat[:,2] /= np.log(10.)
    chain_s82_sat[:,2] /= np.log(10.)
    chain_s82_sat_sys[:,2] /= np.log(10.)

    labels_sat = [r'ln $\phi_0$',r'$A_\phi$',r'log $L_{s0}$',r'$A_s$',r'$\alpha$']
    plot_test.plot_many_contours(np.array([chain_dr8_sat, chain_s82_sat, chain_s82_sat_sys]),10000,labels_sat)
    pyplot.savefig(outdir+'contour_sat_sys.pdf',orientation='landscape')
    pyplot.clf()

    #Lcen/L* evolution -- plot is currently fig. 11 in paper
    #Includes comparison w/ Eli's plots
    #Requires fit parameters as inputs -- full, including evolution
    #Increase tick label sizes
    print "Doing Lcen/L* evolution..."
    matplotlib.rc('xtick', labelsize=18)
    matplotlib.rc('ytick', labelsize=18)

    #Note all parameters here are in ln form, not log_10
    #Centrals parameters from fit -- may be used in later functions as well
    #sigma_c r ln Lc0 A_L B_L s_c
    param_cen = np.array([0.3641, -0.5502, 25.05117, 0.3768, 0.7714, 1.188])
    param_cen_err = np.array([[ 0.01501128,  0.01193099],
                              [ 0.14726338,  0.10903354],
                              [0.03330154 , 0.03318014 ],
                              [0.00747015 , 0.00726248],
                              [0.34234353 , 0.360938],
                              [ 0.08597129 , 0.08362242]])
    #Satellite parameters from fit -- may be used in later functions as well
    #ln phi B_phi A_0,phi A_z,phi ln Ls0 B_s A_0,s A_z,s alpha_0 B_alpha s_s
    param_sat = np.array([3.9356, 0.4823, 0.8640, 0.05134, 23.5356, 1.6501, 0.0594, 0.2386, -0.7866, 0.6647, 1.428])
    param_sat_err = np.array([[ 0.00954383 , 0.00965202],
                              [ 0.12868261 , 0.14049748],
                              [ 0.00554557,  0.00534632],
                              [ 0.07390285 , 0.06918056],
                              [ 0.00969516 , 0.01034964],
                              [ 0.1336759  , 0.13766365],
                              [ 0.00476324 , 0.00552397],
                              [ 0.06816756 , 0.07311754], 
                              [ 0.00849665 , 0.01049897],
                              [ 0.13661951 , 0.13428535],
                              [ 0.07978035 , 0.07872706]])

    #Despite the name, this produces multiple plots -- includes Fig. 8 and 10 from paper
    # Which give the comparison of the z-evolution of all parameters
    #matplotlib.rcdefaults()
    #matplotlib.rc('xtick', labelsize=12)
    #matplotlib.rc('ytick', labelsize=12)
    #plot_test.plot_dr8(outdir)

    #Figure 11 -- Lst evolution in the model
    matplotlib.rc('xtick', labelsize=18)
    matplotlib.rc('ytick', labelsize=18)
    plot_test.Lst_ev(param_cen, param_sat,outdir+'lst_evolution.pdf',err_cen=param_cen_err,err_sat=param_sat_err)
    pyplot.clf()

    #Figure 12 -- comparison of Lcen, L* with mass dependence
    zmid = [0.1, 0.2, 0.3, 0.4, 0.5]
    plot_test.param_plot_set(param_cen, param_sat, zmid, outdir+"param_test_gen.pdf")

    #Figure 13 -- probability of brightest sat brighter than cen plot
    print "Getting P(BNC)..."

    get_p_bcg_cen.plot_bcg_dr8(param_cen, param_sat,param_cen_err,param_sat_err,do_errors=False)
    pyplot.tight_layout()
    pyplot.savefig(outdir+"pbcg_dr8.pdf",orientation='landscape')
    pyplot.clf()

    #Figure 15 -- halo mass dependence, comparison to e.g., Hansen et al
    plot_test.plot_Lcen_mass(param_cen, param_cen_err, outdir)

    return
