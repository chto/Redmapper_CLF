#!/usr/bin/python

import sys

import numpy as np
import pyfits
import esutil.htm as htm
import matplotlib.pyplot as plt
import matplotlib

import cosmo

def plot_match_results(z, dr8_imag, s82_imag):
    print "Okay"
    plt.plot(z, s82_imag - dr8_imag, 'bo', ms=4)
    plt.plot([0, 0.7], [0, 0], 'r--', linewidth=3)
    plt.ylim([-0.65, 0.65])
    plt.xlim([0.05, 0.6])
    
    plt.xlabel('z', fontsize=18)
    plt.ylabel('imag(s82) - imag(DR8)', fontsize=18)

    return

if __name__ == "__main__":
    #Setting up some plot label size increases
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)

    print "Getting the data"
    s82 = pyfits.open("/nfs/slac/g/ki/ki19/des/erykoff/clusters/process/stripe82_redmapper_v5.10/run_ubermem/stripe82_run_redmapper_v5.10_lgt5_catalog.fit")
    dr8 = pyfits.open("/nfs/slac/g/ki/ki19/des/erykoff/clusters/process/dr8_redmapper_v5.10/run_ubermem/dr8_run_redmapper_v5.10_lgt5_catalog.fit")
    s82_mem = pyfits.open("/nfs/slac/g/ki/ki19/des/erykoff/clusters/process/stripe82_redmapper_v5.10/run_ubermem/stripe82_run_redmapper_v5.10_lgt5_catalog_members.fit")    
    s82_mem = s82_mem[1].data

    s82 = s82[1].data
    dr8 = dr8[1].data

    # Since comparing surveys, cutoff in S82 to make matching easier
    #s82 = s82[np.where(s82['lambda_chisq'] > 10)[0]]
    dr8 = dr8[np.where(dr8['lambda_chisq'] > 10)[0]]

    # Matching central positions
    h = htm.HTM(12)
    m1, m2, dist = h.match(dr8['ra'], dr8['dec'],
                           s82_mem['ra'], s82_mem['dec'], 2.0/3600)

    outdir = "/afs/slac.stanford.edu/u/ki/rmredd/data/redm_plots_v4/"

    # Previously measured stuff
    matchdir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/test_galaxies_cenmatch/"
    cen1 = np.loadtxt(matchdir+"galmatch_delta_cen1.dat")

    plot_match_results(dr8[m1]['z_lambda'], dr8[m1]['imag'], s82_mem['imag'][m2])
    plt.plot(cen1[:,0], cen1[:,1], 'go-', linewidth=3)
    plt.plot(cen1[:,0], cen1[:,1] + cen1[:,2], 'g--', linewidth=3)
    plt.plot(cen1[:,0], cen1[:,1] - cen1[:,2], 'g--', linewidth=3)

    plt.savefig(outdir+'galmatch_delta_cen1.pdf')
    plt.clf()

    m1, m2, dist = h.match(dr8['ra_cent'][:, 1], dr8['dec_cent'][:,1], 
                           s82_mem['ra'], s82_mem['dec'], 2.0/3600)

    cen2 = np.loadtxt(matchdir+"galmatch_delta_cen2.dat")

    # Working on the second set
    print "Working on the second set"
    dr8_mem = pyfits.open("/nfs/slac/g/ki/ki19/des/erykoff/clusters/process/dr8_redmapper_v5.10/run_ubermem/dr8_run_redmapper_v5.10_lgt5_catalog_members.fit")
    dr8_mem = dr8_mem[1].data

    dr8_imag = np.zeros(len(m1))
    print "looping"
    print len(m1)
    for i in range(len(m1)):
        place = np.where( (dr8_mem['id'] == dr8['id_cent'][m1[i], 1]) & 
                          (dr8_mem['mem_match_id'] == dr8['mem_match_id'][m1[i]]) )[0]
        if i % 100 == 0:
            print i
        dr8_imag[i] = dr8_mem['imag'][place]
    print "done looping"
                                        
    plot_match_results(dr8[m1]['z_lambda'], dr8_imag, s82_mem['imag'][m2, 1])
    plt.plot(cen1[:,0], cen1[:,1], 'mo-', linewidth=2)
    plt.plot(cen2[:,0], cen2[:,1], 'go-', linewidth=3)
    plt.plot(cen2[:,0], cen2[:,1] + cen2[:,2], 'g--', linewidth=3)
    plt.plot(cen2[:,0], cen2[:,1] - cen2[:,2], 'g--', linewidth=3)

    plt.show()
