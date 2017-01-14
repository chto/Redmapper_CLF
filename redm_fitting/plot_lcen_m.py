#/usr/bin/python

import numpy as np
import matplotlib.pyplot as pyplot

import fit_plm

#Functions for plotting an Lcen(M) comparison

def plot_lcen_m(outdir):
    h = 0.6704
    Mpiv = 2.35e14

    #DR8 fit
    #Current mass-lambda parameters
    lm_param = [Mpiv, 0.843, 0.750, 2.9458]
    fits_dir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10/"
    #Pick up the latest fit for the centrals
    chain = np.loadtxt( fits_dir + "chain_cen_all.dat" )
    cen_param = chain[np.argmax(chain[:,-1])]
    
    #S82 fit
    fits_dir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10/"
    #Pick up the latest fit for the centrals
    chain = np.loadtxt( fits_dir + "chain_cen_all.dat" )
    cen_param_s82 = chain[np.argmax(chain[:,-1])]


    #Mass range we care about for now
    mvals = np.array(range(150))*0.02+13

    #plotting for z~0.1
    pyplot.plot(mvals,(cen_param[2]+cen_param[3]*np.log(10.**mvals/Mpiv) + cen_param[4]*np.log(1.1))/np.log(10),'b')
    pyplot.plot(mvals,(cen_param_s82[2]+cen_param_s82[3]*np.log(10.**mvals/Mpiv) + cen_param_s82[4]*np.log(1.1))/np.log(10),'r')

    #Reading in the Bernardi result
    ber = np.loadtxt("/nfs/slac/g/ki/ki10/rmredd/BolshoiCheck_SHAMtest/Lcen_M_bernardi.dat")
    pyplot.plot(ber[:,0]-np.log10(0.7),ber[:,1],'mo')
    pyplot.errorbar(ber[:,0]-np.log10(0.7),ber[:,1],ber[:,2],fmt=None,ecolor='m')
    
    #Reddick 13 results
    minmass = np.array([12., 12.3, 12.6, 12.9, 13.2, 13.8])
    maxmass = np.array([12.3, 12.6, 12.9, 13.2, 13.8, 14.5])
    Lcen = [10.024, 10.150, 10.238, 10.284, 10.332, 10.381]
    Lcenerr = [0.001,0.002,0.003,0.004,0.004,0.009]
    meanmass = (minmass+maxmass)/2.
    pyplot.plot(meanmass-np.log10(.7),Lcen,'ko')
    pyplot.errorbar(meanmass-np.log10(.7),Lcen,Lcenerr,ecolor='k',fmt=None)

    pyplot.xlim([12,15.5])
    pyplot.xlabel('Mvir')
    pyplot.ylabel('Lcen')
    
    return
