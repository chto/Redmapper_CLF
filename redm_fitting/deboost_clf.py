#/usr/bin/python

import numpy as np
import pyfits
from glob import glob

import fit_plm
import fit_psat
from mag_convert import mag_to_Lsolar

#Various functions that are needed for calculating the deboost correction
#to the redmapper galaxy luminosities

#Estimate the i-band magnitude errors as a function of luminosity
#Note this is done separately for centrals and satellites, as a function
#of redshift
def get_iband_magerr(imagerr_cen,cenmag,cenz,imagerr_sat,mag,z,zmin,zmax):
    nz = len(zmin)

    dL = 0.05
    lumbins = 9 + np.array(range(60))*dL + dL/2.

    cen_err = np.zeros([nz,len(lumbins)])
    sat_err = np.copy(cen_err)
    for i in range(nz):
        for j in range(len(lumbins)):
            #Centrals first
            clist = np.where( (cenmag > lumbins[j]-dL/2.) & (cenmag < lumbins[j]+dL/2.) & (cenz > zmin[i]) & (cenz < zmax[i]) )[0]
            if len(clist) > 0:
                cen_err[i,j] = np.median( imagerr_cen[clist] )

            #Satellites
            slist = np.where( (mag > lumbins[j]-dL/2.) & (mag < lumbins[j]+dL/2.) & (z > zmin[i]) & (z < zmax[i]) )[0]
            if len(slist) > 0:
                sat_err[i,j] = np.median( imagerr_sat[slist] )


    #Magnitude to luminosity conversion
    cen_err = cen_err/2.5
    sat_err = sat_err/2.5
            
    return [lumbins, cen_err, sat_err]

#Deboosting for a central galaxy distribution
def deboost_central(lumbins, cen_err, cen_param, lm_param, mass_param, z, lm):
    '''
    Inputs: luminosity bins, luminosity errors, central fitting parameters, 
            lambda-mass relation parameters, mass function parameters, 
            redshifts to evaluate, richnesses to evaluate
    '''
    nz = len(z)
    nlm = len(lm)
    nlumbins = len(lumbins)
    dL = lumbins[1]-lumbins[0]

    deboost = np.zeros([nz,nlm,nlumbins])
    my_param = np.zeros(7)
    my_param[0] = lm_param[4] #sigma_lm
    my_param[1] = cen_param[0] #sigma_L
    my_param[2] = cen_param[1] #r
    my_param[4] = lm_param[1] #A_lm
    my_param[5] = cen_param[2]
    my_param[6] = cen_param[3]
    Mpiv = lm_param[0]

    for i in range(nz):
        #lnlm0 + B_lm*np.log(1+z)
        my_param[3] = lm_param[3] + lm_param[2]*np.log(1+z[i]) 
        for j in range(nlm):
            for k in range(nlumbins):
                #Get the slope at this point
                p_hi = fit_plm.p_L(10.**(lumbins[k]+dL/2.),lm[j],
                                   mass_param[i][0],mass_param[i][1],
                                   mass_param[i][2],mass_param[i][2],
                                   Mpiv,my_param,z[i],cen_param[4])
                p_lo = fit_plm.p_L(10.**(lumbins[k]-dL/2.),lm[j],
                                   mass_param[i][0],mass_param[i][1],
                                   mass_param[i][2],mass_param[i][2],
                                   Mpiv,my_param,z[i],cen_param[4])
                beta = (np.log(p_hi)-np.log(p_lo))/dL
                deboost[i,j,k] = beta*cen_err[i,k]**2

    return deboost

#Perform the related calculation for the satellites
def deboost_satellite(lumbins, sat_err, sat_param, lm_param, mass_param, z, lm_min,lm_max,mval,nval):
    nz = len(z)
    nlm = len(lm_min)
    nlumbins = len(lumbins)
    dL = lumbins[1]-lumbins[0]

    deboost = np.zeros([nz,nlm,nlumbins])

    for i in range(nz):
        for j in range(nlm):
            p_hi = fit_psat.func_sat_convolved_phi_ev(lumbins+dL/2.,
                                                      lm_min[j],lm_max[j],
                                                      lm_param[0:4],lm_param[4],
                                                      sat_param,mass_param,
                                                      z[i],mval[i],nval[i])
            p_lo = fit_psat.func_sat_convolved_phi_ev(lumbins-dL/2.,
                                                      lm_min[j],lm_max[j],
                                                      lm_param[0:4],lm_param[4],
                                                      sat_param,mass_param,
                                                      z[i],mval[i],nval[i])
            beta = (np.log(p_hi)-np.log(p_lo))/dL
            deboost[i,j] = beta*sat_err[i]**2

    return deboost

#Wrapper for doing the deboosting, centrals and satellites, for S82
def deboost_s82():
    '''
    Note this requires the kcorrections on the redmapper catalogs
    '''
    h = 0.6704
    Mpiv = 2.35e14

    #Read in the input files
    c = pyfits.open("/nfs/slac/g/ki/ki19/des/erykoff/clusters/process/stripe82_redmapper_v5.10/run_ubermem/stripe82_run_redmapper_v5.10_lgt5_catalog.fit")
    c = c[1].data
    g = pyfits.open("/nfs/slac/g/ki/ki19/des/erykoff/clusters/process/stripe82_redmapper_v5.10/run_ubermem/stripe82_run_redmapper_v5.10_lgt5_catalog_members.fit")
    g = g[1].data
    #kcorrections
    kcorr = pyfits.open("/u/ki/rmredd/ki10/redmapper/s82_v5.10_uber_zlambda_0.3.fit")
    kcorr = kcorr[0].data
    #cindex
    cindex = pyfits.open("/u/ki/rmredd/ki10/redmapper/cindex_s82_v5.10_uber_zlambda_0.3.fit")
    cindex = cindex[0].data
    print "Done with catalog readin"

    #Getting luminosities
    mag = g['imag'] + (kcorr[:,2]-g['model_mag'][:,2])
    abs_solar = 4.71493
    #Convert to luminosity and get the centrals
    mag = np.log10(mag_to_Lsolar(mag,use_des=0,abs_solar=abs_solar))
    cenmag = np.zeros_like(cindex).astype(float)
    for i in range(len(cenmag)):
        cenmag[i] = mag[cindex[i]]

    print "Done fixing luminosities"

    #Mass-lambda parameters
    lm_param = [2.35e14, 0.857, 0.1547, 2.7226, 0.1842]

    #Read in and handle the halo mass function
    mf_files = glob("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/mass_functions_planck/mf_planck*.dat")
    mf_files = np.array(mf_files)
    mf_files.sort()
    #print mf_files
    mf_all = []
    mf_files = mf_files[[2,6,12,14,16]]
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

    #Redshift values
    zmin = 0.1+np.array(range(5))*0.1
    zmax = zmin+0.1
    zmid = zmin+0.05
    

    #lambda values and ranges
    lmval = np.array(range(19))*5 + 5
    lm_min = [10., 20., 40.]
    lm_max = [20., 100., 100.]

    #Pick up the photometric errors
    lumbins, cen_err, sat_err = get_iband_magerr(c['imag_err'],cenmag[:,0],c['z_lambda'],g['imag_err'],mag,g['z'],zmin,zmax)
    print "Done getting photometric errors"

    #TESTING PRINTOUT
    #return lumbins, cen_err, sat_err

    #Central parameters
    cen_param = [0.37, -0.9, 24.662, 0.3, 1.27]
    #Satellite parameters
    sat_param = [3.832, 0.845, 23.161, 0.063, 1.35, -0.851, 0.989]

    #Get the central deboost
    deboost_cen = deboost_central(lumbins, cen_err, cen_param, lm_param, mass_param, zmid, lmval)

    #Get the satellite deboost
    deboost_sat = deboost_satellite(lumbins, sat_err, sat_param, lm_param, mass_param, zmid, lm_min, lm_max, mf_all[:,0],mf_all[:,1])
                  
    return [cen_err, sat_err, zmid, lmval, lumbins, deboost_cen, lm_min, lm_max, deboost_sat]

#Wrapper for doing the deboosting, centrals and satellites, for DR8
def deboost_dr8():
    '''
    Note this requires the kcorrections on the redmapper catalogs
    '''
    h = 0.6704
    Mpiv = 2.35e14

    #Read in the input files
    c = pyfits.open("/nfs/slac/g/ki/ki19/des/erykoff/clusters/process/dr8_redmapper_v5.10/run_ubermem/dr8_run_redmapper_v5.10_lgt5_catalog.fit")
    c = c[1].data
    g = pyfits.open("/nfs/slac/g/ki/ki19/des/erykoff/clusters/process/dr8_redmapper_v5.10/run_ubermem/dr8_run_redmapper_v5.10_lgt5_catalog_members.fit")
    g = g[1].data
    #kcorrections
    kcorr = pyfits.open("/u/ki/rmredd/ki10/redmapper/dr8_v5.10_zlambda_0.3.fit")
    kcorr = kcorr[0].data
    #cindex
    cindex = pyfits.open("/u/ki/rmredd/ki10/redmapper/cindex_dr8_v5.10_zlambda_0.3.fit")
    cindex = cindex[0].data
    print "Done with catalog readin"

    #Getting luminosities
    mag = g['imag'] + (kcorr[:,2]-g['model_mag'][:,3])
    abs_solar = 4.71493
    #Convert to luminosity and get the centrals
    mag = np.log10(mag_to_Lsolar(mag,use_des=0,abs_solar=abs_solar))
    cenmag = np.zeros_like(cindex).astype(float)
    for i in range(len(cenmag)):
        cenmag[i] = mag[cindex[i]]

    print "Done fixing luminosities"

    #Mass-lambda parameters
    lm_param = [2.35e14, 0.843, 0.750, 2.9458, 0.1842]

    #Read in and handle the halo mass function
    mf_files = glob("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/mass_functions_planck/mf_planck*.dat")
    mf_files = np.array(mf_files)
    mf_files.sort()
    #print mf_files
    mf_all = []
    mf_files = mf_files[[1,3,5,7,10]]
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

    #Redshift values
    zmin = np.array([0.1,0.15,0.2,0.25,0.3])
    zmax = np.array([0.15,0.2,0.25,0.3,0.33])
    zmid = (zmin+zmax)/2.
    

    #lambda values and ranges
    lmval = np.array(range(19))*5 + 5
    lm_min = [10., 15., 20., 25., 30., 40.]
    lm_max = [15., 20., 25., 30., 40., 100.]

    #Pick up the photometric errors
    lumbins, cen_err, sat_err = get_iband_magerr(c['imag_err'],cenmag[:,0],c['z_lambda'],g['imag_err'],mag,g['z'],zmin,zmax)
    print "Done getting photometric errors"

    #TESTING PRINTOUT
    #return lumbins, cen_err, sat_err

    #Central parameters
    cen_param = [0.329, -0.9, 24.681, 0.380, 1.46]
    #Satellite parameters
    sat_param = [3.995, 0.844, 23.008, 0.065, 2.008, -0.819, 0.634]

    #Get the central deboost
    deboost_cen = deboost_central(lumbins, cen_err, cen_param, lm_param, mass_param, zmid, lmval)

    #Get the satellite deboost
    deboost_sat = deboost_satellite(lumbins, sat_err, sat_param, lm_param, mass_param, zmid, lm_min, lm_max, mf_all[:,0],mf_all[:,1])
                  
    return [cen_err, sat_err, zmid, lmval, lumbins, deboost_cen, lm_min, lm_max, deboost_sat]
