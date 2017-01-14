#/usr/bin/python

#Functions for the summing up of normalized CLFs

import numpy as np
import pyfits
from glob import glob
import scipy
import scipy.interpolate

import fit_plm
import fit_psat
from mag_convert import mag_to_Lsolar
import pext_correct

def single_cluster_gal_value(L,lm_val,lm_param,sigma_lm,sat_param,z,mval,nval,use_beta=False):
    Mpiv = lm_param[0]
    
    #Get the approximated mass parameters
    mass_param = fit_plm.nm_approx_third(mval,nval,Mpiv)

    #Get the value from the convolved CLF
    if use_beta:
        weight = fit_psat.func_sat_conv_sch_beta(np.array([L]),0.99*lm_val,1.01*lm_val,lm_param,sigma_lm,sat_param,mass_param,z,mval,nval)[0]
    else:
        weight = fit_psat.func_sat_convolved_phi_ev(np.array([L]),0.99*lm_val,1.01*lm_val,lm_param,sigma_lm,sat_param,mass_param,z,mval,nval)[0]
    #Get the expected mass value, and then the distribution
    my_sigma = np.sqrt(sigma_lm**2 + 1./lm_val)
    r_phi = 1/np.sqrt(1+1./lm_val/my_sigma**2)
    cov = np.array([[my_sigma**2, r_phi*my_sigma*sigma_lm],
                    [r_phi*my_sigma*sigma_lm, sigma_lm**2]])
    sigma1 = 1./np.dot([lm_param[1],sat_param[1]],np.dot(np.linalg.inv(cov),[lm_param[1],sat_param[1]]))
    Mmean = 1./(1+mass_param[2]*sigma1)*(np.dot([lm_param[1],sat_param[1]],np.dot(np.linalg.inv(cov),[np.log(lm_val)-lm_param[3]-lm_param[2]*np.log(1+z),0])) - mass_param[1] )*sigma1
    sigma2 = np.sqrt(1/(1+mass_param[2]*sigma1)*sigma1)
    #Now get the distribution in Lst
    Lst = (sat_param[2] + sat_param[3]*Mmean + sat_param[4]*np.log(1+z))/np.log(10.)
    lumval = L - Lst
    #print L, Lst, z, lm_val, sat_param[2]/np.log(10), Mmean
    return lumval, weight

#Speed up this crap by only calculating the lumval, weight pairs at certian points,
#then interpolating
#Make the grid values we need
def generate_cluster_gal_grid(lm_param,sigma_lm,sat_param,zmid,mval,nval,use_beta=False):
    nL = 30
    nlm = 25
    nz = 14

    L_grid = np.zeros(nL*nlm*nz)
    lm_grid = np.copy(L_grid)
    z_grid = np.copy(L_grid)

    for i in range(nL):
        for j in range(nlm):
            for k in range(nz):
                L_grid[i*nlm*nz + j*nz + k] = 9+i*0.1
                lm_grid[i*nlm*nz + j*nz + k] = np.log(10.)+j*0.12 #note ln(lambda)
                z_grid[i*nlm*nz + j*nz + k] = 0.01+k*0.05

    lumval_grid = np.zeros(nL*nlm*nz)
    weight_grid = np.zeros(nL*nlm*nz)

    for i in range(nL*nlm*nz):
        mf_place = np.argmin( abs( zmid - z_grid[i] ) )
        lumval_grid[i], weight_grid[i] = single_cluster_gal_value(L_grid[i],np.exp(lm_grid[i]),lm_param,sigma_lm,sat_param,z_grid[i],mval[mf_place],nval[mf_place],use_beta=use_beta)

    return L_grid, lm_grid, z_grid, lumval_grid, weight_grid

#Actually doing all the interpolations that we could want at once
def single_cluster_value_interpolated(L,lm_val,z,L_grid,lm_grid,z_grid,
                                      lumval_grid,weight_grid):

    lumval = scipy.interpolate.griddata((L_grid,lm_grid,z_grid),lumval_grid,(L,np.log(lm_val),z))
    weight = np.exp(scipy.interpolate.griddata((L_grid,lm_grid,z_grid),np.log(weight_grid),(L,np.log(lm_val),z)))

    return lumval, weight

def sum_sat_clf(cat,mem,mag,lm_param,sigma_lm,sat_param,mf_all,zmin,zmax,lm_cut=10.):
    dL = 0.02
    lumbins = -1. + np.array(range(150))*dL + dL/2.

    #Get the index matching galaxies to clusters
    index = np.zeros(np.max(cat['mem_match_id'])+1).astype(long)-1
    index[cat['mem_match_id']] = np.array(range(len(cat)))

    #Get the lambda value for each galaxy
    lm_val = cat[index[mem['mem_match_id']]]['lambda_chisq']
    
    clf_all = 0*lumbins
    clf_all_w = 0*lumbins
    nsat = 0.

    zmid = (zmin+zmax)/2.

    glist = np.where( (mem['z'] >= np.min(zmin)) & (mem['z'] < np.max(zmax)) & (lm_val >= lm_cut))[0]
    ngals = len(glist)
    #Loop over galaxies
    print "Beginning the loop... with ",ngals
    for i in np.array(range(ngals)):
        if i % 10000 == 0:
            print "    ",i
        zbin = np.where( (mem[glist[i]]['z'] >= zmin) & (mem[glist[i]]['z'] < zmax) )[0][0]
        mycluster = index[mem[glist[i]]['mem_match_id']]

        lumval, weight = single_cluster_gal_value(mag[glist[i]],lm_val[glist[i]],lm_param,sigma_lm,sat_param,zmid[zbin],mf_all[zbin,:,0],mf_all[zbin,:,1])

        pval = mem[glist[i]]['p']*mem[glist[i]]['pfree']*cat[mycluster]['scaleval']
        #Check to see if this galaxy matches a central galaxy
        clist = np.where( (cat[mycluster]['ra_cent'] == mem[glist[i]]['ra']) & 
                          (cat[mycluster]['dec_cent'] == mem[glist[i]]['dec']) )[0]
        if len(clist) > 0:
            #pval = pval - cat[mycluster]['p_cen'][0]
            pval = pval*(1 - cat[mycluster]['p_cen'][clist[0]])
        if pval < 0:
            continue

        #Add to the counts
        lbin = np.floor((lumval+1.)/dL)
        #print i, lbin
        if (lbin < 0) | (lbin >= len(lumbins)) | np.isnan(lbin):
            continue

        clf_all[lbin] = clf_all[lbin]+pval
        clf_all_w[lbin] = clf_all_w[lbin] + pval/weight
        nsat = nsat + pval

    #Remember to weight by clusters and dL
    nclusters = len(np.where( (cat['z_lambda']> np.min(zmin)) & (cat['z_lambda'] < np.max(zmax)) & (cat['lambda_chisq'] > lm_cut) )[0])
    

    return lumbins, clf_all/nclusters/dL, clf_all_w/nclusters/dL, nsat

def sum_sat_clf_interp(cat,mem,mag,lm_param,sigma_lm,sat_param,mf_all,zmin,zmax,lm_cut=10.,
                       use_beta=False):
    dL = 0.02
    lumbins = -1. + np.array(range(150))*dL + dL/2.

    #Get the index matching galaxies to clusters
    index = np.zeros(np.max(cat['mem_match_id'])+1).astype(long)-1
    index[cat['mem_match_id']] = np.array(range(len(cat)))

    #Get the lambda value for each galaxy
    lm_val = cat[index[mem['mem_match_id']]]['lambda_chisq']
    
    clf_all = 0*lumbins
    clf_all_w = 0*lumbins
    nsat = 0.

    zmid = (zmin+zmax)/2.

    #Make the interpolation grids
    L_grid, lm_grid, z_grid, lumval_grid, weight_grid = generate_cluster_gal_grid(lm_param,sigma_lm,sat_param,zmid,mf_all[:,:,0],mf_all[:,:,1],use_beta=use_beta)

    print "Grids generated"
    #Select the galaxies we want
    glist = np.where( (mem['z'] >= np.min(zmin)) & (mem['z'] < np.max(zmax)) & (lm_val >= lm_cut))[0]

    #Do the interpolation all at once
    print "Interpolating..."
    lumval, weight = single_cluster_value_interpolated(mag[glist],lm_val[glist],mem['z'][glist],L_grid,lm_grid,z_grid,lumval_grid,weight_grid)

    print lumval[0], weight[0], mag[glist[0]], mem['z'][glist[0]]

    #Get corrected probabilities
    p = mem['p'][:]*mem['pfree'][:]
    p = pext_correct.pext_correct(p,mem['chisq'])
    
    ngals = len(glist)
    #Loop over galaxies
    print "Beginning the loop... with ",ngals
    mycluster = index[mem[glist]['mem_match_id']]
    for i in np.array(range(ngals)):
        if i % 10000 == 0:
            print "    ",i
        #zbin = np.where( (mem[glist[i]]['z'] >= zmin) & (mem[glist[i]]['z'] < zmax) )[0][0]

        pval = p[glist[i]]*cat[mycluster[i]]['scaleval']
        #Check to see if this galaxy matches a central galaxy
        clist = np.where( (cat[mycluster[i]]['ra_cent'] == mem[glist[i]]['ra']) & 
                          (cat[mycluster[i]]['dec_cent'] == mem[glist[i]]['dec']) )[0]
        if len(clist) > 0:
            #pval = pval - cat[mycluster]['p_cen'][0]
            pval = pval*(1 - cat[mycluster[i]]['p_cen'][clist[0]])
        if pval < 0:
            continue

        #Add to the counts
        lbin = np.floor((lumval[i]+1.)/dL)
        #print i, lbin
        if (lbin < 0) | (lbin >= len(lumbins)) | np.isnan(lbin):
            continue

        clf_all[lbin] = clf_all[lbin]+pval
        clf_all_w[lbin] = clf_all_w[lbin] + pval/weight[i]
        nsat = nsat + pval

    #Remember to weight by clusters and dL
    nclusters = len(np.where( (cat['z_lambda']> np.min(zmin)) & (cat['z_lambda'] < np.max(zmax)) & (cat['lambda_chisq'] > lm_cut) )[0])
    

    return lumbins, clf_all/nclusters/dL, clf_all_w/nclusters/dL, clf_all


#Wrapper to handle all the data read-in and manipulation
def sum_sat_clf_s82(lm_cut=10.):
    #Read in the catalogs
    c = pyfits.open("/nfs/slac/g/ki/ki19/des/erykoff/clusters/process/stripe82_redmapper_v5.10/run_ubermem/stripe82_run_redmapper_v5.10_lgt5_catalog.fit")
    c = c[1].data
    g = pyfits.open("/nfs/slac/g/ki/ki19/des/erykoff/clusters/process/stripe82_redmapper_v5.10/run_ubermem/stripe82_run_redmapper_v5.10_lgt5_catalog_members.fit")
    g = g[1].data

    kcorr = pyfits.open("/u/ki/rmredd/ki10/redmapper/s82_v5.10_uber_zlambda_0.3.fit")
    kcorr = kcorr[0].data

    #Set up the luminosities
    mag = g['imag'] + (kcorr[:,2]-g['model_mag'][:,2])
    mag = np.log10(mag_to_Lsolar(mag,abs_solar=4.71493))

    #Set parameters
    h = 0.6704
    Mpiv = 2.35e14
    sigma_lm = 0.1842
    lm_param = [Mpiv, 0.857, 1.547, 2.7226, 0.1842]
    fits_dir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10/"
    schain = np.loadtxt(fits_dir+"chain_sat_ev_all.dat")
    sat_param = schain[ np.argmax(schain[:,-1]) ]

    #Pick up mass functions
    zmin = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    zmax = zmin+0.1
    #Make the mass_param list for the CLFs
    mf_files = glob(fits_dir+"../mass_functions_planck/mf_planck_*.dat")
    mf_files = np.array(mf_files)
    mf_files.sort()
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

    print "Setup done; beginning main section"
    lumbins, clf_all, clf_all_w, nsat = sum_sat_clf(c,g,mag,lm_param,sigma_lm,sat_param,mf_all,zmin,zmax,lm_cut=lm_cut)

    return lumbins, clf_all, clf_all_w, nsat

#Version that runs using interpolation instead -- should be much faster
#Runs on DR8 data
def sum_sat_clf_dr8_interp(lm_cut=10.,use_beta=False):
    #Read in the catalogs
    c = pyfits.open("/nfs/slac/g/ki/ki19/des/erykoff/clusters/process/dr8_redmapper_v5.10/run_ubermem/dr8_run_redmapper_v5.10_lgt5_catalog.fit")
    c = c[1].data
    g = pyfits.open("/nfs/slac/g/ki/ki19/des/erykoff/clusters/process/dr8_redmapper_v5.10/run_ubermem/dr8_run_redmapper_v5.10_lgt5_catalog_members.fit")
    g = g[1].data

    kcorr = pyfits.open("/u/ki/rmredd/ki10/redmapper/dr8_v5.10_zlambda_0.3.fit")
    kcorr = kcorr[0].data

    #Set up the luminosities
    mag = g['imag'] + (kcorr[:,2]-g['model_mag'][:,3])
    mag = np.log10(mag_to_Lsolar(mag,abs_solar=4.71493))
 #Set parameters
    h = 0.6704
    Mpiv = 2.35e14
    sigma_lm = 0.1842
    lm_param = [Mpiv, 0.857, 1.547, 2.7226, 0.1842]
    fits_dir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10/"
    if use_beta:
        fits_dir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10_schcorr/"
    schain = np.loadtxt(fits_dir+"chain_sat_ev_all.dat")
    sat_param = schain[ np.argmax(schain[:,-1]) ]

    #Pick up mass functions
    zmin = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
    zmax = np.array([0.15, 0.2, 0.25, 0.3, 0.33])
    #Make the mass_param list for the CLFs
    mf_files = glob(fits_dir+"../mass_functions_planck/mf_planck_*.dat")
    mf_files = np.array(mf_files)
    mf_files.sort()
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

    print "Setup done; beginning main section"
    lumbins, clf_all, clf_all_w, nsat = sum_sat_clf_interp(c,g,mag,lm_param,sigma_lm,sat_param,mf_all,zmin,zmax,lm_cut=lm_cut,use_beta=use_beta)

    return lumbins, clf_all, clf_all_w, nsat

#Version that runs using interpolation instead -- should be much faster
def sum_sat_clf_s82_interp(lm_cut=10.):
    #Read in the catalogs
    c = pyfits.open("/nfs/slac/g/ki/ki19/des/erykoff/clusters/process/stripe82_redmapper_v5.10/run_ubermem/stripe82_run_redmapper_v5.10_lgt5_catalog.fit")
    c = c[1].data
    g = pyfits.open("/nfs/slac/g/ki/ki19/des/erykoff/clusters/process/stripe82_redmapper_v5.10/run_ubermem/stripe82_run_redmapper_v5.10_lgt5_catalog_members.fit")
    g = g[1].data

    kcorr = pyfits.open("/u/ki/rmredd/ki10/redmapper/s82_v5.10_uber_zlambda_0.3.fit")
    kcorr = kcorr[0].data

    #Set up the luminosities
    mag = g['imag'] + (kcorr[:,2]-g['model_mag'][:,2])
    mag = np.log10(mag_to_Lsolar(mag,abs_solar=4.71493))
 #Set parameters
    h = 0.6704
    Mpiv = 2.35e14
    sigma_lm = 0.1842
    lm_param = [Mpiv, 0.857, 1.547, 2.7226, 0.1842]
    fits_dir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10/"
    schain = np.loadtxt(fits_dir+"chain_sat_ev_all.dat")
    sat_param = schain[ np.argmax(schain[:,-1]) ]

    #Pick up mass functions
    zmin = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    zmax = zmin+0.1
    #Make the mass_param list for the CLFs
    mf_files = glob(fits_dir+"../mass_functions_planck/mf_planck_*.dat")
    mf_files = np.array(mf_files)
    mf_files.sort()
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

    print "Setup done; beginning main section"
    lumbins, clf_all, clf_all_w, nsat = sum_sat_clf_interp(c,g,mag,lm_param,sigma_lm,sat_param,mf_all,zmin,zmax,lm_cut=lm_cut)

    return lumbins, clf_all, clf_all_w, nsat

if __name__ == "__main__":
    #Generic call for queue submission convenience
    lumbins, clf_all, clf_all_w = sum_sat_clf.sum_sat_clf_dr8_interp(lm_cut=20.,use_beta=True)

    f = open("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/test_galaxies/clf_sum_dr8_lm_20_beta.dat",'w')
    for i in range(len(lumbins)):
        print >> f, lumbins[i], clf_all_w[i], clf_all[i]
    f.close()
