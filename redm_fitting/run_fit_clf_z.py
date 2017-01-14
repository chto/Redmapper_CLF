#!/u/ki/dapple/bin/python

import os
import sys

import numpy as np
from glob import glob

import fit_with_covar
import fit_clf

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print >> sys.stderr, "ERROR:  need input"
        print >> sys.stderr, "Usage:  indir"
        sys.exit(1)

    diag_only = False
    do_reg = True

    indir = sys.argv[1]

    infiles_cen = glob(indir+"clf_cen_z*.dat")
    covfiles_cen = glob(indir+"clf_cen_cov*.dat")
    infiles_sat = glob(indir+"clf_sat_z*.dat")
    covfiles_sat = glob(indir+"clf_sat_cov*.dat")

    infiles_cen = np.array(infiles_cen)
    covfiles_cen = np.array(covfiles_cen)
    infiles_sat = np.array(infiles_sat)
    covfiles_sat = np.array(covfiles_sat)

    infiles_cen.sort()
    covfiles_cen.sort()
    infiles_sat.sort()
    covfiles_sat.sort()

    #Read the lmin, lmax, zmin, zmax values from the filenames
    nfiles = len(infiles_cen)
    lmin = np.zeros(nfiles)
    lmax = np.zeros(nfiles)
    zmin = np.zeros(nfiles)
    zmax = np.zeros(nfiles)

    start_cen = []
    start_sat = []

    for i in range(nfiles):
        parts = infiles_cen[i].split('_')
        first = parts.index('z')
        zmin[i] = float(parts[first+1])
        zmax[i] = float(parts[first+2])
        lmin[i] = float(parts[first+4])
        lmax[i] = float(parts[first+5][0:len(parts[first+5])-4])

    lmin_unique = np.unique(lmin)

    nlm = len(lmin_unique)
    fc = open(indir+"param_cen_z.dat",'w')
    fs = open(indir+"param_sat_z.dat",'w')
    print >> sys.stderr, "Entering main loop..."
    #print >> sys.stderr, nfiles,lmin
    for i in range(nlm):
        print >> sys.stderr, "Making list ",i,"..."
        alist = np.where(lmin == lmin_unique[i])[0]
        lmax_unique = np.unique(lmax[alist])
        nlx = len(lmax_unique)
        for j in range(nlx):
            blist = np.where( (lmin == lmin_unique[i]) & (lmax == lmax_unique[j]) )[0]
            if len(blist) == 0:
                print "Oops"
                continue
            #print infiles_cen[blist], covfiles_cen[blist]

            if len(start_cen) == 0:
                start_cen = [10.6, .8, 0.2]
            [chi2_cen, ncen, res, res_covar, x, y, covar] = fit_clf.fit_all_cen(start_cen,fit_clf.func_all_z_cen,fit_clf.func_z_cenparam,infiles_cen[blist],covfiles_cen[blist],zmin[blist],zmax[blist],lmin[blist],lmax[blist],diag_only=diag_only,do_reg=do_reg)
            if chi2_cen > ncen-3:
                start_cen = res
                [chi2_cen, ncen, res, res_covar, x, y, covar] = fit_clf.fit_all_cen(start_cen,fit_clf.func_all_z_cen,fit_clf.func_z_cenparam,infiles_cen[blist],covfiles_cen[blist],zmin[blist],zmax[blist],lmin[blist],lmax[blist],diag_only=diag_only,do_reg=do_reg)
                print res_covar


            print >> fc, lmin[blist[0]], lmax[blist[0]], res[0], np.sqrt(res_covar[0,0]), res[1], np.sqrt(res_covar[1,1]), res[2], np.sqrt(res_covar[2,2]), chi2_cen, ncen-len(res)
            start_cen = res
            

            if len(start_sat) == 0:
                start_sat = [10.26, 1.5, 66, -0.8]
            [chi2_sat, nsat, res, res_covar, x, y, covar] = fit_clf.fit_all_sat(start_sat,fit_clf.func_all_z_sat,fit_clf.func_z_satparam,infiles_sat[blist],covfiles_sat[blist],zmin[blist],zmax[blist],lmin[blist],lmax[blist],diag_only=diag_only,do_reg=do_reg)
            if chi2_sat > ncen-4:
                start_sat = res
                [chi2_sat, nsat, res, res_covar, x, y, covar] = fit_clf.fit_all_sat(start_sat,fit_clf.func_all_z_sat,fit_clf.func_z_satparam,infiles_sat[blist],covfiles_sat[blist],zmin[blist],zmax[blist],lmin[blist],lmax[blist],diag_only=diag_only,do_reg=do_reg)

            print lmin[blist[0]],lmax[blist[0]],chi2_cen,ncen,chi2_sat,nsat
        
            print >> fs, lmin[blist[0]], lmax[blist[0]], res[0], np.sqrt(res_covar[0,0]), res[1], np.sqrt(res_covar[1,1]), res[2], np.sqrt(res_covar[2,2]), res[3], np.sqrt(res_covar[3,3]),chi2_sat,nsat-len(res)
            start_sat = res

    fc.close()
    fs.close()
    print >> sys.stderr, "All done!"
