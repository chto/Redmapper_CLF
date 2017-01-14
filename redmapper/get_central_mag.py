#!/u/ki/dapple/bin/python

#!/usr/bin/env python

import os
import sys

import pyfits
import numpy as np

#Returns central magnitudes and also the galaxies index for the centrals
def get_central_mag(cat,mem,mag,weight_cen=0):
    #First, set up a loop over clusters
    ncluster = len(cat)
    ngals = len(mem)
    
    #Full mags array
    if weight_cen == 1:
        cenmag = np.zeros([ncluster,5])
        cengalindex = np.zeros([ncluster,5])-1
    else:
        cenmag = np.zeros([ncluster,1])
        cengalindex = np.zeros([ncluster,1])-1

    count_lo = 0L
    count_hi = 0L
    ncent = 1
    for i in range(ncluster):
        #if i % 1000 == 0:
        #    print >> sys.stderr, i, ncluster
        #Clusters/gals are sorted on mem_match_id; find matching gals
        while cat['mem_match_id'][i] < mem['mem_match_id'][count_lo]:
            count_lo = count_lo+1
            count_hi = count_lo
        while cat['mem_match_id'][i] == mem['mem_match_id'][count_hi]:
            count_hi = count_hi+1
            if count_hi >= ngals:
                break
        
        #Only care about good centrals
        if weight_cen == 1:
            ncent = cat['ncent_good'][i]
        for j in range(ncent):
            place = np.where( (cat['ra_cent'][i][j] == mem['ra'][count_lo:count_hi]) &
                              (cat['dec_cent'][i][j] == mem['dec'][count_lo:count_hi]) )[0]

            if len(place) == 0:
                print >> sys.stderr, "WARNING:  Possible ID issue in get_central_mag"
                print >> sys.stderr, "Issue at ",cat['mem_match_id'][i],i,j,cat['p_cen'][i][j],cat['z_lambda'][i]
                cenmag[i][j] = 0
                cengalindex[i][j] = -1
                continue

            #And keep the magnitude value
            #print >> sys.stderr, i, j, count_lo,place[0]
            cenmag[i][j] = mag[count_lo+place[0]]
            cengalindex[i][j] = count_lo+place[0]

        count_lo = count_hi

    print >> sys.stderr, "Done getting central mags"
    return cenmag, cengalindex
