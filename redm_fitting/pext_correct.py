#!/u/ki/dapple/bin/python

import os
import sys
import numpy as np
import pyfits

#Functions for correcting the p_ext probabilities; 
#Correction mostly to account for presence of blue galaxies
#in reality
#See Eduardo's work for more details

#Note that this may be updated in the future
def pext_correct(p_old, chi2):
    '''
    Currently, p_old (initial p_ext) is the only input

    Calculates delta_p based on fixed parameter set

    May later add L dependency?

    Note that this requires input of chi^2 for each galaxy
    Note that this requires read-in of epsilon_functions.fit file

    Includes correction for p_new > 1 or p_new < 0
    '''

    #Old version
    #Parameters -- drawn from GAMA
    #p0 = 0.00494789
    #p1 = 0.655880
    #p2 = -2.86675
    #p3 = 2.19239

    #Change in probabilities
    #delta = p0 + p1*p_old + p2*p_old*p_old + p3*p_old**3

    #p_new = p_old + delta

    #Updated, current version
    #Read in fit data
    if (os.path.isfile("epsilon_functions.fit")==False):
        print >> sys.stderr, "WARNING: Unable to perform correction to probabilities."
        print >> sys.stderr, "Running without correction."
        return p_old

    dat = pyfits.open("epsilon_functions.fit")
    dat = dat[1].data

    #Interpolation to obtain the epsilon values
    lnchi2 = np.log(chi2)
    e_chi2 = np.interp(lnchi2,dat['lnchisq'],dat['epschi'])
    e_blue = np.interp(lnchi2,dat['lnchisq'],dat['epsblue'])

    #Constants
    #c = 0.061
    #delta = -0.0168
    #Updated constant
    #c = 0.066
    #delta = -0.02123
    #Constants updated again -- and again
    c = 0.070
    delta = -0.075

    #And putting it all together -- see Rozo et al for details
    p_new = p_old * (1+e_chi2)/(1+p_old*e_chi2) / (1+p_old*c) / (1+p_old*e_blue) * (1+delta) / (1+p_old*delta)

    #Checks for impossibly high/low p_new
    alist = np.where(p_new > 1)[0]
    if len(alist) > 0:
        p_new[alist] = 0*alist + 1
    blist = np.where(p_new < 0)[0]
    if len(blist) > 0:
        p_new[blist] = 0*blist

    return p_new

def pext_correct_partial(p_old):
    c = 0.070

    #First correction for correlated structure
    p_next = p_old * 1./(1+p_old*c)

    #Estimate the delta correction
    delta = np.sum(p_next)/np.sum(p_old)-1.

    #And correct
    p_new = p_next/(1+p_next*delta)
    return p_new
