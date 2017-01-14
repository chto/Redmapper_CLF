#!/usr/bin/python

import numpy as np

#Function for estimating distribution parameters for the lm_param
#distribution

def lm_param_est(dat):
    lm_mean = np.mean(dat,axis=0)

    lm_cov = np.cov(np.transpose(dat))

    return lm_mean, lm_cov
