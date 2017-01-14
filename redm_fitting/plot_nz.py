#!/usr/bin/python

import numpy as np

import cosmo

#Stupid function for converting n(z) file into something I can plot as a
#nice histogram

def nz_plot_conversion(nz_data):
    vol = np.zeros(len(nz_data))

    for i in range(len(vol)):
        vol[i] = cosmo.comoving_volume(nz_data[i,0],nz_data[i,1],H0=100.)

    x = np.zeros(2*len(nz_data))
    y = np.zeros_like(x)

    for i in range(len(vol)):
        x[2*i] = nz_data[i,0]
        x[2*i+1] = nz_data[i,1]
        delta = x[2*i+1]-x[2*i]

        y[2*i] = nz_data[i,2]/vol[i]*delta*41253.
        y[2*i+1] = nz_data[i,2]/vol[i]*delta*41253.

    return x, y
