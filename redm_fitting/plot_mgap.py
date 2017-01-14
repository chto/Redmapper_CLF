#/usr/bin/python

import numpy as np
import matplotlib.pyplot as pyplot

#A few general functions for plotting magnitude gap comparisons

def plot_mgap_set(indir,lm_min,lm_max,zmin,zmax,outfile):
    nz = len(zmin)
    nlm = len(lm_min)

    #Main plotting section
    pyplot.figure(1,[11,8.5])
    for i in range(nlm):
        for j in range(nz):
            mgap = np.loadtxt(indir+"mgap_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            pyplot.subplot(nlm,nz,i*nz+j+1)
            pyplot.semilogy(mgap[:,0],mgap[:,1],'ko-')
            print zmin[j],zmax[j],lm_min[i],lm_max[i],np.sum(mgap[:,0]*mgap[:,1])*(mgap[1,0]-mgap[0,0]),np.sum(mgap[np.where(mgap[:,0]<0)[0],1])*(mgap[1,0]-mgap[0,0])
            
            if i == nlm-1:
                pyplot.xlabel(r'$log(L_{cen}/L_{s1})$')
            if j == 0:
                pyplot.ylabel('P(gap)')
                pyplot.text(-1,2,str(lm_min[i])[0:4]+r'$<\lambda<$'+str(lm_max[i])[0:4])
            if i == 0:
                pyplot.title(str(zmin[j])+'<z<'+str(zmax[j]))
            pyplot.xlim(-1.1,1.1)
            pyplot.ylim(0.01,4)

    pyplot.savefig(outfile,orientation='landscape')
    pyplot.clf()

    return
