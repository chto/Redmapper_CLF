#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.ticker as ticker
import matplotlib.axes as axes
import sys
from glob import glob
import pylab
import pyfits

import cosmo
import fit_plm
import fit_clf
import fit_psat
import get_p_bcg_cen as get_pbcg
import mass_matching
import run_zev_fit_clf as run_zev
import get_sys_err

#Various plots for test output

#Do a single plot, comparing potentially several different sets of satellite parameters
def plot_clf_sat_test(infile,param,mass_param,lambda_val,z,param_sat,outfile,dir_param=[],
                      dir_param_alt=[],zvals=[],Lmin=[]):
    #Read in the data clf
    sat = np.loadtxt(infile)

    pyplot.semilogy(sat[:,0],sat[:,1],'o')
    pyplot.errorbar(sat[:,0],sat[:,1],sat[:,2],ecolor='b',fmt=None)
    pyplot.xlabel('L')
    pyplot.ylabel('Phi')
    pyplot.ylim(0.01,300)
    pyplot.title("z="+str(z)+" lm="+str(lambda_val))

    #Now plot each of the desired fits
    #print len(param_sat)
    colorlist = ['g','r','c','m','y']
    local_param_all = []
    for i in range(len(param_sat)):
        local_param = fit_psat.get_clf_params(lambda_val,z,param,param_sat[i],mass_param,
                                              zvals=zvals,Lmin_val=Lmin)
        local_param_all.append(local_param)
        pyplot.plot(sat[:,0],fit_clf.func_sat(local_param,sat[:,0]),colorlist[i])

    if len(dir_param) > 0:
        pyplot.plot(sat[:,0],fit_clf.func_sat(dir_param,sat[:,0]),'k')

    if len(dir_param_alt) > 0:
        pyplot.plot(sat[:,0],fit_clf.func_sat(dir_param_alt,sat[:,0]),'k--')

    pyplot.plot([9.8,9.8],[0.01,300],'k:')

    pyplot.savefig(outfile)
    pyplot.clf()

    #Test outputs
    Lbins = fit_psat.Lmin_eli(z)+np.array(range(300))*0.01
    slist = np.where(sat[:,1] > 0)[0]
    print lambda_val-1, np.sum(sat[slist,1])*0.08, np.sum(fit_clf.func_sat(local_param_all[0],Lbins))*0.01, np.sum(fit_clf.func_sat(local_param_all[1],Lbins))*0.01, np.sum(fit_clf.func_sat(dir_param,Lbins))*0.01

    return


#Plotting test of fit parameters for satellite CLFs
def plot_clf_sat_params(param,mass_param,lambda_val,z,nz,param_sat,dir_param,dir_p_err,outdir,
                        zvals=[],Lmin=[]):
    #First, get params resulting from overall fits
    over_params = np.zeros([len(z),len(param_sat),3])
    for i in range(len(z)):
        for j in range(len(param_sat)):
            over_params[i,j,:] = fit_psat.get_clf_params(lambda_val[i],z[i],param,param_sat[j],mass_param[i],zvals=zvals,Lmin_val=Lmin)

    #Run for L* first -- lambda on x-axis, redshift indicated by color
    colorlist = ['b','g','r','m','k']
    nlm = len(lambda_val)/nz
    for i in range(nz):
        #Individual fits first
        glist = np.where(z == z[i*nlm])[0]
        pyplot.semilogx(lambda_val[glist],dir_param[glist,0],colorlist[i]+'o')
        pyplot.errorbar(lambda_val[glist],dir_param[glist,0],dir_p_err[glist,0],fmt=None,ecolor=colorlist[i])

        #Now, plot the reported fits -- only does two
        pyplot.plot(lambda_val[glist],over_params[glist,0,0],colorlist[i])

        if len(param_sat) > 1:
            pyplot.plot(lambda_val[glist],over_params[glist,1,0],colorlist[i]+"--")
    pyplot.xlabel('lambda')
    pyplot.ylabel('Lsat')            
        
    pyplot.savefig(outdir+"Lsat_test.png")
    pyplot.clf()
    
    #Now do phi*
    for i in range(nz):
        #Individual fits first
        glist = np.where(z == z[i*nlm])[0]
        pyplot.loglog(lambda_val[glist],dir_param[glist,1],colorlist[i]+'o')
        pyplot.errorbar(lambda_val[glist],dir_param[glist,1],dir_p_err[glist,1],fmt=None,ecolor=colorlist[i])

        #Now, plot the reported fits -- only does two
        pyplot.plot(lambda_val[glist],over_params[glist,0,1],colorlist[i])

        if len(param_sat) > 1:
            pyplot.plot(lambda_val[glist],over_params[glist,1,1],colorlist[i]+"--")
    pyplot.xlabel('lambda')
    pyplot.ylabel('phi*')    

    pyplot.savefig(outdir+"phist_test.png")
    pyplot.clf()
    
    #And alpha
    for i in range(nz):
        #Individual fits first
        glist = np.where(z == z[i*nlm])[0]
        pyplot.semilogx(lambda_val[glist],dir_param[glist,2],colorlist[i]+'o')
        pyplot.errorbar(lambda_val[glist],dir_param[glist,2],dir_p_err[glist,2],fmt=None,ecolor=colorlist[i])

        #Now, plot the reported fits -- only does two
        pyplot.plot(lambda_val[glist],over_params[glist,0,2],colorlist[i])

        if len(param_sat) > 1:
            pyplot.plot(lambda_val[glist],over_params[glist,1,2],colorlist[i]+"--")
    pyplot.xlabel('lambda')
    pyplot.ylabel('alpha')    

    pyplot.savefig(outdir+"alpha_test.png")
    pyplot.clf()

    return

#Run a set of n(lambda) plots, dump to a file, and meanwhile, print out chi^2 estimates 
#(diagonal and w/ full covariance matrix)
#Note this version is tidy and makes only a single output file
def plot_nlambda_set_nice(param,zmin,zmax,mf_files,nlm_files,outdir,xlimits=[10,100],ylimits=[1.1e-9,1e-5],bigger=False,Alm_ev=False,inset_args=[],area_correct=[]):
    '''
    Inputs:
    
    param : vector of input parameters, with Mpiv, A_lambda, B_lambda, ln lm 0 in that order
    zmin 
    zmax
    mf_files
    nlm_files
    cov_files
    outdir

    Assumes scatter is fixed to ~20%
    '''

    Mpiv = param[0]
    A_lm = param[1]
    B_lm = param[2]
    lnlm0 = param[3]

    inparam = np.zeros(8)
    inparam[0] = 0.1842
    inparam[1] = 0.4245
    inparam[2] = 0.1351
    inparam[4] = A_lm
    inparam[5] = 24.1
    inparam[6] = 0.489
    inparam[7] = 1.3667

    h = 0.6704
    bias = 3.
    pyplot.figure(1,[11.,8.5])
    pyplot.subplots_adjust(wspace=0.2)
    for i in range(len(zmin)):
        if bigger:
            ax = pyplot.subplot(3,3,i+1)
        else:
            ax = pyplot.subplot(2,3,i+1)            
        zmid = (zmin[i]+zmax[i])/2.
        inparam[3] = lnlm0 + B_lm*np.log((1+zmid)/1.3)
        if Alm_ev:
            inparam[4] = A_lm*((1+zmid)/1.3)**param[4]

        #Get the halo mass function
        mf = np.loadtxt(mf_files[i])
        
        dat = np.loadtxt(nlm_files[i])
        lm_val = dat[:,0]+0.5
        
        nlm = np.zeros_like(lm_val)
        for j in range(len(lm_val)):
           [ nlm[j], temp ] = fit_plm.n_lm_convolved(lm_val[j],5e-6,1.5,0.69,0.25,Mpiv,inparam,zmid,inparam[7],mf[:,0]/h,mf[:,1]*h**3, extra=True)
        if len(area_correct) > 0:
            dat[:,1] = dat[:,1]/area_correct[i]
            dat[:,2] = dat[:,2]/area_correct[i]
            #print "Woo: ",area_correct[i]

        pyplot.loglog(lm_val,dat[:,1]/h**3,'b.')
        #pyplot.errorbar(lm_val,dat[:,1],np.sqrt(np.diag(cov_sv)),ecolor='k',fmt=None)
        pyplot.errorbar(lm_val,dat[:,1]/h**3,dat[:,2]/h**3,ecolor='b',fmt=None)
        pyplot.loglog(lm_val,nlm/h**3,'r')
        pyplot.ylim(ylimits)
        pyplot.xlim(xlimits)
        pyplot.text(11,ylimits[0]/1.1*2,str(zmin[i])+'<z<'+str(zmax[i]))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.ticklabel_format(style='plain',axis='x')

    pyplot.xlabel(r'$\lambda$', fontsize=18)
    #pyplot.ylabel(r'$n(\lambda)\ [(Mpc/h)^{-3} d\lambda^{-1}]$')
    pyplot.text(2.*xlimits[0]/xlimits[1],1e-4,r'$n(\lambda)\ [(Mpc/h)^{-3} d\lambda^{-1}]$',rotation='vertical', fontsize=18)

    #Add one more figure that shows the mass-lambda relationship for each
    #Redshift value
    if bigger:
        ax = pyplot.subplot(3,3,9)
    else:
        ax = pyplot.subplot(2,3,6)
    
    mvals = 10.**(np.array(range(200))*0.01+13.5)
    colors = ['k','b','g','r','c','m','purple','brown','k--']
    for i in range(len(zmin)):
        zmid = (zmin[i]+zmax[i])/2.
        A_lm_temp = A_lm*((1+zmid)/1.3)**param[4]
        lm_val = np.exp(lnlm0)*(mvals/Mpiv)**A_lm_temp*((1+zmid)/1.3)**B_lm
        pyplot.loglog(mvals,lm_val,colors[i])
    pyplot.xlabel(r'$M_{vir}$ $[M_\odot]$',fontsize=18)
    pyplot.xlim(10.**13.5, 10.**15.5)
    pyplot.ylim(5,250)
    pyplot.plot([10.**13.5, 10.**15.5],[10,10],'k--')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.ticklabel_format(style='plain',axis='y')
    pyplot.text(2e13,35,r'$\lambda$',rotation='vertical')
    #pyplot.ticklabel_format(axis='y',style='plain')

    #Doing the inset in the last plot window
    if len(inset_args) > 0 and Alm_ev:
        zmid = 0.1
        ax = pylab.axes([.805,.13,.095,.1])
        lm_val = np.exp(lnlm0)*(mvals/Mpiv)**(A_lm*((1+zmid)/1.3)**param[4])*((1+zmid)/1.3)**B_lm
        pyplot.loglog(mvals,lm_val,'k')
        lm_param_alt = inset_args[0]
        lm_val = np.exp(lm_param_alt[3])*(mvals/lm_param_alt[0])**(lm_param_alt[1]*((1+zmid)/1.3)**lm_param_alt[4])*((1+zmid)/1.3)**lm_param_alt[2]
        pyplot.loglog(mvals,lm_val,'b--')

        zmid = 0.3
        lm_val = np.exp(lnlm0)*(mvals/Mpiv)**(A_lm*((1+zmid)/1.3)**param[4])*((1+zmid)/1.3)**B_lm
        pyplot.loglog(mvals,lm_val,'g')
        lm_param_alt = inset_args[0]
        lm_val = np.exp(lm_param_alt[3])*(mvals/lm_param_alt[0])**(lm_param_alt[1]*((1+zmid)/1.3)**lm_param_alt[4])*((1+zmid)/1.3)**lm_param_alt[2]
        pyplot.loglog(mvals,lm_val,'c--')
        pyplot.xlim(10.**13.5, 10.**15.5)
        pyplot.ylim(5,250)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.ticklabel_format(style='plain',axis='y')
                                
    #pyplot.tight_layout()
    pyplot.savefig(outdir+"nlm_set.ps",orientation='landscape')
    pyplot.clf()
    #pyplot.close(1)

    return


#Plotting for comparing two data sets
def plot_nlambda_data_nice(zmin,zmax,nlm_files,nlm_files_alt,outdir,xlimits=[10,100],
                           ylimits=[1.1e-9,1e-5],bigger=False):

    h = 0.6704
    pyplot.figure(1,[11.,8.5])
    for i in range(len(zmin)):
        if bigger:
            ax = pyplot.subplot(3,3,i+1)
        else:
            ax = pyplot.subplot(2,3,i+1)            
        zmid = (zmin[i]+zmax[i])/2.

        dat = np.loadtxt(nlm_files[i])
        dat_alt = np.loadtxt(nlm_files_alt[i])
        
        lm_val = dat[:,0]+0.5

        pyplot.loglog(lm_val,dat[:,1]/h**3,'b.')
        pyplot.loglog(lm_val,dat_alt[:,1]/h**3,'r.')
        pyplot.ylim(ylimits)
        pyplot.xlim(xlimits)
        pyplot.text(11,ylimits[0]/1.1*2,str(zmin[i])+'<z<'+str(zmax[i]))
        
    pyplot.xlabel(r'$\lambda$')

    pyplot.text(3.*xlimits[0]/xlimits[1],1e-4,r'$n(\lambda)\ [(Mpc/h)^{-3} d\lambda^{-1}]$',rotation='vertical')
    
    pyplot.savefig(outdir+"nlm_set_comp.ps",orientation='landscape')
    pyplot.clf()
    
    return

#Do a single plot, comparing central data against result from fit parameters
def plot_clf_cen_test(infile,param,lm_param,mass_param,lambda_val,z,outfile):
    #Read in the data clf
    cen = np.loadtxt(infile)

    pyplot.semilogy(cen[:,0],cen[:,1],'bo')
    pyplot.errorbar(cen[:,0],cen[:,1],cen[:,2],ecolor='b',fmt=None)
    pyplot.xlabel('L')
    pyplot.ylabel(r'\Phi')
    pyplot.ylim(0.01,10)
    pyplot.title("z="+str(z)+" lm="+str(lambda_val))

    B_L = param[4]
    Mpiv = lm_param[0]
    A_lm = lm_param[1]
    B_lm = lm_param[2]
    lnlm0 = lm_param[3]
    full_param = np.zeros(8)
    full_param[0] = 0.1842
    full_param[1] = param[0]
    full_param[2] = param[1]
    full_param[3] = lnlm0 + B_lm*np.log(1+z)
    full_param[4] = A_lm
    full_param[5] = param[2]
    full_param[6] = param[3]
    full_param[7] = param[4]

    #Now plot each of the desired fits
    #print len(param_cen)
    pyplot.plot(cen[:,0],fit_plm.p_L(10.**cen[:,0],lambda_val,mass_param[0],mass_param[1],mass_param[2],mass_param[3],Mpiv,full_param,z,B_L),'r')

    pyplot.savefig(outfile)
    pyplot.clf()

    return


#Main CLFs plotting routine
#Plots a 3x3 grid, showing three redshifts and three lambda ranges
#Alternate plotting set plots fits for version with extra evolution in redshift
def plot_clf_set(indir, indir_uber, lm_param, cen_param, sat_param, 
                 lm_min,lm_max,lm_med, mass_param, zmin, zmax, iband_err, mf_all, outfile,
                 abs_solar=4.67966,fix_alpha=True,lm_param_alt=[],sat_param_alt=[],
                 cen_param_alt=[],phi_ev=False,use_beta=False):
    zmid = (zmin+zmax)/2.
    nz = len(zmin)
    nlm = len(lm_min)

    in_param = np.zeros(8)
    Mpiv = lm_param[0]
    A_lm = lm_param[1]
    B_lm = lm_param[2]
    lnlm0 = lm_param[3]
    if len(lm_param) > 4:
        sigma_lm = lm_param[4]
    else:
        sigma_lm = 0.1842
    #print >> sys.stderr, "sigma_lm TEST: ",sigma_lm

    in_param[0] = sigma_lm
    in_param[1] = cen_param[0] #sigma_L
    in_param[2] = cen_param[1] #r
    in_param[3] = lnlm0 + B_lm*np.log(1+0) #log(lm0)
    in_param[4] = A_lm
    in_param[5] = cen_param[2] #ln L0
    in_param[6] = cen_param[3] #A_L
    in_param[7] = cen_param[4] #B_L
    B_L = cen_param[4]
    
    in_param_alt = 0*in_param

    #Collect Lmin data
    zvals = np.arange(1000)*0.001
    Lmin_val = fit_psat.Lmin_eli(zvals,abs_solar=abs_solar)

    #Actual plotting section
    pyplot.figure(1,[11.,8.5])
    for i in range(nlm):
        for j in range(nz):
            #Read in the centrals only data
            cen = np.loadtxt(indir_uber+"clf_cen_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            cen_covar = np.loadtxt(indir_uber+"clf_cen_covar_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            #Read in the satellite data
            sat = np.loadtxt(indir+"clf_sat_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            sat_uber = np.loadtxt(indir_uber+"clf_sat_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            sat_covar = np.loadtxt(indir_uber+"clf_sat_covar_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            #Get characteristic lambda from limited satellite count
            mylambda = np.sum(sat[np.where(sat[:,1]>0)[0],1])*(sat[1,0]-sat[0,0])+1
            #Fix to input lm_med if we don't have a distinct input result
            if indir == indir_uber:
                mylambda = lm_med[i]
                
            pyplot.subplot(nlm,nz,i*nz+j+1)
            pyplot.semilogy(cen[:,0],cen[:,1],'ko')
            pyplot.errorbar(cen[:,0],cen[:,1],np.sqrt(np.diag(cen_covar)),ecolor='k',fmt=None)

            if i == nlm-1:
                pyplot.xlabel(r'$L_i$ [$L_\odot/h^2$]')
            if j == 0:
                pyplot.ylabel(r'$\Phi$ $[(d log L)^{-1}]$')
                pyplot.text(10.3,80,str(lm_min[i])[0:4]+r'$<\lambda<$'+str(lm_max[i])[0:4])
            if i == 0:
                pyplot.title(str(zmin[j])+'<z<'+str(zmax[j]))
            pyplot.xlim(9.,11.5)
            yrange = [0.011,3e2]
            pyplot.ylim(yrange)

            #Set parameters for this lm_med, z combination
            in_param[3] = lnlm0 + B_lm*np.log(1+zmid[j])
            pyplot.plot(cen[:,0],fit_plm.p_L(10.**cen[:,0],mylambda,mass_param[j][0],
                                             mass_param[j][1],mass_param[j][2],mass_param[j][3],Mpiv,
                                             in_param,zmid[j],B_L),'b')
            
            if len(lm_param_alt)==0:
                lm_param_alt = lm_param
            if len(cen_param_alt)>0:
                in_param_alt[0] = sigma_lm
                in_param_alt[1] = cen_param_alt[0] #sigma_L
                in_param_alt[2] = cen_param_alt[1] #r
                in_param_alt[3] = lm_param_alt[3] + lm_param_alt[2]*np.log(1+0) #log(lm0)
                in_param_alt[4] = lm_param_alt[1]
                in_param_alt[5] = cen_param_alt[2] #ln L0
                in_param_alt[6] = cen_param_alt[3] #A_L
                in_param_alt[7] = cen_param_alt[4] #B_L
                B_L_alt = cen_param[4]
                pyplot.plot(cen[:,0],fit_plm.p_L(10.**cen[:,0],mylambda,mass_param[j][0],
                                                 mass_param[j][1],mass_param[j][2],mass_param[j][3],Mpiv,
                                                 in_param_alt,zmid[j],B_L_alt),'r')
            #Plotting the satellite part
            pyplot.semilogy(sat_uber[:,0],sat_uber[:,1],'k^')
            pyplot.errorbar(sat_uber[:,0],sat_uber[:,1],sat_uber[:,2],fmt=None,ecolor='k')

            if phi_ev:
                myparam = [sat_param[0] + sat_param[6]*np.log(1+zmid[j]), sat_param[1], sat_param[2],
                           sat_param[3], sat_param[4], sat_param[5]]
                sat_fit = fit_psat.func_sat_convolved(sat_uber[:,0],lm_min[i],lm_max[i],lm_param,in_param[0],myparam,mass_param[j],zmid[j],mf_all[j][:,0],mf_all[j][:,1])
            if use_beta:
                sat_fit = fit_psat.func_sat_conv_sch_beta(sat_uber[:,0],lm_min[i],lm_max[i],lm_param,in_param[0],sat_param,mass_param[j],zmid[j],mf_all[j][:,0],mf_all[j][:,1])
            if not phi_ev and not use_beta:
                sat_fit = fit_psat.func_sat_convolved(sat_uber[:,0],lm_min[i],lm_max[i],lm_param,in_param[0],sat_param,mass_param[j],zmid[j],mf_all[j][:,0],mf_all[j][:,1])
            pyplot.semilogy(sat_uber[:,0],sat_fit,'b--')
            #print zmid[j],mylambda

            if len(sat_param_alt) > 0:
                myparam_alt = [sat_param_alt[0] + sat_param_alt[2]*np.log(1+zmid[j]), sat_param_alt[1],
                               sat_param_alt[3], sat_param_alt[4], sat_param_alt[5], 
                               sat_param_alt[6] + sat_param_alt[7]*zmid[j]]
                if phi_ev:
                    myparam_alt = [sat_param_alt[0] + sat_param_alt[6]*np.log(1+zmid[j]), 
                                   sat_param_alt[1], sat_param_alt[2],
                                   sat_param_alt[3], sat_param_alt[4], sat_param_alt[5]]
                #print sat_param
                #print myparam_alt
                sat_fit_a = fit_psat.func_sat_convolved(sat_uber[:,0],lm_min[i],lm_max[i],lm_param_alt,in_param[0],myparam_alt,mass_param[j],zmid[j],mf_all[j][:,0],mf_all[j][:,1])
                pyplot.semilogy(sat_uber[:,0],sat_fit_a,'r--')
                #print sat_fit_a/sat_fit
            

            #Quick calculation of chi^2 in this bin
            Lcut = fit_psat.Lmin_eli(zmid[j])
            slist = np.where((sat_uber[:,0] > Lcut) & (sat_uber[:,1] > 0))[0]
            slist = slist[1:]
            delta = sat_uber[slist,1] - sat_fit[slist]
            cov_trim = np.copy(sat_covar)
            cov_trim = cov_trim[:,slist]
            cov_trim = cov_trim[slist,:]
            chi2 = np.dot(delta,np.dot(np.linalg.inv(cov_trim),delta))
            #print chi2, len(slist), mylambda, zmid[j]
            #pyplot.text(9.1, 1, r'$\chi^2/N=$'+str(chi2)[0:4]+"/"+str(len(slist)))
            #print delta/sat_uber[slist,1]
            #Plot the cutoff luminosity
            pyplot.plot([Lcut,Lcut],yrange,'k--')

            #OFFSET TEST PRINT
            slist = np.where(sat_uber[:,1] > 0)[0]
            Lcut_bin = np.min(np.where(cen[:,0] > fit_psat.Lmin_eli( zmid[j], abs_solar=abs_solar))[0])
            Lcut_bin_lo = np.min(np.where(cen[:,0] > fit_psat.Lmin_eli( zmid[j], abs_solar=abs_solar)-np.log10(0.2)+np.log10(0.1))[0])
            print fit_psat.Lmin_eli( zmid[j], abs_solar=abs_solar)
            #myparam = [sat_param_alt[0] + sat_param_alt[6]*np.log(1+zmid[j]), 
            #           sat_param_alt[1], sat_param_alt[2],
            #           sat_param_alt[3], sat_param_alt[4], sat_param_alt[5]]
            #print zmid[j],lm_min[i],sat_uber[[Lcut_bin,Lcut_bin_lo],0],sat_uber[[Lcut_bin,Lcut_bin_lo],1]/fit_psat.func_sat_convolved(sat_uber[[Lcut_bin,Lcut_bin_lo],0],lm_min[i],lm_max[i],lm_param,in_param[0],myparam,mass_param[j],zmid[j],mf_all[j][:,0],mf_all[j][:,1])
    
    pyplot.savefig(outfile,orientation='landscape')
    pyplot.clf()
    #pyplot.close(1)
    return

#Updated plotting routine for handling new parameterization
def plot_clf_set_new(indir, lm_param, cen_param, sat_param, lm_min,lm_max,lm_med, 
                     mass_param, zmin, zmax, mf_all, outfile, abs_solar=4.67966,
                     lm_param_alt=[],
                     sat_param_alt=[],cen_param_alt=[]):
    zmid = (zmin+zmax)/2.
    nz = len(zmin)
    nlm = len(lm_min)

    in_param = np.zeros(8)
    Mpiv = lm_param[0]
    A_lm = lm_param[1]
    B_lm = lm_param[2]
    lnlm0 = lm_param[3]
    if len(lm_param) > 5:
        sigma_lm = lm_param[5]
    else:
        sigma_lm = 0.1842

    in_param[0] = sigma_lm
    in_param[1] = cen_param[0] #sigma_L
    in_param[2] = cen_param[1] #r
    in_param[3] = lnlm0 + B_lm*np.log(1+0) #log(lm0)
    in_param[4] = A_lm
    in_param[5] = cen_param[2] #ln L0
    in_param[6] = cen_param[3] #A_L
    in_param[7] = cen_param[4] #B_L
    B_L = cen_param[4]
    
    in_param_alt = 0*in_param

    #Collect Lmin data
    zvals = np.arange(1000)*0.001
    Lmin_val = fit_psat.Lmin_eli(zvals,abs_solar=abs_solar)

    #Actual plotting section
    pyplot.figure(1,[11.,8.5])
    for i in range(nlm):
        for j in range(nz):
            #Read in the centrals only data
            cen = np.loadtxt(indir+"clf_cen_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            #Read in the satellite data
            sat = np.loadtxt(indir+"clf_sat_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")

            #Fix to input lm_med if we don't have a distinct input result
            if indir == indir:
                mylambda = lm_med[i]
                
            pyplot.subplot(nlm,nz,i*nz+j+1)
            pyplot.semilogy(cen[:,0],cen[:,1],'ko')
            pyplot.errorbar(cen[:,0],cen[:,1],cen[:,2],ecolor='k',fmt=None)

            if i == nlm-1:
                pyplot.xlabel(r'$L_i$ [$L_\odot/h^2$]')
            if j == 0:
                pyplot.ylabel(r'$\Phi$ $[(d log L)^{-1}]$')
                pyplot.text(10.3,80,str(lm_min[i])[0:4]+r'$<\lambda<$'+str(lm_max[i])[0:4])
            if i == 0:
                pyplot.title(str(zmin[j])+'<z<'+str(zmax[j]))
            pyplot.xlim(9.,11.5)
            yrange = [0.011,3e2]
            pyplot.ylim(yrange)

            #Set parameters for this lm_med, z combination
            in_param[3] = lnlm0 + B_lm*np.log((1+zmid[j])/1.3)
            in_param[4] = A_lm*((1+zmid[j])/1.3)**lm_param[4]
            in_param[5] = cen_param[2] + cen_param[4]*np.log((1+zmid[j])/1.3)
            B_L = 0.
            in_param[7] = 0.
            pyplot.plot(cen[:,0],fit_plm.p_L(10.**cen[:,0],mylambda,mass_param[j][0],
                                             mass_param[j][1],mass_param[j][2],mass_param[j][3],Mpiv,
                                             in_param,zmid[j],B_L),'b')
            
            if len(lm_param_alt)==0:
                lm_param_alt = lm_param
            if len(cen_param_alt)>0:
                in_param_alt[0] = sigma_lm
                in_param_alt[1] = cen_param_alt[0] #sigma_L
                in_param_alt[2] = cen_param_alt[1] #r
                in_param_alt[3] = lm_param_alt[3] + lm_param_alt[2]*np.log(1+0) #log(lm0)
                in_param_alt[4] = lm_param_alt[1]
                in_param_alt[5] = cen_param_alt[2] #ln L0
                in_param_alt[6] = cen_param_alt[3] #A_L
                in_param_alt[7] = cen_param_alt[4] #B_L
                B_L_alt = cen_param[4]
                pyplot.plot(cen[:,0],fit_plm.p_L(10.**cen[:,0],mylambda,mass_param[j][0],
                                                 mass_param[j][1],mass_param[j][2],mass_param[j][3],Mpiv,
                                                 in_param_alt,zmid[j],B_L_alt),'r')
            #Plotting the satellite part
            pyplot.semilogy(sat[:,0],sat[:,1],'k^')
            pyplot.errorbar(sat[:,0],sat[:,1],sat[:,2],fmt=None,ecolor='k')

            lm_param_temp = np.zeros(5)
            lm_param_temp[0:4] = lm_param[0:4]
            lm_param_temp[4] = lm_param[5]
            lm_param_temp[1] = A_lm*((1+zmid[j])/1.3)**lm_param[4]
            lm_param_temp[2] = 0.
            lm_param_temp[3] = lnlm0 + B_lm*np.log((1+zmid[j])/1.3)

            #print lm_param_temp
            sat_param_temp = np.zeros(6)
            sat_param_temp[0] = sat_param[0] + sat_param[1]*np.log((1+zmid[j])/1.3)#ln phi0
            sat_param_temp[1] = sat_param[2] + sat_param[3]*np.log((1+zmid[j])/1.3)#A_phi
            sat_param_temp[2] = sat_param[4] + sat_param[5]*np.log((1+zmid[j])/1.3)#lnLs0
            sat_param_temp[3] = sat_param[6] + sat_param[7]*np.log((1+zmid[j])/1.3)#As
            sat_param_temp[4] = 0. #Just ditching the Ls evolution into earlier parameter
            sat_param_temp[5] = sat_param[8] + sat_param[9]*np.log((1+zmid[j])/1.3)#alpha
            sat_fit = fit_psat.func_sat_convolved(sat[:,0],lm_min[i],lm_max[i],lm_param_temp,in_param[0],sat_param_temp,mass_param[j],zmid[j],mf_all[j][:,0],mf_all[j][:,1])
            pyplot.semilogy(sat[:,0],sat_fit,'b--')
            #print zmid[j],mylambda

            if len(sat_param_alt) > 0:
                myparam_alt = [sat_param_alt[0] + sat_param_alt[2]*np.log(1+zmid[j]), sat_param_alt[1],
                               sat_param_alt[3], sat_param_alt[4], sat_param_alt[5], 
                               sat_param_alt[6] + sat_param_alt[7]*zmid[j]]
                if phi_ev:
                    myparam_alt = [sat_param_alt[0] + sat_param_alt[6]*np.log(1+zmid[j]), 
                                   sat_param_alt[1], sat_param_alt[2],
                                   sat_param_alt[3], sat_param_alt[4], sat_param_alt[5]]
                #print sat_param
                #print myparam_alt
                sat_fit_a = fit_psat.func_sat_convolved(sat[:,0],lm_min[i],lm_max[i],lm_param_alt,in_param[0],myparam_alt,mass_param[j],zmid[j],mf_all[j][:,0],mf_all[j][:,1])
                pyplot.semilogy(sat[:,0],sat_fit_a,'r--')
                #print sat_fit_a/sat_fit
            
            Lcut = fit_psat.Lmin_eli(zmid[j],abs_solar=abs_solar)
            pyplot.plot([Lcut,Lcut],yrange,'k--')
            #print "Lcut: ",Lcut

            
    pyplot.savefig(outfile,orientation='landscape')
    pyplot.clf()
    #pyplot.close(1)
    return

#Plotting the CLFs again, but this time producing a ratio between the data and the model
def plot_clf_set_ratio(indir, indir_uber, lm_param, cen_param, sat_param, 
                 lm_min,lm_max,lm_med, mass_param, zmin, zmax, iband_err, mf_all, outfile,
                 abs_solar=4.67966,fix_alpha=True,sat_param_alt=[],yrange=[-1,2],phi_ev=False):
    zmid = (zmin+zmax)/2.
    nz = len(zmin)
    nlm = len(lm_min)

    in_param = np.zeros(8)
    Mpiv = lm_param[0]
    A_lm = lm_param[1]
    B_lm = lm_param[2]
    lnlm0 = lm_param[3]
    if len(lm_param) > 4:
        sigma_lm = lm_param[4]
    else:
        sigma_lm = 0.1842

    in_param[0] = sigma_lm
    in_param[1] = cen_param[0] #sigma_L
    in_param[2] = cen_param[1] #r
    in_param[3] = lnlm0 + B_lm*np.log(1+0) #log(lm0)
    in_param[4] = A_lm
    in_param[5] = cen_param[2] #ln L0
    in_param[6] = cen_param[3] #A_L
    in_param[7] = cen_param[4] #B_L
    B_L = cen_param[4]
    
    #Collect Lmin data
    zvals = np.arange(1000)*0.001
    Lmin_val = fit_psat.Lmin_eli(zvals,abs_solar=abs_solar)


    #Actual plotting section
    pyplot.figure(1,[11.,8.5])
    for i in range(nlm):
        for j in range(nz):
            #Read in the centrals only data
            cen = np.loadtxt(indir_uber+"clf_cen_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            cen_covar = np.loadtxt(indir_uber+"clf_cen_covar_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            #Read in the satellite data
            sat = np.loadtxt(indir+"clf_sat_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            sat_uber = np.loadtxt(indir_uber+"clf_sat_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            sat_covar = np.loadtxt(indir_uber+"clf_sat_covar_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            #Get characteristic lambda from limited satellite count
            mylambda = np.sum(sat[np.where(sat[:,1]>0)[0],1])*(sat[1,0]-sat[0,0])+1
            #Fix to input lm_med if we don't have a distinct input result
            if indir == indir_uber:
                mylambda = lm_med[i]
            
            #Set parameters for this lm_med, z combination
            in_param[3] = lnlm0 + B_lm*np.log(1+zmid[j])
            cen_fit = fit_plm.p_L(10.**cen[:,0],mylambda,mass_param[j][0],
                                             mass_param[j][1],mass_param[j][2],mass_param[j][3],Mpiv,
                                             in_param,zmid[j],B_L)
            clist = np.where(cen[:,1] > 0)[0]

            pyplot.subplot(nlm,nz,i*nz+j+1)
            pyplot.plot(cen[clist,0],cen[clist,1]/cen_fit[clist]-1,'ko')
            pyplot.errorbar(cen[clist,0],cen[clist,1]/cen_fit[clist]-1,np.sqrt(np.diag(cen_covar))[clist]/cen_fit[clist],ecolor='k',fmt=None)

            if i == nlm-1:
                pyplot.xlabel(r'$L_i$ [$L_\odot/h^2$]')
            if j == 0:
                pyplot.ylabel(r'$\Phi$ $[(d log L)^{-1}]$')
                pyplot.text(10.3,80,str(lm_min[i])[0:4]+r'$<\lambda<$'+str(lm_max[i])[0:4])
            if i == 0:
                pyplot.title(str(zmin[j])+'<z<'+str(zmax[j]))
            pyplot.xlim(9.,11.5)
            print yrange
            pyplot.ylim(yrange)

            #Plotting the satellite part
            slist = np.where(sat_uber[:,1] > 0)[0]
            if phi_ev:
                myparam = [sat_param[0] + sat_param[6]*np.log(1+zmid[j]), sat_param[1], sat_param[2],
                           sat_param[3], sat_param[4], sat_param[5]]
                sat_fit = fit_psat.func_sat_convolved(sat_uber[:,0],lm_min[i],lm_max[i],lm_param,in_param[0],myparam,mass_param[j],zmid[j],mf_all[j][:,0],mf_all[j][:,1])
            else:
                sat_fit = fit_psat.func_sat_convolved(sat_uber[:,0],lm_min[i],lm_max[i],lm_param,in_param[0],sat_param,mass_param[j],zmid[j],mf_all[j][:,0],mf_all[j][:,1])

            pyplot.plot(sat_uber[slist,0],sat_uber[slist,1]/sat_fit[slist]-1,'b^')
            pyplot.errorbar(sat_uber[slist,0],sat_uber[slist,1]/sat_fit[slist]-1,sat_uber[slist,2]/sat_fit[slist],fmt=None,ecolor='b')
            pyplot.plot(sat_uber[:,0],sat_uber[:,0]*0,'k--')
            
            if len(sat_param_alt) > 0:
                myparam_alt = [sat_param_alt[0] + sat_param_alt[2]*np.log(1+zmid[j]), sat_param_alt[1],
                               sat_param_alt[3], sat_param_alt[4], sat_param_alt[5], 
                               sat_param_alt[6] + sat_param_alt[7]*zmid[j]]
                sat_fit_a = fit_psat.func_sat_convolved(sat_uber[:,0],lm_min[i],lm_max[i],lm_param,in_param[0],myparam_alt,mass_param[j],zmid[j],mf_all[j][:,0],mf_all[j][:,1])
                pyplot.semilogy(sat_uber[:,0],sat_fit_a,'r--')
                #print sat_fit_a/sat_fit
    
    pyplot.savefig(outfile,orientation='landscape')
    pyplot.clf()
    return

#Plotting the CLFs again, but this time producing a ratio between the data and the model
#This plots with the non-schechter function correction
def plot_clf_set_ratio_nonsch(indir, indir_uber, lm_param, cen_param, sat_param, 
                              lm_min,lm_max,lm_med, mass_param, zmin, zmax, 
                              iband_err, mf_all, outfile,
                              abs_solar=4.67966,fix_alpha=True,
                              sat_param_alt=[],yrange=[-1,2],phi_ev=False,use_beta=False):
    zmid = (zmin+zmax)/2.
    nz = len(zmin)
    nlm = len(lm_min)

    in_param = np.zeros(8)
    Mpiv = lm_param[0]
    A_lm = lm_param[1]
    B_lm = lm_param[2]
    lnlm0 = lm_param[3]
    if len(lm_param) > 4:
        sigma_lm = lm_param[4]
    else:
        sigma_lm = 0.1842

    in_param[0] = sigma_lm
    in_param[1] = cen_param[0] #sigma_L
    in_param[2] = cen_param[1] #r
    in_param[3] = lnlm0 + B_lm*np.log(1+0) #log(lm0)
    in_param[4] = A_lm
    in_param[5] = cen_param[2] #ln L0
    in_param[6] = cen_param[3] #A_L
    in_param[7] = cen_param[4] #B_L
    B_L = cen_param[4]
    
    #Collect Lmin data
    zvals = np.arange(1000)*0.001
    Lmin_val = fit_psat.Lmin_eli(zvals,abs_solar=abs_solar)


    #Actual plotting section
    pyplot.figure(1,[11.,8.5])
    for i in range(nlm):
        for j in range(nz):
            #Read in the centrals only data
            cen = np.loadtxt(indir_uber+"clf_cen_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            cen_covar = np.loadtxt(indir_uber+"clf_cen_covar_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            #Read in the satellite data
            sat = np.loadtxt(indir+"clf_sat_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            sat_uber = np.loadtxt(indir_uber+"clf_sat_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            sat_covar = np.loadtxt(indir_uber+"clf_sat_covar_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            #Get characteristic lambda from limited satellite count
            mylambda = np.sum(sat[np.where(sat[:,1]>0)[0],1])*(sat[1,0]-sat[0,0])+1
            #Fix to input lm_med if we don't have a distinct input result
            if indir == indir_uber:
                mylambda = lm_med[i]
            
            #Set parameters for this lm_med, z combination
            in_param[3] = lnlm0 + B_lm*np.log(1+zmid[j])
            cen_fit = fit_plm.p_L(10.**cen[:,0],mylambda,mass_param[j][0],
                                             mass_param[j][1],mass_param[j][2],mass_param[j][3],Mpiv,
                                             in_param,zmid[j],B_L)
            clist = np.where(cen[:,1] > 0)[0]

            pyplot.subplot(nlm,nz,i*nz+j+1)
            pyplot.plot(cen[clist,0],cen[clist,1]/cen_fit[clist]-1,'ko')
            pyplot.errorbar(cen[clist,0],cen[clist,1]/cen_fit[clist]-1,np.sqrt(np.diag(cen_covar))[clist]/cen_fit[clist],ecolor='k',fmt=None)

            if i == nlm-1:
                pyplot.xlabel(r'$L_i$ [$L_\odot/h^2$]')
            if j == 0:
                pyplot.ylabel(r'$\Phi$ $[(d log L)^{-1}]$')
                pyplot.text(10.3,80,str(lm_min[i])[0:4]+r'$<\lambda<$'+str(lm_max[i])[0:4])
            if i == 0:
                pyplot.title(str(zmin[j])+'<z<'+str(zmax[j]))
            pyplot.xlim(9.,11.5)
            print yrange
            pyplot.ylim(yrange)

            #Plotting the satellite part
            slist = np.where(sat_uber[:,1] > 0)[0]
            if phi_ev:
                myparam = [sat_param[0] + sat_param[6]*np.log(1+zmid[j]), sat_param[1], sat_param[2],
                           sat_param[3], sat_param[4], sat_param[5]]
                sat_fit = fit_psat.func_sat_conv_sch_corr(sat_uber[:,0],lm_min[i],lm_max[i],lm_param,in_param[0],sat_param,mass_param[j],zmid[j],mf_all[j][:,0],mf_all[j][:,1])
            if use_beta:
                sat_fit = fit_psat.func_sat_conv_sch_beta(sat_uber[:,0],lm_min[i],lm_max[i],lm_param,in_param[0],sat_param,mass_param[j],zmid[j],mf_all[j][:,0],mf_all[j][:,1])
            if not phi_ev and not use_beta:
                sat_fit = fit_psat.func_sat_conv_sch_corr(sat_uber[:,0],lm_min[i],lm_max[i],lm_param,in_param[0],sat_param,mass_param[j],zmid[j],mf_all[j][:,0],mf_all[j][:,1])

            pyplot.plot(sat_uber[slist,0],sat_uber[slist,1]/sat_fit[slist]-1,'b^')
            pyplot.errorbar(sat_uber[slist,0],sat_uber[slist,1]/sat_fit[slist]-1,sat_uber[slist,2]/sat_fit[slist],fmt=None,ecolor='b')
            pyplot.plot(sat_uber[:,0],sat_uber[:,0]*0,'k--')
        
            if len(sat_param_alt) > 0:
                myparam_alt = [sat_param_alt[0] + sat_param_alt[2]*np.log(1+zmid[j]), sat_param_alt[1],
                               sat_param_alt[3], sat_param_alt[4], sat_param_alt[5], 
                               sat_param_alt[6] + sat_param_alt[7]*zmid[j]]
                sat_fit_a = fit_psat.func_sat_convolved(sat_uber[:,0],lm_min[i],lm_max[i],lm_param,in_param[0],myparam_alt,mass_param[j],zmid[j],mf_all[j][:,0],mf_all[j][:,1])
                pyplot.semilogy(sat_uber[:,0],sat_fit_a,'r--')
                #print sat_fit_a/sat_fit
    
    pyplot.savefig(outfile,orientation='landscape')
    pyplot.clf()
    return

#Plotting the CLFs again, but this time producing a ratio between the data and the model
#This plots with the non-schechter function that has a "beta" parameter
def plot_clf_set_ratio_beta(indir, indir_uber, lm_param, cen_param, sat_param, 
                            lm_min,lm_max,lm_med, mass_param, zmin, zmax, 
                            iband_err, mf_all, outfile,
                            abs_solar=4.67966,fix_alpha=True,
                            sat_param_alt=[],yrange=[-1,2]):
    zmid = (zmin+zmax)/2.
    nz = len(zmin)
    nlm = len(lm_min)

    in_param = np.zeros(8)
    Mpiv = lm_param[0]
    A_lm = lm_param[1]
    B_lm = lm_param[2]
    lnlm0 = lm_param[3]
    if len(lm_param) > 4:
        sigma_lm = lm_param[4]
    else:
        sigma_lm = 0.1842

    in_param[0] = sigma_lm
    in_param[1] = cen_param[0] #sigma_L
    in_param[2] = cen_param[1] #r
    in_param[3] = lnlm0 + B_lm*np.log(1+0) #log(lm0)
    in_param[4] = A_lm
    in_param[5] = cen_param[2] #ln L0
    in_param[6] = cen_param[3] #A_L
    in_param[7] = cen_param[4] #B_L
    B_L = cen_param[4]
    
    #Collect Lmin data
    zvals = np.arange(1000)*0.001
    Lmin_val = fit_psat.Lmin_eli(zvals,abs_solar=abs_solar)


    #Actual plotting section
    pyplot.figure(1,[11.,8.5])
    for i in range(nlm):
        for j in range(nz):
            #Read in the centrals only data
            cen = np.loadtxt(indir_uber+"clf_cen_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            cen_covar = np.loadtxt(indir_uber+"clf_cen_covar_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            #Read in the satellite data
            sat = np.loadtxt(indir+"clf_sat_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            sat_uber = np.loadtxt(indir_uber+"clf_sat_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            sat_covar = np.loadtxt(indir_uber+"clf_sat_covar_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            #Get characteristic lambda from limited satellite count
            mylambda = np.sum(sat[np.where(sat[:,1]>0)[0],1])*(sat[1,0]-sat[0,0])+1
            #Fix to input lm_med if we don't have a distinct input result
            if indir == indir_uber:
                mylambda = lm_med[i]
            
            #Set parameters for this lm_med, z combination
            in_param[3] = lnlm0 + B_lm*np.log(1+zmid[j])
            cen_fit = fit_plm.p_L(10.**cen[:,0],mylambda,mass_param[j][0],
                                             mass_param[j][1],mass_param[j][2],mass_param[j][3],Mpiv,
                                             in_param,zmid[j],B_L)
            clist = np.where(cen[:,1] > 0)[0]

            pyplot.subplot(nlm,nz,i*nz+j+1)
            pyplot.plot(cen[clist,0],cen[clist,1]/cen_fit[clist]-1,'ko')
            pyplot.errorbar(cen[clist,0],cen[clist,1]/cen_fit[clist]-1,np.sqrt(np.diag(cen_covar))[clist]/cen_fit[clist],ecolor='k',fmt=None)

            if i == nlm-1:
                pyplot.xlabel(r'$L_i$ [$L_\odot/h^2$]')
            if j == 0:
                pyplot.ylabel(r'$\Phi$ $[(d log L)^{-1}]$')
                pyplot.text(10.3,80,str(lm_min[i])[0:4]+r'$<\lambda<$'+str(lm_max[i])[0:4])
            if i == 0:
                pyplot.title(str(zmin[j])+'<z<'+str(zmax[j]))
            pyplot.xlim(9.,11.5)
            print yrange
            pyplot.ylim(yrange)

            #Plotting the satellite part
            slist = np.where(sat_uber[:,1] > 0)[0]
            sat_fit = fit_psat.func_sat_conv_sch_beta(sat_uber[:,0],lm_min[i],lm_max[i],lm_param,in_param[0],sat_param,mass_param[j],zmid[j],mf_all[j][:,0],mf_all[j][:,1])
            
            pyplot.plot(sat_uber[slist,0],sat_uber[slist,1]/sat_fit[slist]-1,'b^')
            pyplot.errorbar(sat_uber[slist,0],sat_uber[slist,1]/sat_fit[slist]-1,sat_uber[slist,2]/sat_fit[slist],fmt=None,ecolor='b')
            pyplot.plot(sat_uber[:,0],sat_uber[:,0]*0,'k--')
        
            if len(sat_param_alt) > 0:
                sat_fit_a = fit_psat.func_sat_convolved(sat_uber[:,0],lm_min[i],lm_max[i],lm_param,in_param[0],sat_param_alt,mass_param[j],zmid[j],mf_all[j][:,0],mf_all[j][:,1])
                pyplot.semilogy(sat_uber[:,0],sat_fit_a,'r--')
                #print sat_fit_a/sat_fit
    
    pyplot.savefig(outfile,orientation='landscape')
    pyplot.clf()
    return

#Plotting the CLFs again, but this time producing a ratio between the data and the model
def plot_clf_brightest(indir,lm_param,sat_param, 
                       lm_min,lm_max,lm_med, mass_param, zmin, zmax, mf_all, outfile,
                       yrange=[0.01,10],phi_ev=False,chain=[],
                       cen_param=[]):
    zmid = (zmin+zmax)/2.
    nz = len(zmin)
    nlm = len(lm_min)

    in_param = np.zeros(8)
    Mpiv = lm_param[0]
    A_lm = lm_param[1]
    B_lm = lm_param[2]
    lnlm0 = lm_param[3]
    if len(lm_param) > 4:
        sigma_lm = lm_param[4]
    else:
        sigma_lm = 0.1842
    if len(cen_param) > 0:
        in_param[1] = cen_param[0] #sigma_L
        in_param[2] = cen_param[1] #r
        in_param[3] = lnlm0 + B_lm*np.log(1.0) #log(lm0)
        in_param[4] = A_lm
        in_param[5] = cen_param[2] #ln L0
        in_param[6] = cen_param[3] #A_L
        in_param[7] = cen_param[4] #B_L
        B_L = cen_param[4]

    #Actual plotting section
    pyplot.figure(1,[11.,8.5])
    for i in range(nlm):
        for j in range(nz):
            #Read in the brightest satellite data
            bsat = np.loadtxt(indir+"clf_sat_bright_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            bsat_cov = np.loadtxt(indir+"clf_sat_bright_covar_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")

            mylambda = lm_med[i]
            
            #Set parameters for this lm_med, z combination
            pyplot.subplot(nlm,nz,i*nz+j+1)
            pyplot.semilogy(bsat[:,0],bsat[:,1],'ko')
            pyplot.errorbar(bsat[:,0],bsat[:,1],bsat[:,2],fmt=None,ecolor='k')

            if i == nlm-1:
                pyplot.xlabel(r'$L_i$ [$L_\odot/h^2$]')
            if j == 0:
                pyplot.ylabel(r'$\Phi$ $[(d log L)^{-1}]$')
                pyplot.text(9.3,5,str(lm_min[i])[0:4]+r'$<\lambda<$'+str(lm_max[i])[0:4])
            if i == 0:
                pyplot.title(str(zmin[j])+'<z<'+str(zmax[j]))
            pyplot.xlim(9.,11.5)
            pyplot.ylim(yrange)

            #Plotting the fit
            mass = (mylambda/np.exp(lnlm0))**(1./A_lm)/(1+zmid[j])**(B_lm/A_lm)*Mpiv
            Lst = (sat_param[2]+sat_param[3]*np.log(mass/Mpiv)+sat_param[4]*np.log(1+zmid[j]))/np.log(10.)
            phist = np.exp( sat_param[0] )*(mass/Mpiv)**sat_param[1]
            if phi_ev:
                phist = np.exp( sat_param[0] )*(mass/Mpiv)**sat_param[1]*(1+zmid[j])**sat_param[6]
            alpha = sat_param[5]

            #print i, zmid[i], np.log10(mass), Lst, phist, alpha
            lumbins, bsat_fit = get_pbcg.brightest_satellite_distr(Lst,phist,alpha)
            pyplot.plot(lumbins,bsat_fit,'b-')
            blist= np.where(bsat[:,1] > 1e-2)[0]
            bsat_cov = bsat_cov[blist,:]
            bsat_cov = bsat_cov[:,blist]
            bvals =  np.interp(bsat[blist,0],lumbins,bsat_fit)

            #Plot centrals distribution
            if len(cen_param) > 0:
                cen_fit = fit_plm.p_L(10.**lumbins,mylambda,mass_param[j][0],
                                      mass_param[j][1],mass_param[j][2],mass_param[j][3],
                                      Mpiv,in_param,zmin[j],B_L)
                pyplot.plot(lumbins,cen_fit,'r')
                ptest = 0.
                ptest_obs = 0.
                dL = lumbins[1]-lumbins[0]
                for ii in range(len(lumbins)):
                    for jj in range(len(lumbins)):
                        if ii < jj:
                            ptest = ptest + cen_fit[ii]*bsat_fit[jj]*dL*dL
                for ii in range(len(lumbins)):
                    for jj in range(len(bsat[:,0])):
                        if lumbins[ii] < bsat[jj,0]:
                            ptest_obs = ptest_obs + cen_fit[ii]*bsat[jj,1]*dL*(bsat[1,0]-bsat[0,0])
                print >> sys.stderr, "BSAT>CEN: ",ptest,ptest_obs

            #Plot an estimated range for the model fits
            if len(chain) > 0:
                for k in range(len(chain)):
                    myparam = chain[k]
                    mass = (mylambda/np.exp(lnlm0))**(1./A_lm)/(1+zmid[j])**(B_lm/A_lm)*Mpiv
                    Lst = (myparam[2]+myparam[3]*np.log(mass/Mpiv)+myparam[4]*np.log(1+zmid[j]))/np.log(10.)
                    phist = np.exp( myparam[0] )*(mass/Mpiv)**myparam[1]
                    if phi_ev:
                        phist = np.exp( myparam[0] )*(mass/Mpiv)**myparam[1]*(1+zmid[j])**myparam[6]
                    alpha = myparam[5]
                    lumbins, bsat_fit_t = get_pbcg.brightest_satellite_distr(Lst,phist,alpha)
                    pyplot.plot(lumbins,bsat_fit_t,'r-',alpha=0.2)
                    #print bsat_fit_t-bsat_fit
            #Replot
            pyplot.semilogy(bsat[:,0],bsat[:,1],'ko')
            pyplot.errorbar(bsat[:,0],bsat[:,1],bsat[:,2],fmt=None,ecolor='k')
            pyplot.plot(lumbins,bsat_fit,'b-')
                    
            delta = bsat[blist,1] -  np.exp(np.interp(bsat[blist,0],lumbins,np.log(bsat_fit)))
            print "BSAT", zmin[j] , len(blist), np.sum( delta**2/
                                                bsat[blist,2]**2 ), np.dot( delta,np.dot(np.linalg.inv(bsat_cov),delta) )

    pyplot.savefig(outfile,orientation='landscape')
    pyplot.clf()
    return

#Brightest satellite version, to work with new parameterization
def plot_clf_brightest_new(indir,lm_param,sat_param, 
                           lm_min,lm_max,lm_med, mass_param, zmin, zmax, mf_all, outfile,
                           yrange=[0.03,10],cen_param=[]):
    zmid = (zmin+zmax)/2.
    nz = len(zmin)
    nlm = len(lm_min)
    in_param = np.zeros(8)
    Mpiv = lm_param[0]
    A_lm = lm_param[1]
    B_lm = lm_param[2]
    lnlm0 = lm_param[3]
    if len(lm_param) > 5:
        sigma_lm = lm_param[5]
    else:
        sigma_lm = 0.1842
    if len(cen_param) > 0:
        in_param[1] = cen_param[0] #sigma_L
        in_param[2] = cen_param[1] #r
        in_param[3] = lnlm0 + B_lm*np.log(1.0) #log(lm0)
        in_param[4] = A_lm
        in_param[5] = cen_param[2] #ln L0
        in_param[6] = cen_param[3] #A_L
        in_param[7] = 0. #B_L
        B_L = 0.


    #Actual plotting section
    pyplot.figure(1,[11,8.5])
    for i in range(nlm):
        for j in range(nz):

    #Handling issues with redshift dependence -- translating versions
            lm_param_temp = np.zeros(5)
            lm_param_temp[0:4] = lm_param[0:4]
            lm_param_temp[4] = lm_param[5]
            lm_param_temp[1] = A_lm*((1+zmid[j])/1.3)**lm_param[4]
            lm_param_temp[2] = 0.
            lm_param_temp[3] = lnlm0 + B_lm*np.log((1+zmid[j])/1.3)
            sat_param_temp = np.zeros(6)
            sat_param_temp[0] = sat_param[0] + sat_param[1]*np.log((1+zmid[j])/1.3)#ln phi0
            sat_param_temp[1] = sat_param[2] + sat_param[3]*np.log((1+zmid[j])/1.3)#A_phi
            sat_param_temp[2] = sat_param[4] + sat_param[5]*np.log((1+zmid[j])/1.3)#lnLs0
            sat_param_temp[3] = sat_param[6] + sat_param[7]*np.log((1+zmid[j])/1.3)#As
            sat_param_temp[4] = 0. #Just ditching the Ls evolution into earlier parameter
            sat_param_temp[5] = sat_param[8] + sat_param[9]*np.log((1+zmid[j])/1.3)#alpha

            #Read in the brightest satellite data
            bsat = np.loadtxt(indir+"clf_sat_bright_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            mylambda = lm_med[i]

            #Set up parameters for this lm_med, z combo
            pyplot.subplot(nlm,nz,i*nz+j+1)
            pyplot.semilogy(bsat[:,0],bsat[:,1],'ko')
            pyplot.errorbar(bsat[:,0],bsat[:,1],bsat[:,2],fmt=None,ecolor='k')
            
            if i == nlm-1:
                pyplot.xlabel(r'$L_i$ [$L_\odot/h^2$]')
            if j == 0:
                pyplot.ylabel(r'$\Phi$ $[(d log L)^{-1}]$')
                pyplot.text(9.8,5,str(lm_min[i])[0:4]+r'$<\lambda<$'+str(lm_max[i])[0:4])
            if i == 0:
                pyplot.title(str(zmin[j])+'<z<'+str(zmax[j]))
            pyplot.xlim(9.5,11.5)
            pyplot.ylim(yrange)

            #Plotting the fit
            A_lm_bin = A_lm*((1+zmid[j])/1.3)**lm_param[4]
            mass = (mylambda/np.exp(lnlm0))**(1./A_lm_bin)/(1+zmid[j])**(B_lm/A_lm_bin)*Mpiv
            
            phist = np.exp( sat_param[0] )*(mass/Mpiv)**(sat_param[2]+sat_param[3]*np.log((1+zmid[j])/1.3))*((1+zmid[j])/1.3)**sat_param[1]
            Lst = (sat_param[4]+(sat_param[6]+sat_param[7]*np.log((1+zmid[j])/1.3))*np.log(mass/Mpiv)+sat_param[5]*np.log((1+zmid[j])/1.3))/np.log(10.)
            alpha = sat_param[8]+sat_param[9]*np.log((1+zmid[j])/1.3)
            
            print mylambda, mass, Lst, phist, alpha

            lumbins, bsat_fit = get_pbcg.brightest_satellite_distr(Lst, phist, alpha)
            pyplot.semilogy(lumbins, bsat_fit,'b-')
            if len(cen_param) > 0:
                in_param[3] = lnlm0 + B_lm*np.log((1+zmid[j])/1.3)
                in_param[5] = cen_param[2] + cen_param[4]*np.log((1+zmid[j])/1.3)
                cen_fit = fit_plm.p_L(10.**lumbins,mylambda,mass_param[j][0],
                                      mass_param[j][1],mass_param[j][2],mass_param[j][3],
                                      Mpiv,in_param,zmin[j],B_L)
                pyplot.plot(lumbins,cen_fit,'r--')       
            
    pyplot.savefig(outfile,orientation='landscape')
    pyplot.clf()

    return

#Plotting the CLFs again, but this time producing a ratio between the data and the model
def plot_clf_bsat_ratio(indir,lm_param,sat_param, 
                       lm_min,lm_max,lm_med, mass_param, zmin, zmax, mf_all, outfile,
                       yrange=[-1,1],phi_ev=False):
    zmid = (zmin+zmax)/2.
    nz = len(zmin)
    nlm = len(lm_min)

    in_param = np.zeros(8)
    Mpiv = lm_param[0]
    A_lm = lm_param[1]
    B_lm = lm_param[2]
    lnlm0 = lm_param[3]
    if len(lm_param) > 4:
        sigma_lm = lm_param[4]
    else:
        sigma_lm = 0.1842

    #Actual plotting section
    pyplot.figure(1,[11.,8.5])
    for i in range(nlm):
        for j in range(nz):
            #Read in the brightest satellite data
            bsat = np.loadtxt(indir+"clf_sat_bright_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            bsat_cov = np.loadtxt(indir+"clf_sat_bright_covar_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")

            mylambda = lm_med[i]
            
            #Set parameters for this lm_med, z combination
            pyplot.subplot(nlm,nz,i*nz+j+1)
            #pyplot.semilogy(bsat[:,0],bsat[:,1],'ko')
            
            if i == nlm-1:
                pyplot.xlabel(r'$L_i$ [$L_\odot/h^2$]')
            if j == 0:
                pyplot.ylabel(r'$\Phi$ $[(d log L)^{-1}]$')
                pyplot.text(9.2,0.7,str(lm_min[i])[0:4]+r'$<\lambda<$'+str(lm_max[i])[0:4])
            if i == 0:
                pyplot.title(str(zmin[j])+'<z<'+str(zmax[j]))
            pyplot.xlim(9.,11.5)
            pyplot.ylim(yrange)

            #Plotting the fit
            mass = (mylambda/np.exp(lnlm0))**(1./A_lm)/(1+zmid[j])**(B_lm/A_lm)*Mpiv
            Lst = (sat_param[2]+sat_param[3]*np.log(mass/Mpiv)+sat_param[4]*np.log(1+zmid[j]))/np.log(10.)
            phist = np.exp( sat_param[0] )*(mass/Mpiv)**sat_param[1]
            if phi_ev:
                phist = np.exp( sat_param[0] )*(mass/Mpiv)**sat_param[1]*(1+zmid[j])**sat_param[6]
            alpha = sat_param[5]

            #print i, zmid[i], np.log10(mass), Lst, phist, alpha
            lumbins, bsat_fit = get_pbcg.brightest_satellite_distr(Lst,phist,alpha)
            blist= np.where(bsat[:,1] > 0)[0]
            bsat_cov = bsat_cov[blist,:]
            bsat_cov = bsat_cov[:,blist]
            bvals =  np.interp(bsat[blist,0],lumbins,bsat_fit)
            
            pyplot.plot(lumbins,0*lumbins,'b--')
            pyplot.plot(bsat[blist,0],bsat[blist,1]/bvals-1,'ko')
            pyplot.errorbar(bsat[blist,0],bsat[blist,1]/bvals-1,bsat[blist,2]/bvals,fmt=None,ecolor='k')

    pyplot.savefig(outfile,orientation='landscape')
    pyplot.clf()
    return


#Plot for comparing two different sets of data, does NOT include any fits
def plot_clf_set_data(indir, indir_alt, lm_min,lm_max,
                      zmin, zmax, outfile,lm_min_alt=[],lm_max_alt=[]):
    zmid = (zmin+zmax)/2.
    nz = len(zmin)
    nlm = len(lm_min)

    if (len(lm_min_alt)==0) | (len(lm_max_alt)==0):
        lm_min_alt = lm_min
        lm_max_alt = lm_max

    if len(np.array(lm_min_alt).flatten()) > 3:
        #Note that we're actually using lm_min_alt specified for each redshift
        specific = True
    else:
        specific = False

    #Actual plotting section
    pyplot.figure(1,[11.,8.5])
    for i in range(nlm):
        for j in range(nz):
            #Read in the centrals only data
            cen = np.loadtxt(indir+"clf_cen_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            if specific:
                cen_alt = np.loadtxt(indir_alt+"clf_cen_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+lm_min_alt[j][i]+"_"+lm_max_alt[j][i]+".dat")
            else:
                cen_alt = np.loadtxt(indir_alt+"clf_cen_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min_alt[i])[0:5]+"_"+str(lm_max_alt[i])[0:5]+".dat")
            #Read in the satellite data
            sat = np.loadtxt(indir+"clf_sat_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            if specific:
                sat_alt = np.loadtxt(indir_alt+"clf_sat_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+lm_min_alt[j][i]+"_"+lm_max_alt[j][i]+".dat")
            else:
                sat_alt = np.loadtxt(indir_alt+"clf_sat_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min_alt[i])[0:5]+"_"+str(lm_max_alt[i])[0:5]+".dat")

            pyplot.subplot(nlm,nz,i*nz+j+1)
            pyplot.semilogy(cen[:,0],cen[:,1],'ko')
            pyplot.errorbar(cen[:,0],cen[:,1],cen[:,2],ecolor='k',fmt=None)
            pyplot.semilogy(cen_alt[:,0],cen_alt[:,1],'ro')
            #pyplot.errorbar(cen_alt[:,0],cen_alt[:,1],cen_alt[:,2],ecolor='r',fmt=None)

            if i == nlm-1:
                pyplot.xlabel(r'$L_i$ [$L_\odot/h^2$]')
            if j == 0:
                pyplot.ylabel(r'$\Phi$ $[(d log L)^{-1}]$')
                pyplot.text(10.3,80,str(lm_min[i])[0:4]+r'$<\lambda<$'+str(lm_max[i])[0:4])
            if i == 0:
                pyplot.title(str(zmin[j])+'<z<'+str(zmax[j]))
            pyplot.xlim(9.,11.5)
            pyplot.ylim(0.011,3e2)

            #Plotting the satellite part
            pyplot.semilogy(sat[:,0],sat[:,1],'k^')
            pyplot.errorbar(sat[:,0],sat[:,1],sat[:,2],fmt=None,ecolor='k')
            pyplot.semilogy(sat_alt[:,0],sat_alt[:,1],'r^')
            #pyplot.errorbar(sat_alt[:,0],sat_alt[:,1],sat_alt[:,2],fmt=None,ecolor='k')
            
    
    pyplot.savefig(outfile,orientation='landscape')
    pyplot.clf()
    return


#Plot for comparing two different sets of data, does NOT include any fits
def plot_clf_set_data(indir, indir_alt, lm_min,lm_max,
                      zmin, zmax, outfile,lm_min_alt=[],lm_max_alt=[]):
    zmid = (zmin+zmax)/2.
    nz = len(zmin)
    nlm = len(lm_min)

    if (len(lm_min_alt)==0) | (len(lm_max_alt)==0):
        lm_min_alt = lm_min
        lm_max_alt = lm_max

    if len(np.array(lm_min_alt).flatten()) > 3:
        #Note that we're actually using lm_min_alt specified for each redshift
        specific = True
    else:
        specific = False

    #Actual plotting section
    pyplot.figure(1,[11.,8.5])
    for i in range(nlm):
        for j in range(nz):
            #Read in the centrals only data
            cen = np.loadtxt(indir+"clf_cen_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            if specific:
                cen_alt = np.loadtxt(indir_alt+"clf_cen_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+lm_min_alt[j][i]+"_"+lm_max_alt[j][i]+".dat")
            else:
                cen_alt = np.loadtxt(indir_alt+"clf_cen_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min_alt[i])[0:5]+"_"+str(lm_max_alt[i])[0:5]+".dat")
            #Read in the satellite data
            sat = np.loadtxt(indir+"clf_sat_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min[i])[0:5]+"_"+str(lm_max[i])[0:5]+".dat")
            if specific:
                sat_alt = np.loadtxt(indir_alt+"clf_sat_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+lm_min_alt[j][i]+"_"+lm_max_alt[j][i]+".dat")
            else:
                sat_alt = np.loadtxt(indir_alt+"clf_sat_z_"+str(zmin[j])+"_"+str(zmax[j])+"_lm_"+str(lm_min_alt[i])[0:5]+"_"+str(lm_max_alt[i])[0:5]+".dat")

            pyplot.subplot(nlm,nz,i*nz+j+1)
            pyplot.semilogy(cen[:,0],cen[:,1],'ko')
            pyplot.errorbar(cen[:,0],cen[:,1],cen[:,2],ecolor='k',fmt=None)
            pyplot.semilogy(cen_alt[:,0],cen_alt[:,1],'ro')
            #pyplot.errorbar(cen_alt[:,0],cen_alt[:,1],cen_alt[:,2],ecolor='r',fmt=None)

            if i == nlm-1:
                pyplot.xlabel(r'$L_i$ [$L_\odot/h^2$]')
            if j == 0:
                pyplot.ylabel(r'$\Phi$ $[(d log L)^{-1}]$')
                pyplot.text(10.3,80,str(lm_min[i])[0:4]+r'$<\lambda<$'+str(lm_max[i])[0:4])
            if i == 0:
                pyplot.title(str(zmin[j])+'<z<'+str(zmax[j]))
            pyplot.xlim(9.,11.5)
            pyplot.ylim(0.011,3e2)

            #Plotting the satellite part
            pyplot.semilogy(sat[:,0],sat[:,1],'k^')
            pyplot.errorbar(sat[:,0],sat[:,1],sat[:,2],fmt=None,ecolor='k')
            pyplot.semilogy(sat_alt[:,0],sat_alt[:,1],'r^')
            #pyplot.errorbar(sat_alt[:,0],sat_alt[:,1],sat_alt[:,2],fmt=None,ecolor='k')
            
    
    pyplot.savefig(outfile,orientation='landscape')
    pyplot.clf()
    return



#Plot for demoing the mass, redshift dependence of the various parameters
def param_plot_set(cen_param,sat_param,zmid,outfile,
                   zvals=[],Lmin=[],
                   abs_solar=4.67966,Mpiv=1.645e14):
    '''
    Note that default Mpiv is set for Msolar/h (h = 0.6704); divide by h to get Msolar
    '''
    #Get Lmin if we don't have it already
    if len(zvals)==0:
        zvals = np.arange(1000)*0.001
        Lmin_val = fit_psat.Lmin_eli(zvals,abs_solar=abs_solar)

        
    #Set up the plots
    #pyplot.figure(1,figsize=[8.5,11])
    pyplot.figure(1,figsize=[11,8.5])

    mvals = 13.8 + np.array(range(150))*0.01

    #Plots Lcen, L*
    #pyplot.subplot(3,1,1)
    zlabel = []
    colors = ['k','b','g','r','c','m']
    for i in range(len(zmid)):
        zlabel.append('z='+str(zmid[i]))
        pyplot.plot(mvals, (cen_param[2] + cen_param[3]*np.log(10.**mvals/Mpiv) + cen_param[4]*np.log((1+zmid[i])/1.3))/np.log(10.),colors[i],label=zlabel[i])
        
    for i in range(len(zmid)):        
        pyplot.plot(mvals, (sat_param[4] + (sat_param[6]+sat_param[7]*np.log((1+zmid[i])/1.3))*np.log(10.**mvals/Mpiv) + sat_param[5]*np.log((1+zmid[i])/1.3))/np.log(10.) ,colors[i]+'--')
        

    #pyplot.legend(loc='upper left')
    pyplot.xlabel(r'log(M) $[M_\odot/h]$',fontsize=30)
    pyplot.ylabel(r'$L_{cen}$ or $L_{sat}$ [$L_\odot/h^2$]',fontsize=30)
    pyplot.tick_params(axis='both',which='major',labelsize=20)
    pyplot.ylim([10.01, 11.4])
    pyplot.xlim([13.5,15.5])

    #Plots phi*

    #Plots alpha
    pyplot.tight_layout()

    pyplot.savefig(outfile,orientation='landscape')
    pyplot.clf()
    pyplot.close(1)

    return

#Assorted functions for making contour plots

#Function for making a histogram
def make_2d_hist(xvals,yvals,nbins,xmin,xmax,ymin,ymax):
    hist = np.zeros([nbins, nbins]).astype(long)

    dx = (xmax-xmin)/nbins
    dy = (ymax-ymin)/nbins
    for i in range(nbins):
        for j in range(nbins):
            hist[j,i] = len(np.where( (xvals >= xmin+dx*i) & (xvals < xmin+dx*(i+1)) & (yvals >= ymin+dy*j) & (yvals < ymin+dy*(j+1) ) )[0])
    return hist

#Function for getting desired contour levels
def get_levels(f_levels,hist):
    levels = 0*np.array(f_levels)
    hmax = np.max(hist)
    oldf=0.
    for i in range(hmax):
        f = np.sum(hist[np.where(hist > i)])/float(np.sum(hist))
        flist = np.where( (f < f_levels) & ( oldf > f_levels ) )[0]
        oldf = f
        if len(flist) == 0:
            continue
        levels[flist] = np.repeat(i,len(flist))
    print("levels:{0}".format(levels))
    return np.sort(levels)

#Make a single contour from an x-y list
def plot_single_contour(xvals,yvals,f_levels,nbins,my_xrange,my_yrange,
                        nox=False,noy=False,x2=[],y2=[],rotation=19):
    '''
    Input: xvals, yvals, f_levels, xmin,xmax,ymin,ymax,xrange,yrange,nbins
    '''

    #Make the 2D histogram    
    xmin = np.min(xvals)
    xmax = np.max(xvals)
    ymin = np.min(yvals)
    ymax = np.max(yvals)
    dx = (xmax-xmin)/nbins
    dy = (ymax-ymin)/nbins
    hist = make_2d_hist(xvals,yvals,nbins,xmin,xmax,ymin,ymax)
            
    #Find the levels that we want to use
    levels = get_levels(f_levels,hist)

    x_input = xmin+np.array(range(nbins))*dx+dx/2.
    y_input = ymin+np.array(range(nbins))*dy+dy/2.

    #Now that we know what levels to use, do the plot
    pyplot.contour(x_input, y_input, hist, levels, colors='k')
    pyplot.xlim(my_xrange)
    pyplot.ylim(my_yrange)

    #Remove axis labels if requested, and do some clean-up otherwise
    locs = np.array([my_yrange[0]+(my_yrange[1]-my_yrange[0])/3., my_yrange[0]+2.*(my_yrange[1]-my_yrange[0])/3.])
    if noy:
        pyplot.yticks(locs,np.repeat('',len(locs)))
    else:
        labels = [str(locs[0])[0:6],str(locs[1])[0:6]]
        pyplot.yticks(locs,labels)
    #x-axis part
    locs = np.array([my_xrange[0]+(my_xrange[1]-my_xrange[0])/3., my_xrange[0]+2.*(my_xrange[1]-my_xrange[0])/3.])
    if nox:
        pyplot.xticks(locs,np.repeat('',len(locs)))
    else:
        labels = [str(locs[0])[0:6],str(locs[1])[0:6]]
        pyplot.xticks(locs,labels,rotation=rotation)

    #If the second chain exists, make a second contour set and overplot it
    if len(x2) > 0:
        #Make the 2D histogram    
        xmin = np.min(x2)
        xmax = np.max(x2)
        ymin = np.min(y2)
        ymax = np.max(y2)
        dx = (xmax-xmin)/nbins
        dy = (ymax-ymin)/nbins
        hist = make_2d_hist(x2,y2,nbins,xmin,xmax,ymin,ymax)
            
        #Find the levels that we want to use
        levels = get_levels(f_levels,hist)
        x_input = xmin+np.array(range(nbins))*dx+dx/2.
        y_input = ymin+np.array(range(nbins))*dy+dy/2.
        pyplot.contour(x_input, y_input, hist, levels, colors='b')

    return

#Script for plotting a full set of contours and associated histograms from 
#a single input chain; requires input labels as well
def plot_set_contour(chain,cutoff,labels,pmin,pmax,nbins=20,chain_alt=[],
                     rotation=19):
    '''
    Inputs: chain, cutoff, labels, pmin, pmax
    cutoff is the cut in the chain -- minimum position in the chain to use
    '''

    nparam = len(chain[0])-1

    #Set up our figure
    pyplot.figure(1,[11,8.5])
    pyplot.subplots_adjust(wspace=0,hspace=0)
    for i in range(nparam):
        for j in range(nparam):
            if j > i:
                #Leave upper right half of plot blank
                continue
            pyplot.subplot(nparam,nparam,i*nparam+j+1)
            if i==j:
                #This is where we'll want to make a histogram
                hist = np.zeros(nbins)
                xmin = np.min(chain[cutoff:,i])
                xmax = np.max(chain[cutoff:,i])
                dx = (xmax-xmin)/nbins
                for k in range(nbins):
                    hist[k] = len(np.where( (chain[cutoff:,i] >= xmin+dx*k) & 
                                            (chain[cutoff:,i] < xmin+dx*(k+1)) )[0])
                hist = hist/np.sum(hist)/dx
                pyplot.plot(xmin+np.array(range(nbins))*dx+dx/2.,hist,'k')
                pyplot.xlim(pmin[i],pmax[i])
                pyplot.yticks([])
                locs = [pmin[i]+(pmax[i]-pmin[i])/3.,pmin[i]+2*(pmax[i]-pmin[i])/3.]
                myx_labels = [str(locs[0])[0:6],str(locs[1])[0:6]]
                pyplot.xticks(locs,myx_labels,rotation=rotation)

                #Make the second histogram if chain_alt exists
                if len(chain_alt) > 0:
                    hist = np.zeros(nbins)
                    xmin = np.min(chain_alt[cutoff:,i])
                    xmax = np.max(chain_alt[cutoff:,i])
                    dx = (xmax-xmin)/nbins
                    for k in range(nbins):
                        hist[k] = len(np.where( (chain_alt[cutoff:,i] >= xmin+dx*k) & 
                                            (chain_alt[cutoff:,i] < xmin+dx*(k+1)) )[0])
                    hist = hist/np.sum(hist)/dx
                    pyplot.plot(xmin+np.array(range(nbins))*dx+dx/2.,hist,'b')
            else:
                #And here we make a contour plot
                if (j == 0) & (i>0):
                    noy = False
                else:
                    noy = True
                if i == nparam-1:
                    nox = False
                else:
                    nox = True
                #Switch between cases of one and two input chains
                if len(chain_alt) == 0:
                    plot_single_contour(chain[cutoff:,j],chain[cutoff:,i],
                                        np.array([0.68,0.95]),nbins,
                                        [pmin[j],pmax[j]],[pmin[i],pmax[i]],noy=noy,nox=nox,rotation=rotation)
                else:
                    plot_single_contour(chain[cutoff:,j],chain[cutoff:,i],
                                        np.array([0.68,0.95]),nbins,
                                        [pmin[j],pmax[j]],[pmin[i],pmax[i]],
                                        noy=noy,nox=nox,
                                        x2=chain_alt[cutoff:,j],y2=chain_alt[cutoff:,i],rotation=rotation)
            if i == nparam-1:
                pyplot.xlabel(labels[j])
            if (j == 0) & (i > 0):
                pyplot.ylabel(labels[i])
    return

#Make several contousr from input x-y lists
def plot_some_contours(xvals,yvals,f_levels,nbins,my_xrange,my_yrange,
                       nox=False,noy=False,x2=[],y2=[],colors=[],rotation=25,
                       linetypes=[], fill_contours=[]):
    '''
    Input: xvals, yvals, f_levels, xmin,xmax,ymin,ymax,xrange,yrange,nbins
    '''
    if len(colors) == 0:
        colors = ['k','b','g','r','m','c']
    if len(linetypes) == 0:
        linetypes = ['-','-','-','-','-']
    if len(fill_contours) == 0:
        fill_contours = [True, True, False, False, False]
        fill_colors = [('#000000', '#000001', '#000002'),
                       ('#0000fd','#0000fe', '#0000ff')]

    #Get the number of contours
    nchains = len(xvals)

    for i in range(nchains):
        #Make the 2D histogram
        xmin = np.min(xvals[i])
        xmax = np.max(xvals[i])
        ymin = np.min(yvals[i])
        ymax = np.max(yvals[i])
        dx = (xmax-xmin)/nbins
        dy = (ymax-ymin)/nbins
        hist = make_2d_hist(xvals[i],yvals[i],nbins,xmin,xmax,ymin,ymax)
            
        #Find the levels that we want to use
        levels = get_levels(f_levels,hist)

        x_input = xmin+np.array(range(nbins))*dx+dx/2.
        y_input = ymin+np.array(range(nbins))*dy+dy/2.

        #Now that we know what levels to use, do the plot
        my_linewidth = 1
        marker = None
        if i == 2:
            marker = 'o'
        if fill_contours[i]:
            pyplot.contourf(x_input, y_input, hist, levels, colors=fill_colors[i], alpha=0.5,
                            extend='max')
        pyplot.contour(x_input, y_input, hist, levels, colors=colors[i],linestyles=linetypes[i],
                       linewidths=my_linewidth, markers=marker, ms=5)
    
    pyplot.xlim(my_xrange)
    pyplot.ylim(my_yrange)

    #Remove axis labels if requested, and do some clean-up otherwise
    locs = get_tick_positions(my_yrange[0], my_yrange[1])
    if noy:
        pyplot.yticks(locs,np.repeat('',len(locs)))
    else:
        labels = []
        for loc in locs:
            labels.append(str(loc))
        pyplot.yticks(locs,labels)
    #x-axis part
    locs = get_tick_positions(my_xrange[0], my_xrange[1])
    if nox:
        pyplot.xticks(locs,np.repeat('',len(locs)))
    else:
        labels = []
        for loc in locs:
            labels.append(str(loc))
        pyplot.xticks(locs,labels,rotation=rotation)

    return

# Get better tick positions for contour plots
# Currently produces two ticks only
def get_tick_positions(pmin, pmax):
    delta = (pmax-pmin)/3.

    ticks = [pmin+delta, pmin+2*delta]
    ndigits = 0
    while round(ticks[0], ndigits) == round(ticks[1], ndigits) or round(ticks[0],ndigits) < pmin or round(ticks[1], ndigits) > pmax:
        ndigits += 1
    
    ticks[0] = round(ticks[0], ndigits)
    ticks[1] = round(ticks[1], ndigits)

    if ticks[0] == 0:
        ticks[0] = 0
    if ticks[1] == 0:
        ticks[1] = 0

    delta = ticks[1] - ticks[0]

    if ticks[1] + delta < pmax - delta/10.:
        ticks.append(ticks[1] + delta)
    if ticks[0] - delta > pmin + delta/10.:
        ticks = [ticks[0] - delta] + ticks

    return ticks

#Plots N contours, instead of just one or two
def plot_many_contours(chains,cutoff,labels,pmin=[],pmax=[],nbins=20,rotation=35):
    '''
    Inputs: chain, cutoff, labels, pmin, pmax
    cutoff is the cut in the chain -- minimum position in the chain to use
    '''

    nparam = np.min([len(chains[0][0])-1,len(labels)])
    nchains = len(chains)

    #Set plotting boundaries if not already given
    if len(pmin)==0:
        pmin = np.zeros(nparam)
        for i in range(nparam):
            pmin[i] = np.min( chains[:,cutoff:,i])
    if len(pmax)==0:
        pmax = np.zeros(nparam)
        for i in range(nparam):
            pmax[i] = np.max( chains[:,cutoff:,i])

    colors = ['k','b','r','g','m','c']
    linetypes = ['-','--',':','.-']
    linetypes_text = ['solid','dashed','dotted','dashdot']

    #Set up our figure
    pyplot.figure(1,[11,8.5])
    pyplot.subplots_adjust(wspace=0,hspace=0)
    for i in range(nparam):
        for j in range(nparam):
            if j > i:
                #Leave upper right half of plot blank
                continue
            pyplot.subplot(nparam,nparam,i*nparam+j+1)
            if i==j:
                #This is where we'll want to make a histogram
                for mychain in range(nchains):
                    chain = chains[mychain]
                    hist = np.zeros(nbins)
                    xmin = np.min(chain[cutoff:,i])
                    xmax = np.max(chain[cutoff:,i])
                    dx = (xmax-xmin)/nbins
                    for k in range(nbins):
                        hist[k] = len(np.where( (chain[cutoff:,i] >= xmin+dx*k) & 
                                                (chain[cutoff:,i] < xmin+dx*(k+1)) )[0])
                    hist = hist/np.sum(hist)/dx
                    pyplot.plot(xmin+np.array(range(nbins))*dx+dx/2.,hist,colors[mychain]+linetypes[mychain])
                pyplot.xlim(pmin[i],pmax[i])
                pyplot.yticks([])
                # locs = [pmin[i]+(pmax[i]-pmin[i])/3.,pmin[i]+2*(pmax[i]-pmin[i])/3.]
                locs = get_tick_positions(pmin[i], pmax[i])
                myx_labels = []
                for loc in locs:
                    myx_labels.append(str(loc))
                pyplot.xticks(locs,myx_labels,rotation=rotation)
            else:
                #And here we make a contour plot
                if (j == 0) & (i>0):
                    noy = False
                else:
                    noy = True
                if i == nparam-1:
                    nox = False
                else:
                    nox = True
                #Plot each chain
                plot_some_contours(chains[:,cutoff:,j],chains[:,cutoff:,i],
                                   np.array([0.68,0.95]),nbins,
                                   [pmin[j],pmax[j]],[pmin[i],pmax[i]],noy=noy,nox=nox,
                                   rotation=rotation,colors=colors,linetypes=linetypes_text)
            if i == nparam-1:
                pyplot.xlabel(labels[j])
            if (j == 0) & (i > 0):
                pyplot.ylabel(labels[i])

    return

#General plotting function -- assumes one or two chains, gets the appropriate limits and labels
def plot_contour_set_nice(chain,chain_alt=[],cen=True,labels=[],
                          rotation=19,nbins=20):
    if len(labels)==0:
        if cen:
            labels = [r'$\sigma_L$','r',r'ln $L_0$',r'$A_L$',r'$B_L$',r'$s_c$']
        else:
            labels = [r'ln $\phi_0$',r'$A_\phi$',r'ln $L_{s0}$',r'$A_s$',r'$B_s$',r'$\alpha$',r'$s_s$']

    pmin = np.zeros(len(labels))
    pmax = np.copy(pmin)
    for i in range(len(pmin)):
        if len(chain_alt)==0:
            pmin[i] = np.min(chain[:,i])
            pmax[i] = np.max(chain[:,i])
        else:
            pmin[i] = np.min([np.min(chain[:,i]),np.min(chain_alt[:,i])])
            pmax[i] = np.max([np.max(chain[:,i]),np.max(chain_alt[:,i])])
            

    plot_set_contour(chain,0,labels,pmin,pmax,chain_alt=chain_alt,
                     rotation=rotation,nbins=nbins)

    return

#Parameter cleanup function
def parameter_cleanup(cen_param, sat_param, cen_err, sat_err):
    #Reording the parameters:
    #sigma_lm r lnLc0 A_L B_L s_c
    cen_param = [cen_param[0,0],cen_param[1,0],cen_param[2,0],cen_param[3,0],
                 cen_param[2,1],cen_param[4,0]]
    cen_err = [cen_err[0,0],cen_err[1,0],cen_err[2,0],cen_err[3,0],
                 cen_err[2,1],cen_err[4,0]]

    #Satellite parameters: lnphi0, B_phi, A_phi, b_A_phi, lnLs0, B_s, A_s, B_A_s, alpha, B_a, s
    sat_param = [sat_param[0,0],sat_param[0,1],sat_param[1,0],sat_param[1,1],
                 sat_param[2,0],sat_param[2,1],sat_param[3,0], sat_param[3,1],
                 sat_param[4,0],sat_param[4,1],sat_param[5,0]]
    sat_err = [sat_err[0,0],sat_err[0,1],sat_err[1,0],sat_err[1,1],
               sat_err[2,0],sat_err[2,1],sat_err[3,0], sat_err[3,1],
               sat_err[4,0],sat_err[4,1],sat_err[5,0]]
    return cen_param, sat_param, cen_err, sat_err

#Fixed script for DR8 plotting with minimal input
def plot_dr8(outdir):
    h = 0.6704
    Mpiv = 2.35e14

    #Old mass-lambda parameters
    #lm_param = [Mpiv, 0.838, 0.835, 3.0180]
    #Current mass-lambda parameters
    #lm_param = [Mpiv, 0.843, 0.750, 2.9458]
    #lm_param = [Mpiv, 0.7751, 0.4094, 3.0120]
    #lm_param = [Mpiv, 0.791, 0.4094, 3.0120]
    #lm_param = [Mpiv, 0.768, 0.5193, 3.0210]
    #lm_param = [Mpiv, 0.847, 0.541, 2.995]
    lm_param = [Mpiv, 0.842, 0.642, 3.141, -0.03, 0.1842]

    fits_dir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10_v2/"
    fits_dir_beta = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10_schcorr/"
    #Pick up the latest fit for the centrals
    #chain = np.loadtxt( fits_dir + "chain_cen_all.dat" )
    #cen_param = chain[np.argmax(chain[:,-1])]

    #TESTING
    #cen_param = [0.375, -0.55, 24.660, 0.391, 1.419]

    #Pick up the latest fit for the satellites
    #schain = np.loadtxt( fits_dir + "chain_sat_ev_all.dat" )
    #sat_param = schain[ np.argmax(schain[:,-1]) ]
    #sat_param = [3.971, 0.846, 23.186, 0.013, 1.81, -0.861]

    #Fit including extra parameter for non-schechter cutoff
    #schain_b = np.loadtxt(fits_dir_beta+"chain_sat_ev_all.dat")
    #sat_param_b = schain_b[ np.argmax(schain[:,-1])]

    #get the fit parameters
    #cen_param, cen_err, sat_param, sat_err = run_zev.fit_across_surveys_all(["/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10_v3/","/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10_v3/"],[0.1,0.2])
    cen_param, cen_err, sat_param, sat_err = run_zev.fit_across_surveys_all(["/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10_v3/"],[0.1])

    #Hard-coded parameter values
    cen_param = np.array([[0.3641, 0],[-0.5502, 0], [25.05117, 0.7714], [0.3769, 0], [1.1879, 0]])
    sat_param = np.array([[3.9356, 0.4823],[0.8640, 0.05134],[23.5356, 1.6501],[0.0594, 0.2386],[-0.7866, 0.6647],[1.4280, 0]])
    cen_syserr = [0, 0., 0.0775, 0.08602, 0]
    sat_syserr = [0.0326, 0.09431, 0.04446, 0.04272, 0.1235, 0]

    #Making the parameter plots
    dirlist = ["/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10_v3/","/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10_v3/"]
    run_zev.plot_zev_fit_cen(dirlist,[0.1,0.2],cen_param, cen_err,outdir+"pev_cen_all.ps",syserr=cen_syserr)
    run_zev.plot_zev_fit_sat(dirlist,[0.1,0.2],sat_param, sat_err,outdir+"pev_sat_all.ps",syserr=sat_syserr)

    #Reording the parameters:
    #sigma_lm r lnLc0 A_L B_L s_c
    cen_param = [cen_param[0,0],cen_param[1,0],cen_param[2,0],cen_param[3,0],
                 cen_param[2,1],cen_param[4,0]]
    cen_err = [cen_err[0,0],cen_err[1,0],cen_err[2,0],cen_err[3,0],
                 cen_err[2,1],cen_err[4,0]]

    #Satellite parameters: lnphi0, B_phi, A_phi, b_A_phi, lnLs0, B_s, A_s, B_A_s, alpha, B_a, s
    sat_param = [sat_param[0,0],sat_param[0,1],sat_param[1,0],sat_param[1,1],
                 sat_param[2,0],sat_param[2,1],sat_param[3,0], sat_param[3,1],
                 sat_param[4,0],sat_param[4,1],sat_param[5,0]]
    sat_err = [sat_err[0,0],sat_err[0,1],sat_err[1,0],sat_err[1,1],
               sat_err[2,0],sat_err[2,1],sat_err[3,0], sat_err[3,1],
               sat_err[4,0],sat_err[4,1],sat_err[5,0]]
    
    print >> sys.stderr, "Centrals: "
    for i in range(len(cen_param)):
        print >> sys.stderr, cen_param[i], cen_err[i]
    print >> sys.stderr, "Satellites:"
    for i in range(len(sat_param)):
        print >> sys.stderr, sat_param[i], sat_err[i]

    #Set up the input directories
    indir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/dr8_zlambda_v5.10_cencorr/"
    indir_uber = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/dr8_zlambda_v5.10_cencorr/"

    #Set up the redshift ranges
    zmin = [0.1, 0.15, 0.2, 0.25, 0.3]
    zmax = [0.15, 0.2, 0.25, 0.3, 0.33]
    zmin = np.array(zmin)
    zmax = np.array(zmax)
    
    #Set up lambda ranges
    lm_min = [20., 30., 60.]
    lm_max = [25., 40., 100.]
    lm_min = np.array(lm_min)
    lm_max = np.array(lm_max)
    lm_med = [22.1, 33.9, 70.7]

    #Set up iband error -- not currently relevant
    iband_err = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

    #Get file list for halo masses
    mf_files = glob(indir+"../mass_functions_planck/mf_planck_*.dat")
    mf_files = np.array(mf_files)
    mf_files.sort()
    mf_files = mf_files[[1,3,5,7,10]]

    #Make the n(lambda) file list
    #print indir_uber
    nlm_files = glob(indir_uber+"nlambda_z_*.dat")
    nlm_files = np.array(nlm_files)
    nlm_files.sort()
    #Fix the ordering
    nlm_files = nlm_files[[1,0,3,2,4]]
    
    #Make the mass_param list for the CLFs
    mf_all = []
    mass_param = []
    for i in range(len(mf_files)):
        mf = np.loadtxt(mf_files[i])
        mf[:,0] = mf[:,0]/h
        mf[:,1] = mf[:,1]*h**3
        mf_all.append(mf)
        mass_param.append(fit_plm.nm_approx_third(mf[:,0], mf[:,1], Mpiv))
    mf_all = np.array(mf_all)
    mass_param = np.array(mass_param)

    #Plot the n(lambda) comparison
    #print nlm_files, mf_files
    plot_nlambda_set_nice(lm_param,zmin,zmax,mf_files,nlm_files,outdir,bigger=False,
                          Alm_ev=True)

    #Plot a set of CLFs
    plot_clf_set_new(indir,lm_param,cen_param,sat_param,lm_min,lm_max,lm_med,mass_param,zmin[[0,2,4]],zmax[[0,2,4]],mf_all,outdir+"clf_set_dr8.ps")

    #Plot brightest satellite distribution
    plot_clf_brightest_new(indir,lm_param,sat_param,lm_min,lm_max,lm_med,mass_param,zmin[[0,2,4]],zmax[[0,2,4]],
                           mf_all,outdir+"clf_bsat_dr8.ps",cen_param=cen_param)

    return


#Same as above, but setup to run for S82
def plot_s82(outdir):
    h = 0.6704
    Mpiv = 2.35e14

    #Old mass-lambda parameters
    #lm_param = [Mpiv, 0.867, 0.861, 2.9555, 0.1842]
    #New mass-lambda parameters
    lm_param = [Mpiv, 0.916, 1.11, 3.18, -1.04, 0.1842]
    
    #DR8 parameters
    lm_param_alt = [Mpiv, 0.842, 0.642, 3.141, -0.03, 0.1842]
    #lm_param = lm_param_alt

    fits_dir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10_v2/"
    #Pick up the latest fit for the centrals
    #chain = np.loadtxt( fits_dir + "chain_cen_all.dat" )
    #cen_param = chain[np.argmax(chain[:,-1])]

    #cen_param_alt = [0.37, -0.5, 24.682, 0.302, 1.23, 3.1]
    #cen_param_alt = []

    #Pick up the latest fit for the satellites
    #schain = np.loadtxt( fits_dir + "../fits_plm_full_s82_v5.10/chain_sat_ev_all.dat" )
    #sat_param = schain[ np.argmax(schain[:,-1]) ]
    #sat_param = [4.12, 0.8, 22.9, 0.05, 2.0, -0.95, 0.1, 3.84]

    #Hard-coded -- parameters from fitting to DR8 only, fixed to avoid issues
    cen_param = [[0.3641],[-0.5502], [25.05117, 0.7714], [0.3769], [1.1879]]
    sat_param = np.array([[3.9356, 0.4823],[0.8640, 0.05134],[23.5356, 1.6501],[0.0594, 0.2386],[-0.7866, 0.6647],[1.4280, 0]])

    #get the fit parameters
    #cen_param, cen_err, sat_param, sat_err = run_zev.fit_across_surveys_all(["/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10_v3/","/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10_v3/"],[0.1,0.2])
    #cen_param, cen_err, sat_param, sat_err = run_zev.fit_across_surveys_all(["/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10_v3/"],[0.1])

    #Estimate the systematic error we need
    cen_syserr, sat_syserr = get_sys_err.get_s82_sys_errs(cen_param,sat_param)
    print cen_syserr
    print sat_syserr
    print "Moving on..."

    #Reording the parameters:
    #sigma_lm r lnLc0 A_L B_L s_c
    cen_param = [cen_param[0][0],cen_param[1][0],cen_param[2][0],cen_param[3][0],
                 cen_param[2][1],cen_param[4][0]]
    #cen_err = [cen_err[0][0],cen_err[1][0],cen_err[2][0],cen_err[3][0],
    #             cen_err[2][1],cen_err[4][0]]

    #Satellite parameters: lnphi0, B_phi, A_phi, b_A_phi, lnLs0, B_s, A_s, B_A_s, alpha, B_a, s
    sat_param = [sat_param[0,0],sat_param[0,1],sat_param[1,0],sat_param[1,1],
                 sat_param[2,0],sat_param[2,1],sat_param[3,0], sat_param[3,1],
                 sat_param[4,0],sat_param[4,1],sat_param[5,0]]
    #sat_err = [sat_err[0,0],sat_err[0,1],sat_err[1,0],sat_err[1,1],
    #           sat_err[2,0],sat_err[2,1],sat_err[3,0], sat_err[3,1],
    #           sat_err[4,0],sat_err[4,1],sat_err[5,0]]


    print >> sys.stderr, cen_param
    print >> sys.stderr, sat_param

    #Set up the input directories
    indir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/s82_v5.10_uber/"

    #Set up the redshift ranges
    zmin = [0.1, 0.2, 0.3, 0.4, 0.5]
    zmin = np.array(zmin)
    zmax = zmin + 0.1
    
    #Set up lambda ranges
    lm_min = [10., 20., 40.]
    #lm_max = [20., 100., 100.]
    lm_max = [20., 40., 100.]
    lm_min = np.array(lm_min)
    lm_max = np.array(lm_max)
    lm_med = [13., 25.3, 47.5]

    #Set up iband error -- not currently relevant
    iband_err = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

    #Get file list for halo masses
    mf_files = glob(indir+"../mass_functions_planck/mf_planck_*.dat")
    mf_files = np.array(mf_files)
    mf_files.sort()
    mf_files = mf_files[[2,6,12,14,16]]

    #Make the n(lambda) file list
    #print indir_uber
    nlm_files = glob(indir+"nlambda_z_*.dat")
    nlm_files = np.array(nlm_files)
    nlm_files.sort()
    
    #Make the mass_param list for the CLFs
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

    #Plot the n(lambda) comparison
    #print nlm_files, mf_files
    plot_nlambda_set_nice(lm_param,zmin,zmax,mf_files,nlm_files,outdir,bigger=False,inset_args=[lm_param_alt],Alm_ev=True)

    #Plot a set of CLFs
    plot_clf_set_new(indir,lm_param,cen_param,sat_param,lm_min,lm_max,lm_med,mass_param,zmin[[1,2,4]],zmax[[1,2,4]],mf_all,outdir+"clf_set_s82.ps")
    

    return


#Plotting routine for comparing a pair of satellite fits to Eli's input Lst evolution
def Lst_ev(param_cen,param_sat,outfile,err_cen=[],err_sat=[]):
    #Get the values we want
    z = np.array(range(1000))*0.001
    Lval = fit_psat.Lmin_eli(z) - np.log10(0.2)

    Lst = (param_sat[4] + param_sat[5]*np.log((1+z)/1.3))/np.log(10)

    #Main plotting section
    pyplot.figure(1,[11., 8.5])
    p3 = pyplot.plot(z,Lval,'r',label=r'$L_{*,def}$',lw=2)
    p1 = pyplot.plot(z,Lst,'k--',label='$L_s(z)$')
    if len(err_sat)>0:
        pyplot.fill_between(z,Lst+(err_sat[4][0])/np.log(10),Lst-(err_sat[4][1])/np.log(10),color='0.75',label='_nolabel_')
        pyplot.fill_between(z,Lst+(err_sat[5][0]*np.log((1+z)/1.3))/np.log(10),Lst-(err_sat[5][1]*np.log((1+z)/1.3))/np.log(10),color='0.75',label='_nolabel_')

    Lcen = (param_cen[2] + param_cen[4]*np.log((1+z)/1.3))/np.log(10.)
    p4 = pyplot.plot(z,Lcen,'k',label='$L_c(z)$')
    if len(err_cen)>0:
        pyplot.fill_between(z,Lcen+(err_cen[2][0])/np.log(10),Lcen-(err_cen[2][1])/np.log(10),
                            color='0.75',label='_nolabel_')
        pyplot.fill_between(z,Lcen+(err_cen[4][0]*np.log((1+z)/1.3))/np.log(10),Lcen-(err_cen[4][1]*np.log((1+z)/1.3))/np.log(10),color='0.75',label='_nolabel_')
       
    pyplot.xlim([0.1,0.7])
    pyplot.ylim([9.9,11.2])

    pyplot.xlabel('z',fontsize=30)
    pyplot.ylabel(r'log(L) $L_\odot/h^2$',fontsize=30)

    #pyplot.legend(loc='lower right')
    pyplot.tick_params(axis='both', which='major', labelsize=20)

    pyplot.tight_layout()

    pyplot.savefig(outfile,orientation='landscape')
    pyplot.clf()
    
    return


#Plotting evolution with SVA1 included -- does an arbitrary number of lines
def Lst_ev_set(param,param_cen,labels,outfile):
    #Get the values we want
    z = np.array(range(1000))*0.001
    Lval = fit_psat.Lmin_eli(z) - np.log10(0.2)
    
    nsets = len(param)
    Lst = np.zeros([len(param),len(z)])
    Lcen = np.copy(Lst)
    for i in range(nsets):
        Lst[i,:] = (param[i][2] + param[i][4]*np.log(1+z))/np.log(10)
        Lcen[i,:] = (param_cen[i][2] + param_cen[i][4]*np.log(1+z))/np.log(10.)

    colorlist = ['k','b','r','g','c']
    pyplot.figure(1,[11., 8.5])
    for i in range(nsets):
        pyplot.plot(z,Lcen[i],label=labels[i],color=colorlist[i])
        pyplot.plot(z,Lst[i],color=colorlist[i],linestyle='--',label='_nolabel_')
        print Lst[i][0]
    
    pyplot.plot(z,Lval,color='c',linewidth=2)
    
    pyplot.xlim([0.1,0.9])
    pyplot.ylim([9.9,11.2])

    pyplot.xlabel('z',fontsize=20)
    pyplot.ylabel(r'log(L) $L_\odot/h^2$',fontsize=20)

    pyplot.legend(loc='lower right')
    pyplot.tick_params(axis='both', which='major', labelsize=20)

    pyplot.savefig(outfile,orientation='landscape')
    pyplot.clf()

    return


#Same as above, but setup to run for SVA1 gold catalog
#Currently displaying old S82 fit results
def plot_sva1(outdir):
    h = 0.6704
    Mpiv = 2.35e14

    #Old mass-lambda parameters
    #lm_param = [Mpiv, 0.867, 0.861, 2.9555]
    #New mass-lambda parameters for S82
    #lm_param_s82 = [Mpiv, 0.7766, 1.5665, 2.6852, 0.1842]
    lm_param_s82 = [Mpiv, 0.916, 1.11, 3.18, -1.04, 0.1842]

    #mass-lambda parameters for sva1
    #lm_param = [Mpiv, 0.8346, 0.4687, 3.0199,0.1842]
    #lm_param = [Mpiv, 0.7696, 1.4646, 2.6239, 0.1842] #0.2<z<0.9 fit
    lm_param = [Mpiv, 0.918, 1.18,3.118, -0.42, 0.1842]

    fits_dir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10_v2/"
    #Pick up the latest fit for the centrals
    chain = np.loadtxt( fits_dir + "chain_cen_all.dat" )
    cen_param = chain[np.argmax(chain[:,-1])]

    fits_dir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10/"
    #Pick up the latest fit for the satellites
    schain = np.loadtxt( fits_dir + "chain_sat_ev_all.dat" )
    sat_param = schain[ np.argmax(schain[:,-1]) ]

    #Now get the parameters for SVA1
    fits_dir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_sva1_v6.1.3/"
    chain_sv = np.loadtxt(fits_dir+"chain_cen_all.dat")
    cen_param_sv = chain_sv[np.argmax(chain_sv[:,-1])]
    #cen_param_sv = [0.36, -0.8, 24.85, 0.31, 1.2, 1.]
    #print cen_param, cen_param_sv

    #sat_param_sv = np.copy(sat_param)
    schain_sv= np.loadtxt(fits_dir+"chain_sat_ev_all.dat")
    sat_param_sv = schain_sv[np.argmax(schain_sv[:,-1])]

    print >> sys.stderr, cen_param
    print >> sys.stderr, cen_param_sv
    #print >> sys.stderr, sat_param

    #Set up the input directories
    indir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/sv1_gold_v6.1.3_uber/"
    indir_uber = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/sv1_gold_v6.1.3_uber/"
    #Set up the redshift ranges
    zmin = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    zmin = np.array(zmin)
    zmax = zmin + 0.1
    
    #Set up lambda ranges
    lm_min = [10., 20., 40.]
    lm_max = [20., 40., 100.]
    lm_min = np.array(lm_min)
    lm_max = np.array(lm_max)
    #lm_med = [13., 27.4, 47.5]
    lm_med = [13., 25.3, 47.5]

    #Set up iband error -- not currently relevant
    iband_err = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    #Get file list for halo masses
    mf_files = glob(indir+"../mass_functions_planck/mf_planck_*.dat")
    mf_files = np.array(mf_files)
    mf_files.sort()
    mf_files = mf_files[[2,6,12,14,16,18,20,22]]

    #Make the n(lambda) file list
    #print indir_uber
    nlm_files = glob(indir_uber+"nlambda_z_*.dat")
    nlm_files = np.array(nlm_files)
    nlm_files.sort()
    
    #Make the mass_param list for the CLFs
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

    #Plot the n(lambda) comparison
    #print nlm_files, mf_files
    #Area correction
    area_table = pyfits.open("/nfs/slac/g/ki/ki19/des/erykoff/des/sva1/gold_1.0/redmapper_v6.1/run/sva1_gold_1.0.2_run_redmapper_v6.1.3_lgt20_vlim_area.fit")
    area_table = area_table[1].data
    area_bin = mass_matching.estimate_area(zmin[1:],zmax[1:],area_table['z'],area_table['area'])/254.4
    plot_nlambda_set_nice(lm_param,zmin[1:],zmax[1:],mf_files[1:],nlm_files[1:],
                          outdir,bigger=True,Alm_ev=True,area_correct=area_bin)

    #Plot a set of CLFs
    #plot_clf_set(indir, indir_uber, lm_param, cen_param_sv, sat_param_sv, lm_min, lm_max, lm_med, mass_param[[1,2,4]], zmin[[1,2,4]], zmax[[1,2,4]], iband_err[[1,2,4]], mf_all[[1,2,4]], outdir+"clf_set_sva1.ps",abs_solar=4.71493,fix_alpha=False,phi_ev=True,
    #             lm_param_alt=lm_param_s82,cen_param_alt=cen_param,sat_param_alt=sat_param)
    #plot_clf_set(indir, indir_uber, lm_param, cen_param_sv, sat_param_sv, lm_min, lm_max, lm_med, mass_param[[5,6,7]], zmin[[5,6,7]], zmax[[5,6,7]], iband_err[[5,6,7]], mf_all[[5,6,7]], outdir+"clf_set_sva1_hiz.ps",abs_solar=4.71493,fix_alpha=False,phi_ev=True,lm_param_alt=lm_param_s82,cen_param_alt=cen_param,sat_param_alt=sat_param)

    #plot_clf_set_ratio(indir,indir_uber,lm_param,cen_param_sv,sat_param_sv,lm_min,lm_max,lm_med,
    #                    mass_param[[5,6,7]], zmin[[5,6,7]], zmax[[5,6,7]], iband_err[[5,6,7]], mf_all[[5,6,7]], outdir+"clf_ratio_sva1_hiz.ps",abs_solar=4.71493,phi_ev=True,fix_alpha=False,
    #                   yrange=[-0.5,0.5])

    return


#Anothr plotting routine wrapper, this one for Aardvark w/ S82 photometry
#Currently plotting against S82 fit for comparison
def plot_aa_s82(outdir):
    h = 0.6704
    Mpiv = 2.35e14

    #New mass-lambda parameters
    lm_param = [Mpiv, 0.857, 1.547, 2.7226, 0.1842]
    
    fits_dir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10/"
    #Pick up the latest fit for the centrals
    chain = np.loadtxt( fits_dir + "chain_cen_all.dat" )
    cen_param = chain[np.argmax(chain[:,-1])]

    #Pick up the latest fit for the satellites
    schain = np.loadtxt( fits_dir + "chain_sat_ev_all.dat" )
    sat_param = schain[ np.argmax(schain[:,-1]) ]
    #sat_param = [4.12, 0.8, 22.9, 0.05, 2.0, -0.95, 0.1, 3.84]

    #ibanderr - not actually used right now
    ibanderr = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

    #Set up the input directories
    indir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/aa1.0_s82_v5.10/"
    indir_uber = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/aa1.0_s82_v5.10//"

    #Set up the redshift ranges
    zmin = [0.1, 0.15, 0.2, 0.25, 0.3, 0.25, 0.4, 0.25, 0.5, 0.55]
    zmin = np.array(zmin)
    zmax = zmin + 0.05
    
    #Set up lambda ranges
    lm_min = [20., 20., 25., 30., 40., 40., 60.]
    lm_max = [100., 25., 30., 40., 60., 100., 100.]
    lm_min = np.array(lm_min)
    lm_max = np.array(lm_max)
    lm_med = np.array([26.23, 22.08, 27.10, 33.73, 46.15, 49.171, 70.83])

    #Set up iband error -- not currently relevant
    iband_err = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

    #Get file list for halo masses
    mf_files = glob(indir+"../mass_functions_planck/mf_planck_*.dat")
    mf_files = np.array(mf_files)
    mf_files.sort()
    mf_files = mf_files[[1,3,5,7,11,13,14,15,16,17]]
    print mf_files

    #Make the n(lambda) file list
    #print indir_uber
    nlm_files = glob(indir_uber+"nlambda_z_*.dat")
    nlm_files = np.array(nlm_files)
    nlm_files.sort()
    
    #Make the mass_param list for the CLFs
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

    #Plot the n(lambda) comparison
    #print nlm_files, mf_files
    #plot_nlambda_set_nice(lm_param,zmin,zmax,mf_files,nlm_files,outdir,bigger=False)

    #Plot a set of CLFs
    plot_clf_set(indir, indir_uber, lm_param, cen_param, sat_param, lm_min[[1,3,5]], lm_max[[1,3,5]], lm_med[[1,3,5]], mass_param[[0,4,9]], zmin[[0,4,9]], zmax[[0,4,9]], ibanderr[[0,2,4]], mf_all[[0,4,9]], outdir+"clf_set_s82_chi2.ps",abs_solar=4.71493,fix_alpha=False,phi_ev=True)

    #plot_clf_set(indir, indir_uber, lm_param, cen_param, sat_param, lm_min, lm_max, lm_med, mass_param[[0,1,3]], zmin[[0,1,3]], zmax[[0,1,3]], iband_err[[0,1,3]], mf_all[[0,1,3]], outdir+"clf_set_s82_chi2_extra.ps",abs_solar=4.71493,fix_alpha=False,sat_param_alt=sat_param_alt,cen_param_alt=cen_param_alt,phi_ev=True)

    #Plot a CLF ratio test
    #plot_clf_set_ratio(indir, indir_uber, lm_param, cen_param, sat_param, lm_min, lm_max, lm_med, mass_param[[0,2,4]], zmin[[0,2,4]], zmax[[0,2,4]], iband_err[[0,2,4]], mf_all[[0,2,4]], outdir+"clf_ratio_s82_chi2.ps",abs_solar=4.71493,fix_alpha=False,phi_ev=True)

    #Plot CLF comparison with brightest satellite distribution
    #plot_clf_brightest(indir, lm_param, sat_param, lm_min, lm_max, lm_med, mass_param[[1,2,4]], zmin[[1,2,4]], zmax[[1,2,4]], mf_all[[1,2,4]], outdir+"clf_sat_brightest_s82_chi2.ps",phi_ev=True)

    
    return

#Plot of Lcen mass dependence vs Hansen 2009, other literature references
def plot_Lcen_mass(cen_param, cen_err, outdir):
    h = 0.6704
    Mpiv = 2.35e14
    zval = 0.25

    #Mass bins
    mbins = np.array(range(201))*0.01+13

    #Hansen values, converted to virial mass
    Mpiv_hansen = 1.66e14/h

    #Getting the central luminosities
    Lcen_hansen = 6e10*(10.**mbins/Mpiv_hansen)**0.3
    Lcen_h_hi = 8e10*(10.**mbins/Mpiv_hansen)**0.3
    Lcen_h_lo = 4e10*(10.**mbins/Mpiv_hansen)**0.3

    #Getting DR8 central luminosities at z=0.25
    #fits_dir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10_v2/"
    #chain_dr8 = np.loadtxt(fits_dir + "chain_cen_all.dat")
    #cen_param_dr8 = chain_dr8[np.argmax(chain_dr8[:,-1])]

    #Getting S82 central luminosities at z=0.25
    #fits_dir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10_v2/"
    #chain_s82 = np.loadtxt(fits_dir + "chain_cen_all.dat")
    #cen_param_s82 = chain_s82[np.argmax(chain_s82[:,-1])]

    pyplot.figure(1,[11.,8.5])

    #cen_param_dr8 = [0.363667675133445, -0.52335780572853252, 25.071104716809213, 0.37294772175919816, 1.1170727503329549, 1.2370806203231806]
    #cen_err = [[ 0.01123446,  0.01148164], [ 0.1008863 ,  0.11445454], [ 0.0206819 ,  0.02246899], [ 0.00721075,  0.00704131], [ 0.20869451,  0.19020458], [ 0.08093765,  0.0666278 ]]

    pyplot.plot(mbins,np.log10(Lcen_hansen),'k',label='Hansen 09 (SDSS maxBCG)')
    pyplot.fill_between(mbins,np.log10(Lcen_h_hi),np.log10(Lcen_h_lo),color='0.85')

    pyplot.plot(mbins,(cen_param[2]+cen_param[3]*(mbins*np.log(10.)-np.log(Mpiv)) + cen_param[4]*np.log((1+zval)/1.3))/np.log(10.),'b--' ,label='This paper (SDSS redmapper)', lw=3)
    pyplot.fill_between(mbins,(cen_param[2]+0.009*np.log(10)+cen_param[3]*(mbins*np.log(10.)-np.log(Mpiv)) + cen_param[4]*np.log((1+zval)/1.3))/np.log(10.),(cen_param[2]-0.009*np.log(10)+cen_param[3]*(mbins*np.log(10.)-np.log(Mpiv)) + cen_param[4]*np.log((1+zval)/1.3))/np.log(10.),color=[0.7, 0.7, 1.],alpha='0.8')

    #Get the abundance matching results
    mydir = "/nfs/slac/g/ki/ki10/rmredd/BolshoiCheck_SHAMtest/"
    Lcen_ber = np.loadtxt(mydir+"Lcen_M_bernardi.dat")
    Lcen_dr7 = np.loadtxt(mydir+"Lcen_M_dr7.dat")

    pyplot.plot(Lcen_ber[:,0]-np.log10(.7),Lcen_ber[:,1],'go-',label='SHAM, Bernarder 2013')
    pyplot.plot(Lcen_dr7[:,0]-np.log10(.7),Lcen_dr7[:,1],'mo-',label='SHAM, DR7')
    pyplot.xlim([13,15])

    pyplot.xlabel(r'$log(M_{vir})$ $[M_\odot]$',fontsize=30)
    pyplot.ylabel(r'$log(L_{cen})$ $[L_\odot/h^2]$',fontsize=30)

    pyplot.legend(loc='upper left', frameon=False)
    pyplot.tick_params(axis='both',which='major',labelsize=20)

    pyplot.tight_layout()

    pyplot.savefig(outdir+"Lcen_lit_comp.pdf",orientation='landscape')
    pyplot.clf()

    return
