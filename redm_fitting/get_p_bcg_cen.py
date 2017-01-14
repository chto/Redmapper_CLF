#!/usr/bin/python

import numpy as np
import math
import scipy.special
import matplotlib.pyplot as pyplot

import fit_plm
import fit_psat

#Function for making scipy spit out results for gamma(a,x) with a < 0
def gamma_neg(a,x):
    if a > 0:
        gamma = scipy.special.gammaincc(a,x)*scipy.special.gamma(a)
    if a == 0:
        gamma = -scipy.special.expi(-x)
    if a < 0:
        gamma = (gamma_neg(a+1,x) - x**a*np.exp(-x))/a
    
    return gamma

#Basic function for getting P(BCG!=central).
def get_p_bcg_cen_basic(L0,sigma_L,Lst,phist,alpha,Lmin=9.):
    '''
    Basic function for getting P(BCG!=central)
    
    Inputs:
    L0 (mean central luminosity)
    sigma_L ; assumed to be in terms of log10(L)
    Lst (characteristic satellite luminosity)
    phist 
    alpha
    Lmin=9. by default
    Note all luminosity inputs should be in log10(L)
    '''

    #First, get the expected number of satellites
    Ns = phist/np.log(10.)*gamma_neg(alpha+1,10.**Lmin/10.**Lst)
    #print "Ns: ",Ns,phist/np.log(10.),alpha

    #Now, run the integral
    #Set up the log(L) bins
    dlogL = 0.01
    Lbins = Lmin+dlogL*np.array(range(500))

    p = np.sum( dlogL/np.sqrt(2*np.pi)/sigma_L*np.exp(-(Lbins-L0)*(Lbins-L0)/2./sigma_L/sigma_L) *(1-np.exp(-Ns*gamma_neg(alpha+1,10.**(Lbins-Lst))/gamma_neg(alpha+1,10.**(Lmin-Lst)))) )
    #for i in range(len(Lbins)):
    #    if i < 20:
    #        print i, dlogL/np.sqrt(2*np.pi)/sigma_L*np.exp(-(Lbins[i]-L0)*(Lbins[i]-L0)/2./sigma_L/sigma_L),1,Ns,-np.exp(-Ns*gamma_neg(alpha+1,10.**(Lbins[i]-Lst))/gamma_neg(alpha+1,10.**(Lmin-Lst)))

    return p

#Gets P(BCG!=central) for central/satellite parameters from CLF fits
def get_p_bcg_cen(param_cen,param_sat,mhalo=2.35e14,z=0.3,phi_ev=False):
    '''
    Operates for a single redshift and host halo mass
    '''
    
    Mpiv = 2.35e14
    
    sigma_L = param_cen[0]/np.log(10.)
    L0 = (param_cen[2] + param_cen[3]*np.log(mhalo/Mpiv)+param_cen[4]*np.log(1.+z))/np.log(10.)
    phist = np.exp(param_sat[0] + param_sat[1]*np.log(mhalo/Mpiv))
    if phi_ev:
        phist = phist*(1.+z)**param_sat[6]
    Ls = (param_sat[2] + param_sat[3]*np.log(mhalo/Mpiv)+param_sat[4]*np.log(1.+z))/np.log(10.)
    alpha = param_sat[5]

    #print L0, sigma_L,Ls,phist,alpha
    p = get_p_bcg_cen_basic(L0,sigma_L,Ls,phist,alpha)

    return p

#Gets P(BCG!=central) for central/satellite parameters with complete correlation (r=+-1)
def get_p_bcg_corr_complete(L0,sigma_L,Lst,phist,alpha,Lmin=9.,r=1):
    '''
    Basic function for getting P(BCG!=central)
    
    Inputs:
    L0 (mean central luminosity)
    sigma_L ; assumed to be in terms of log10(L)
    Lst (characteristic satellite luminosity)
    phist 
    alpha
    Lmin=9. by default
    r=1 by default
    Note all luminosity inputs should be in log10(L)
    '''

    if abs(r) != 1:
        print "Error: require r=1 or r=-1"
        return -1

    #First, get the expected number of satellites
    Ns = phist/np.log(10.)*gamma_neg(alpha+1,10.**Lmin/10.**Lst)    
    

    nvals = np.array(range(300))
    pvals = 0.*nvals
    for i in range(len(nvals)):
        if i < 100:
            if i == 0:
                pvals[i] = (Ns**nvals[i]/np.math.factorial(nvals[i]))*np.exp(-Ns)*(1-(1-gamma_neg(alpha+1,0)/gamma_neg(alpha+1,10.**(Lmin-Lst)))**nvals[i])
            else:
                pvals[i] = (Ns**nvals[i]/np.math.factorial(nvals[i]))*np.exp(-Ns)*(1-(1-gamma_neg(alpha+1,10.**(L0-Lst)*(nvals[i]/Ns)**(r*sigma_L*np.sqrt(Ns)))/gamma_neg(alpha+1,10.**(Lmin-Lst)))**nvals[i])
            #print i, Ns, pvals[i], (Ns**nvals[i]/np.math.factorial(nvals[i]))*np.exp(-Ns), 1-(1-gamma_neg(alpha+1,10.**(L0-Lst)*(nvals[i]/Ns)**(r*sigma_L*np.sqrt(Ns)))/gamma_neg(alpha+1,10.**(Lmin-Lst)))**nvals[i]
        else:
            #Switch to stirling's approximation
            pvals[i] = (Ns/nvals[i])**nvals[i]/np.sqrt(2*np.pi*nvals[i])*np.exp(nvals[i]-Ns)*(1-(1-gamma_neg(alpha+1,10.**(L0-Lst)*(nvals[i]/Ns)**(r*sigma_L*np.sqrt(Ns)))/gamma_neg(alpha+1,10.**(Lmin-Lst)))**nvals[i])
            #print i, Ns, pvals[i], (Ns/nvals[i])**nvals[i]/np.sqrt(2*np.pi*nvals[i])*np.exp(nvals[i]-Ns), 1-(1-gamma_neg(alpha+1,10.**(L0-Lst)*(nvals[i]/Ns)**(r*sigma_L*np.sqrt(Ns)))/gamma_neg(alpha+1,10.**(Lmin-Lst)))**nvals[i]

    return np.sum(pvals)

#Get P(BCG!=central) using complete correlations and data fit results
def get_p_bcg_cen_corr_comp_fit(param_cen,param_sat,mhalo=2.35e14,z=0.3,r=1):
    '''
    Operates for a single redshift and host halo mass
    '''
    
    Mpiv = 2.35e14
    
    sigma_L = param_cen[0]/np.log(10.)
    L0 = (param_cen[2] + param_cen[3]*np.log(mhalo/Mpiv)+param_cen[4]*np.log(1.+z))/np.log(10.)
    phist = np.exp(param_sat[0] + param_sat[1]*np.log(mhalo/Mpiv))
    Ls = (param_sat[2] + param_sat[3]*np.log(mhalo/Mpiv)+param_sat[4]*np.log(1.+z))/np.log(10.)
    alpha = param_sat[5]

    #print L0, sigma_L,Ls,phist,alpha
    p = get_p_bcg_corr_complete(L0,sigma_L,Ls,phist,alpha,r=r)
    
    return p

#Full version, including full range for -1 < r < 1
def get_p_bcg_corr_gen(L0,sigma_L,Lst,phist,alpha,Lmin=9.,r=0.):
    '''
    Basic function for getting P(BCG!=central)
    
    Inputs:
    L0 (mean central luminosity)
    sigma_L ; assumed to be in terms of log10(L)
    Lst (characteristic satellite luminosity)
    phist 
    alpha
    Lmin=9. by default
    r=0 by default
    Note all luminosity inputs should be in log10(L)
    '''

    #First, get the expected number of satellites
    Ns = phist/np.log(10.)*gamma_neg(alpha+1,10.**Lmin/10.**Lst)    
    
    #Now, run the integral
    #Set up the log(L) bins
    dlogL = 0.01
    Lbins = Lmin+dlogL*np.array(range(500))

    nvals = np.array(range(600))
    pvals = np.zeros([len(nvals),len(Lbins)])
    

    fac = (1-gamma_neg(alpha+1,10.**(Lbins-Lst))/gamma_neg(alpha+1,10.**(Lmin-Lst)))
    for i in range(len(nvals)):
        if i < 100:
            if i == 0:
                pvals[i,:] = 0*Lbins
            else:
                pvals[i,:] = (Ns**nvals[i]/np.math.factorial(nvals[i]))*np.exp(-Ns)*(1-fac**nvals[i])*np.exp( -(Lbins - L0 - r*sigma_L*np.sqrt(Ns)*np.log10(nvals[i]/Ns) )**2/2./sigma_L/sigma_L/(1-r*r) )
        else:
            #Switch to stirling's approximation
            pvals[i,:] = (Ns/nvals[i])**nvals[i]/np.sqrt(2*np.pi*nvals[i])*np.exp(nvals[i]-Ns)*(1-fac**nvals[i])*np.exp( -(Lbins - L0 - r*sigma_L*np.sqrt(Ns)*np.log10(nvals[i]/Ns) )**2/2./sigma_L/sigma_L/(1-r*r) )
        #print i,np.sum(pvals[i])/np.sqrt(2*np.pi*(1-r*r))/sigma_L*dlogL

    p = np.sum(pvals)/np.sqrt(2*np.pi*(1-r*r))/sigma_L*dlogL
    
    return p

#Takes input parameters; may choose to override r
def get_p_bcg_corr_gen_fit(param_cen,param_sat,mhalo=2.35e14,z=0.3,r=None,Lmin=9.,phi_ev=False):
    '''
    Operates for a single redshift and host halo mass
    '''
    
    Mpiv = 2.35e14
    
    sigma_L = param_cen[0]/np.log(10.)
    L0 = (param_cen[2] + param_cen[3]*np.log(mhalo/Mpiv)+param_cen[4]*np.log(1.+z))/np.log(10.)
    phist = np.exp(param_sat[0] + param_sat[1]*np.log(mhalo/Mpiv))
    if phi_ev:
        phist = phist*(1+z)**param_sat[6]
    Ls = (param_sat[2] + param_sat[3]*np.log(mhalo/Mpiv)+param_sat[4]*np.log(1.+z))/np.log(10.)
    alpha = param_sat[5]

    if r == None:
        r = param_cen[1]

    #print L0, sigma_L,Ls,phist,alpha
    p = get_p_bcg_corr_gen(L0,sigma_L,Ls,phist,alpha,r=r,Lmin=Lmin) 

    return p

#Versions for getting P(BNC) that modify the input Schechter function
#Note this is slower due to requiring tabulated integrations
#This version also assumes evolution in phi*
#Segment that does the computation
def get_p_bcg_sch_corr_single(L0,sigma_L,Ls,phist,alpha,r,Lmin=9.,beta=1.,use_sch_fac=False,do_output=False):
    #First, set up the luminosity array we're integrating over
    dL = 0.01
    Lbins = Lmin+dL/2.+np.array(range(int((12-Lmin)/dL)))*dL

    #Get the non-schechter correction factor
    ratio = 10.**(Lbins-Ls)
    if use_sch_fac:
        sch_fac = fit_psat.schechter_corr_dr8(np.log10(ratio))

    #Now get the expected number of satellites
    Ns = np.sum( phist*ratio**(alpha+1)*np.exp(-ratio**beta) )*dL
    #And the single-satellite probability distribution
    psat = phist*ratio**(alpha+1)*np.exp(-ratio**beta)/Ns
    if use_sch_fac:
        psat = psat*sch_fac

    #The probability that a single satellite is dimmer than Lbins
    p_dim = 0*Lbins
    for i in range(len(p_dim)):
        p_dim[i] = 1 - np.sum(psat[i:])*dL

    #Note we need to reset the luminosity bin positions to left edges
    Lbins = Lbins - dL/2.


    nvals = np.array(range(600))
    pvals = np.zeros([len(nvals),len(Lbins)])
    ptot = np.copy(pvals)
    
    p = 0.
    for i in range(len(nvals)):
        if i < 100:
            if i == 0:
                pvals[i,:] = 0*Lbins#+np.exp( -(Lbins - L0 - r*sigma_L*np.sqrt(Ns)*np.log10(nvals[i]/Ns) )**2/2./sigma_L/sigma_L/(1-r*r) )
            else:
                pvals[i,:] = (Ns**nvals[i]/np.math.factorial(nvals[i]))*np.exp(-Ns)*(1-p_dim**nvals[i])*np.exp( -(Lbins - L0 - r*sigma_L*np.sqrt(Ns)*np.log10(nvals[i]/Ns) )**2/2./sigma_L/sigma_L/(1-r*r) )
            
            if do_output:
                print i, Ns, (1-p_dim**nvals[i])
        else:
            #Switch to stirling's approximation
            pvals[i,:] = (Ns/nvals[i])**nvals[i]/np.sqrt(2*np.pi*nvals[i])*np.exp(nvals[i]-Ns)*(1-p_dim**nvals[i])*np.exp( -(Lbins - L0 - r*sigma_L*np.sqrt(Ns)*np.log10(nvals[i]/Ns) )**2/2./sigma_L/sigma_L/(1-r*r) )

    #p = np.sum(pvals)/np.sqrt(2*np.pi*(1-r*r))/sigma_L*dL
    p = np.sum(pvals)/np.sqrt(2*np.pi*(1-r*r))/sigma_L*dL

    return p, pvals

#Wrapper portion
def get_p_bcg_sch_corr(param_cen,param_sat,mhalo=2.35e14,z=0.3,r=None,Lmin=9.,beta=1.,use_sch_fac=False,do_output=False):
    Mpiv = 2.35e14
    sigma_L = param_cen[0]/np.log(10.)
    L0 = (param_cen[2] + param_cen[3]*np.log(mhalo/Mpiv)+param_cen[4]*np.log(1.+z))/np.log(10.)


    Ls = (param_sat[2] + param_sat[3]*np.log(mhalo/Mpiv)+param_sat[4]*np.log(1.+z))/np.log(10.)
    phist = np.exp(param_sat[0] + param_sat[1]*np.log(mhalo/Mpiv) + param_sat[6]*np.log(1+z))
    alpha = param_sat[5]

    if r == None:
        r = param_cen[1]
    p, pvals = get_p_bcg_sch_corr_single(L0,sigma_L,Ls,phist,alpha,r,Lmin=Lmin,beta=beta,use_sch_fac=use_sch_fac,do_output=do_output)
    return p

#Quick plotting function for current main plot of interest
def plot_pbcg(param_cen, param_sat):
    #Set up for test (Skibba+Sheth) version
    mtest = 13.+np.array(range(200))*0.01
    Ltest = np.log10(10.**((mtest-11.07)*3.273)*10.**9.935/(1+10.**(mtest-11.07))**3.018)
    atest = -2+0.501*(1-2./np.pi*np.arctan(2.106*(mtest-12.)))
    phtest = 10.**(-0.766 + 1.008*(mtest-12)-0.094*(mtest-12.)**2)

    p = np.zeros_like(Ltest)
    p_s82 = np.copy(p)
    p_dr8 = np.copy(p)

    #Calculate pbcg for Skibba+Sheth version
    for i in range(len(Ltest)):
        p[i] = get_p_bcg_cen_basic(Ltest[i],0.14,Ltest[i]-0.25,phtest[i],atest[i]+1)

    #Getting the DR8 fits + results
    indir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10_cencorr/"
    for i in range(len(mtest)):
        #p_dr8[i] = get_p_bcg_cen(param_cen,param_sat,z=0.3,mhalo=10.**mtest[i],phi_ev=True)
        p_dr8[i] = get_p_bcg_sch_corr(param_cen,param_sat,z=0.3,mhalo=10.**mtest[i])
    '''
    p_dr8_temp = np.zeros([nsamples,len(mtest)])
    for i in range(nsamples):
        for j in range(len(mtest)):
            p_dr8_temp[i,j] = get_p_bcg_cen(chain_cen[-1-i],chain_sat[-1-i],z=0.3,mhalo=10.**mtest[j],phi_ev=True)
    print "Done with DR8 loop."
    p_dr8_hi = np.percentile(p_dr8_temp,84,axis=0)
    p_dr8_lo = np.percentile(p_dr8_temp,16,axis=0)
    '''

    #Reading in the measured version from S82
    dat1 = np.loadtxt("/u/ki/rmredd/data/redmapper/s82_v5.10_uber/pbcg_cen_z_0.1_0.2.dat")
    dat2 = np.loadtxt("/u/ki/rmredd/data/redmapper/s82_v5.10_uber/pbcg_cen_z_0.2_0.3.dat")
    dat3 = np.loadtxt("/u/ki/rmredd/data/redmapper/s82_v5.10_uber/pbcg_cen_z_0.3_0.4.dat")
    dat4 = np.loadtxt("/u/ki/rmredd/data/redmapper/s82_v5.10_uber/pbcg_cen_z_0.4_0.5.dat")
    dat5 = np.loadtxt("/u/ki/rmredd/data/redmapper/s82_v5.10_uber/pbcg_cen_z_0.5_0.6.dat")
    dat_tot = dat1+dat2+dat3+dat4+dat5
    dat_tot[:,0] = np.copy(dat1[:,0])
    
    bin_edge = np.array([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,40,50,60,70,100.])
    lmbin = np.zeros(len(bin_edge)-1)
    pbcg = 0*lmbin
    pcount = 0*lmbin
    perr = 0*lmbin
    for i in range(len(pbcg)):
        clist = np.where( (dat_tot[:,0] >= bin_edge[i]) & (dat_tot[:,0] < bin_edge[i+1]))[0]
        pbcg[i] = np.sum(dat_tot[clist,2])/np.sum(dat_tot[clist,1])
        pcount[i] = np.sum(dat_tot[clist,1])
        perr[i] = np.sqrt( np.sum(dat_tot[clist,2])/np.sum(dat_tot[clist,1])**2 + 
                           np.sum(dat_tot[clist,2])**2/np.sum(dat_tot[clist,1])**3 )
        lmbin[i] = np.sum(dat_tot[clist,1]*dat_tot[clist,0])/np.sum(dat_tot[clist,1])
    mass_lm = np.log10( (lmbin/1.3**1.547/np.exp(2.7226))**(1./0.857)*2.35e14)
    print mass_lm

    #Reading in the SHAM results
    dat_bl = np.loadtxt("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/test_galaxies/pbcg_sham_blanton.dat")
    dat_ber = np.loadtxt("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/test_galaxies/pbcg_sham_bernardi.dat")
    meanmass = (dat_bl[:,0]+dat_bl[:,1])/2.
        
    pyplot.plot(mtest,p,'k')
    #pyplot.plot(mtest,p_dr8,'b')
    #pyplot.plot(mtest,p_dr8_hi,'b--')
    #pyplot.plot(mtest,p_dr8_lo,'b--')
    #pyplot.plot(mtest,p_s82,'r')
    #pyplot.plot(mtest,p_s82_hi,'r--')
    #pyplot.plot(mtest,p_s82_lo,'r--')
    pyplot.plot(mass_lm,pbcg,'go-')
    pyplot.errorbar(mass_lm,pbcg,perr,fmt=None,ecolor='g')
    pyplot.plot(meanmass,dat_bl[:,2],'co-')
    pyplot.plot(meanmass,dat_ber[:,2],'mo-')
    pyplot.xlabel('log(Mvir)',fontsize=20)
    pyplot.ylabel('P(BCG!=cen)',fontsize=20)

    return

def plot_bcg_s82(do_errors=True):
    #Set up for test (Skibba+Sheth) version
    mtest = 13.+np.array(range(220))*0.01
    Ltest = np.log10(10.**((mtest-11.07)*3.273)*10.**9.935/(1+10.**(mtest-11.07))**3.018)
    atest = -2+0.501*(1-2./np.pi*np.arctan(2.106*(mtest-12.)))
    phtest = 10.**(-0.766 + 1.008*(mtest-12)-0.094*(mtest-12.)**2)

    p = np.zeros_like(Ltest)
    p_s82 = np.copy(p)

    #Calculate pbcg for Skibba+Sheth version
    for i in range(len(Ltest)):
        p[i] = get_p_bcg_cen_basic(Ltest[i],0.14,Ltest[i]-0.25,phtest[i],atest[i]+1)

    #Getting the S82 fits + results
    indir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_s82_v5.10/"
    chain_cen = np.loadtxt(indir+"chain_cen_all.dat")
    param_cen = chain_cen[np.argmax(chain_cen[:,-1])]
    chain_sat = np.loadtxt(indir+"chain_sat_ev_all.dat")
    param_sat = chain_sat[np.argmax(chain_sat[:,-1])]
    #zvals = [0.15, 0.25, 0.35, 0.45, 0.55]
    zvals = [0.25,0.55]
    nzbins = len(zvals)
    p_s82 = np.zeros([nzbins,len(mtest)])
    for i in range(len(mtest)):
        for j in range(nzbins):
            #p_s82[j,i] = get_p_bcg_corr_gen_fit(param_cen,param_sat,z=zvals[j],mhalo=10.**mtest[i],phi_ev=True)
            p_s82[j,i] = get_p_bcg_sch_corr(param_cen,param_sat,z=zvals[j],mhalo=10.**mtest[i])
    #Get 100 sample results to get error ranges for our model
    if do_errors:
        print "Starting S82 loop..."
        nsamples = 100;
        p_s82_temp = np.zeros([nsamples,nzbins,len(mtest)])
        for i in range(nsamples):
            for j in range(len(mtest)):
                for k in range(nzbins):
                    p_s82_temp[i,k,j] = get_p_bcg_corr_gen_fit(chain_cen[-1-i],chain_sat[-1-i],z=zvals[k],mhalo=10.**mtest[j],phi_ev=True)
        print "Done with S82 loop."
        p_s82_hi = np.percentile(p_s82_temp,84,axis=0)
        p_s82_lo = np.percentile(p_s82_temp,16,axis=0)
    

    #Reading in the measured version from S82
    #dat1 = np.loadtxt("/u/ki/rmredd/data/redmapper/s82_v5.10_uber/pbcg_cen_z_0.1_0.2.dat")
    dat2 = np.loadtxt("/u/ki/rmredd/data/redmapper/s82_v5.10_uber/pbcg_cen_z_0.2_0.3.dat")
    #dat3 = np.loadtxt("/u/ki/rmredd/data/redmapper/s82_v5.10_uber/pbcg_cen_z_0.3_0.4.dat")
    #dat4 = np.loadtxt("/u/ki/rmredd/data/redmapper/s82_v5.10_uber/pbcg_cen_z_0.4_0.5.dat")
    dat5 = np.loadtxt("/u/ki/rmredd/data/redmapper/s82_v5.10_uber/pbcg_cen_z_0.5_0.6.dat")
    #dat_set = [dat1, dat2, dat3, dat4, dat5]
    dat_set = [dat2,dat5]
    
    bin_edge = np.array([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,40,50,60,70,100.])
    lmbin = np.zeros(len(bin_edge)-1)
    pbcg = np.zeros([nzbins,len(lmbin)])
    pcount = 0*lmbin
    perr = np.zeros([nzbins,len(lmbin)])
    mass_lm = np.zeros([nzbins,len(lmbin)])
    for j in range(nzbins):
        for i in range(len(lmbin)):
            clist = np.where( (dat_set[j][:,0] >= bin_edge[i]) & (dat_set[j][:,0] < bin_edge[i+1]))[0]
            pbcg[j,i] = np.sum(dat_set[j][clist,2])/np.sum(dat_set[j][clist,1])
            pcount[i] = np.sum(dat_set[j][clist,1])
            perr[j,i] = np.sqrt( np.sum(dat_set[j][clist,2])/np.sum(dat_set[j][clist,1])**2 + 
                               np.sum(dat_set[j][clist,2])**2/np.sum(dat_set[j][clist,1])**3 )
            lmbin[i] = np.sum(dat_set[j][clist,1]*dat_set[j][clist,0])/np.sum(dat_set[j][clist,1])
        mass_lm[j] = np.log10( (lmbin/(1.+zvals[j])**1.547/np.exp(2.7226))**(1./0.857)*2.35e14)
    #print mass_lm[2], pbcg[2], dat_set[2][:,0],dat_set[1][:,1]
        
    #Reading in the SHAM results
    dat_bl = np.loadtxt("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/test_galaxies/pbcg_sham_blanton.dat")
    dat_ber = np.loadtxt("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/test_galaxies/pbcg_sham_bernardi.dat")
    meanmass = (dat_bl[:,0]+dat_bl[:,1])/2.
        
    colorlist = ['b','r','g','purple','sienna']

    pyplot.plot(mtest,p,'k')
    for i in range(nzbins):
        pyplot.plot(mtest,p_s82[i],color=colorlist[i])
        if do_errors:
            pyplot.plot(mtest,p_s82_hi[i],color=colorlist[i],linestyle='--')
            pyplot.plot(mtest,p_s82_lo[i],color=colorlist[i],linestyle='--')
        pyplot.plot(mass_lm[i],pbcg[i],color=colorlist[i],ls='None',marker='^')
        pyplot.errorbar(mass_lm[i],pbcg[i],perr[i],fmt=None,ecolor=colorlist[i])

    pyplot.plot(meanmass,dat_bl[:,2],'co-')
    pyplot.plot(meanmass,dat_ber[:,2],'mo-')
    pyplot.xlabel('log(Mvir)')
    pyplot.ylabel('P(BCG!=cen)')
    pyplot.ylim([0,0.8])

    return

#A couple of functions for the sole purpose of sanity checking
def galaxies_expected(L0, sigma_L, Lst, phist, alpha, Lmin):
    '''
    Calculates expected number of satellite/fraction of centrals above
    some fixed input luminosity
    '''

    ncen = 0.5 - 0.5*scipy.special.erf( (Lmin-L0)/(np.sqrt(2.)*sigma_L) )

    nsat = phist/np.log(10.)*gamma_neg(alpha+1,10.**Lmin/10.**Lst)

    return [ncen, nsat]

def galaxies_expected_fit(param_cen, param_sat, Lmin=-1, mhalo=2.35e14, z=0.3):
    '''
    Calculates expected number of satellites/fraction of centrals above 
    some fixed input luminosity ; uses input of redshift, mass and fitted
    parameters to calculate this
    '''

    Mpiv = 2.35e14

    sigma_L = param_cen[0]/np.log(10.)
    L0 = (param_cen[2] + param_cen[3]*np.log(mhalo/Mpiv)+param_cen[4]*np.log(1.+z))/np.log(10.)
    phist = np.exp(param_sat[0] + param_sat[1]*np.log(mhalo/Mpiv))
    Lst = (param_sat[2] + param_sat[3]*np.log(mhalo/Mpiv)+param_sat[4]*np.log(1.+z))/np.log(10.)
    alpha = param_sat[5]

    if Lmin == -1:
        Lmin = L0 - 2*sigma_L

    ncen, nsat = galaxies_expected(L0, sigma_L, Lst, phist, alpha, Lmin)

    return [ncen, nsat]


#Different function entirely -- calculation of the distribution of the brightest satellite
#Given schechter input parameters
def brightest_satellite_distr(Lst, phist, alpha):
    dL = 0.02
    lumbins = 9+np.array(range(150))*dL

    bsat = np.exp(-phist/np.log(10.)*gamma_neg(alpha+1,10.**(lumbins-Lst)))*10.**( (alpha+1)*( lumbins-Lst) )*np.exp(-10.**(lumbins-Lst))*phist

    #Normalization
    norm = np.sum(bsat)
    bsat = bsat/norm/dL

    return lumbins, bsat

def plot_bcg_dr8(param_cen,param_sat,err_cen,err_sat,do_errors=True):
    #Set up for test (Skibba+Sheth) version
    mtest = 13.+np.array(range(220))*0.01
    Ltest = np.log10(10.**((mtest-11.07)*3.273)*10.**9.935/(1+10.**(mtest-11.07))**3.018)
    atest = -2+0.501*(1-2./np.pi*np.arctan(2.106*(mtest-12.)))
    phtest = 10.**(-0.766 + 1.008*(mtest-12)-0.094*(mtest-12.)**2)

    p = np.zeros_like(Ltest)
    p_dr8 = np.copy(p)

    #Calculate pbcg for Skibba+Sheth version
    for i in range(len(Ltest)):
        p[i] = get_p_bcg_cen_basic(Ltest[i],0.14,Ltest[i]-0.25,phtest[i],atest[i]+1)

    #Fits and results -- currently hard coded because I'm lazy
    #param_cen = [0.363667675133445, -0.52335780572853252, 25.071104716809213, 0.37294772175919816, 1.1170727503329549, 1.2370806203231806]
    #param_sat = [3.9438395962864941, 0.64685415687462167, 0.84804482226676792, -0.25238394246331147, 23.523841845508873, 1.4333379734881324, 0.049778387652138142, 0.12632251065613598, -0.82164440867739408, 0.19605885744113272, 1.1597685580868327]
    #err_cen = [[ 0.01123446,  0.01148164], [ 0.1008863 ,  0.11445454], [ 0.0206819 ,  0.02246899], [ 0.00721075,  0.00704131], [ 0.20869451,  0.19020458], [ 0.08093765,  0.0666278 ]]
    #err_sat = [[ 0.0065253 ,  0.00659084], [ 0.08016366,  0.07851143], [ 0.00386201,  0.00353705], [ 0.04101625,  0.04090862], [ 0.00611387,  0.00566172], [ 0.08161472,  0.07379233], [ 0.00383233,  0.00424895], [ 0.04768194,  0.04635835], [ 0.00630591,  0.00696978], [ 0.07277073,  0.06769767], [ 0.05253671,  0.06429355]]

    zvals = [0.125,0.315]
    nzbins = len(zvals)
    p_dr8 = np.zeros([nzbins,len(mtest)])
    for i in range(len(mtest)):
        for j in range(nzbins):
            param_cen_temp = [param_cen[0], param_cen[1],param_cen[2]+param_cen[4]*np.log((zvals[j]+1)/1.3),param_cen[3],0.]
            param_sat_temp = [param_sat[0]+param_sat[1]*np.log((1+zvals[j])/1.3), 
                              param_sat[2]+param_sat[3]*np.log((1+zvals[j])/1.3), 
                              param_sat[4]+param_sat[5]*np.log((1+zvals[j])/1.3), 
                              param_sat[6]+param_sat[7]*np.log((1+zvals[j])/1.3), 
                              0., 
                              param_sat[8]+param_sat[9]*np.log((1+zvals[j])/1.3), 
                              0.]
            p_dr8[j,i] = get_p_bcg_sch_corr(param_cen_temp,param_sat_temp,z=zvals[j],mhalo=10.**mtest[i],beta=1.,use_sch_fac=False)

    if do_errors:
        #Make the error limits somehow
        p_dr8_hi = 0*p_dr8
        p_dr8_lo = 0*p_dr8
        for i in range(len(mtest)):
            for j in range(nzbins):
                param_cen_temp = np.array([param_cen[0], param_cen[1],
                                  param_cen[2]+param_cen[4]*np.log((zvals[j]+1)/1.3),param_cen[3],0.])
                param_sat_temp = np.array([param_sat[0]+param_sat[1]*np.log((1+zvals[j])/1.3), 
                                  param_sat[2]+param_sat[3]*np.log((1+zvals[j])/1.3), 
                                  param_sat[4]+param_sat[5]*np.log((1+zvals[j])/1.3), 
                                  param_sat[6]+param_sat[7]*np.log((1+zvals[j])/1.3), 
                                  0., 
                                  param_sat[8]+param_sat[9]*np.log((1+zvals[j])/1.3), 
                                  0.])
                p_dr8_hi[j,i] = get_p_bcg_sch_corr(param_cen_temp + np.array([err_cen[0][0], 0., err_cen[2][0], err_cen[3][0], 0]),
                                                   param_sat_temp - np.array([err_sat[0][1], err_sat[2][1], err_sat[4][1], err_sat[6][1], 0., 0., 0]),z=zvals[j],mhalo=10.**mtest[i],beta=1.,use_sch_fac=False)
                p_dr8_lo[j,i] = get_p_bcg_sch_corr(param_cen_temp - np.array([err_cen[0][1], 0., err_cen[2][1], err_cen[3][1], 0]),
                                                   param_sat_temp + np.array([err_sat[0][0], err_sat[2][0], err_sat[4][0], err_sat[6][0], 0., 0., 0]),z=zvals[j],mhalo=10.**mtest[i],beta=1.,use_sch_fac=False)
        
    #Read in measured DR8 data
    dat1 = np.loadtxt("/u/ki/rmredd/data/redmapper/dr8_zlambda_v5.10_cencorr/pbcg_cen_z_0.1_0.15.dat")
    #dat2 = np.loadtxt("/u/ki/rmredd/data/redmapper/dr8_zlambda_v5.10_cencorr/pbcg_cen_z_0.15_0.2.dat")
    #dat3 = np.loadtxt("/u/ki/rmredd/data/redmapper/dr8_zlambda_v5.10_cencorr/pbcg_cen_z_0.2_0.25.dat")
    #dat4 = np.loadtxt("/u/ki/rmredd/data/redmapper/dr8_zlambda_v5.10_cencorr/pbcg_cen_z_0.25_0.3.dat")
    dat5 = np.loadtxt("/u/ki/rmredd/data/redmapper/dr8_zlambda_v5.10_cencorr/pbcg_cen_z_0.3_0.33.dat")
    dat_set = [dat1, dat5]

    bin_edge = np.array([5,6,7,8,9,10,12,14,16,20,25,30,40,50,60,70,100.])
    lmbin = np.zeros(len(bin_edge)-1)
    pbcg = np.zeros([nzbins,len(lmbin)])
    pcount = 0*lmbin
    perr = np.zeros([nzbins,len(lmbin)])
    mass_lm = np.zeros([nzbins,len(lmbin)])
    for j in range(nzbins):
        for i in range(len(lmbin)):
            clist = np.where( (dat_set[j][:,0] >= bin_edge[i]) & (dat_set[j][:,0] < bin_edge[i+1]))[0]
            pbcg[j,i] = np.sum(dat_set[j][clist,2])/np.sum(dat_set[j][clist,1])
            pcount[i] = np.sum(dat_set[j][clist,1])
            perr[j,i] = np.sqrt( np.sum(dat_set[j][clist,2])/np.sum(dat_set[j][clist,1])**2 + 
                               np.sum(dat_set[j][clist,2])**2/np.sum(dat_set[j][clist,1])**3 )
            lmbin[i] = np.sum(dat_set[j][clist,1]*dat_set[j][clist,0])/np.sum(dat_set[j][clist,1])
        mass_lm[j] = np.log10( (lmbin/(1.+zvals[j])**1.547/np.exp(2.7226))**(1./0.857)*2.35e14)

    #Reading in the SHAM results
    dat_bl = np.loadtxt("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/test_galaxies/pbcg_sham_blanton.dat")
    dat_ber = np.loadtxt("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/test_galaxies/pbcg_sham_bernardi.dat")
    meanmass = (dat_bl[:,0]+dat_bl[:,1])/2.
    colorlist = ['b','r','g','purple','sienna']
    pyplot.plot(mtest,p,'k')
    for i in range(nzbins):
        pyplot.plot(mtest,p_dr8[i],color=colorlist[i])
        if do_errors:
            pyplot.plot(mtest,p_dr8_hi[i],color=colorlist[i],linestyle='--')
            pyplot.plot(mtest,p_dr8_lo[i],color=colorlist[i],linestyle='--')
        pyplot.plot(mass_lm[i],pbcg[i],color=colorlist[i],ls='None',marker='^')
        pyplot.errorbar(mass_lm[i],pbcg[i],perr[i],fmt=None,ecolor=colorlist[i])

    pyplot.plot(meanmass,dat_bl[:,2],'co-')
    pyplot.plot(meanmass,dat_ber[:,2],'mo-')
    pyplot.xlabel(r'log(Mvir) $[log(M_\odot)$]',fontsize=20)
    pyplot.ylabel('P(BCG!=cen)',fontsize=20)
    pyplot.ylim([0,0.8])

    return

def plot_bcg_dr8_old(do_errors=True,use_beta=False,use_sch_fac=False,cen_test=False):
    #Set up for test (Skibba+Sheth) version
    mtest = 13.+np.array(range(220))*0.01
    Ltest = np.log10(10.**((mtest-11.07)*3.273)*10.**9.935/(1+10.**(mtest-11.07))**3.018)
    atest = -2+0.501*(1-2./np.pi*np.arctan(2.106*(mtest-12.)))
    phtest = 10.**(-0.766 + 1.008*(mtest-12)-0.094*(mtest-12.)**2)

    p = np.zeros_like(Ltest)
    p_dr8 = np.copy(p)

    #Calculate pbcg for Skibba+Sheth version
    for i in range(len(Ltest)):
        p[i] = get_p_bcg_cen_basic(Ltest[i],0.14,Ltest[i]-0.25,phtest[i],atest[i]+1)

    #Getting the S82 fits + results
    indir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10_v2/"
    indir_beta = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10_schcorr/"

    chain_cen = np.loadtxt(indir+"chain_cen_all.dat")
    param_cen = chain_cen[np.argmax(chain_cen[:,-1])]
    if cen_test:
        param_cen[2] = param_cen[2] - 0.08*np.log(10)
        param_cen[3] = 0.22
    
    if use_sch_fac:
        indir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_dr8_v5.10_sch_empcorr/"
    if use_beta:
        chain_sat = np.loadtxt(indir_beta+"chain_sat_ev_all.dat")
    else:
        chain_sat = np.loadtxt(indir+"chain_sat_ev_all.dat")
    param_sat = chain_sat[np.argmax(chain_sat[:,-1])]
    print param_sat
    zvals = [0.125,0.315]
    nzbins = len(zvals)
    p_dr8 = np.zeros([nzbins,len(mtest)])
    beta = 1.
    if use_beta:
            beta = param_sat[7]
    for i in range(len(mtest)):
        for j in range(nzbins):
            p_dr8[j,i] = get_p_bcg_sch_corr(param_cen,param_sat,z=zvals[j],mhalo=10.**mtest[i],beta=beta,use_sch_fac=use_sch_fac)
    #Get 100 sample results to get error ranges for our model
    if do_errors:
        print "Starting DR8 loop..."
        nsamples = 100
        p_dr8_temp = np.zeros([nsamples,nzbins,len(mtest)])
        for i in range(nsamples):
            print i
            for j in range(len(mtest)):
                for k in range(nzbins):
                    p_dr8_temp[i,k,j] = get_p_bcg_sch_corr(chain_cen[-1-i],chain_sat[-1-i],z=zvals[k],mhalo=10.**mtest[j],beta=beta)
        print "Done with S82 loop."
        p_dr8_hi = np.percentile(p_dr8_temp,84,axis=0)
        p_dr8_lo = np.percentile(p_dr8_temp,16,axis=0)
    

    #Reading in the measured version from DR8
    dat1 = np.loadtxt("/u/ki/rmredd/data/redmapper/dr8_zlambda_v5.10/pbcg_cen_z_0.1_0.15.dat")
    #dat2 = np.loadtxt("/u/ki/rmredd/data/redmapper/dr8_zlambda_v5.10/pbcg_cen_z_0.15_0.2.dat")
    #dat3 = np.loadtxt("/u/ki/rmredd/data/redmapper/dr8_zlambda_v5.10/pbcg_cen_z_0.2_0.25.dat")
    #dat4 = np.loadtxt("/u/ki/rmredd/data/redmapper/dr8_zlambda_v5.10/pbcg_cen_z_0.25_0.3.dat")
    dat5 = np.loadtxt("/u/ki/rmredd/data/redmapper/dr8_zlambda_v5.10/pbcg_cen_z_0.3_0.33.dat")
    #dat_set = [dat1, dat2, dat3, dat4, dat5]
    dat_set = [dat1,  dat5]
    
    #bin_edge = np.array([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,40,50,60,70,100.])
    bin_edge = np.array([5,6,7,8,9,10,12,14,16,20,25,30,40,50,60,70,100.])
    lmbin = np.zeros(len(bin_edge)-1)
    pbcg = np.zeros([nzbins,len(lmbin)])
    pcount = 0*lmbin
    perr = np.zeros([nzbins,len(lmbin)])
    mass_lm = np.zeros([nzbins,len(lmbin)])
    for j in range(nzbins):
        for i in range(len(lmbin)):
            clist = np.where( (dat_set[j][:,0] >= bin_edge[i]) & (dat_set[j][:,0] < bin_edge[i+1]))[0]
            pbcg[j,i] = np.sum(dat_set[j][clist,2])/np.sum(dat_set[j][clist,1])
            pcount[i] = np.sum(dat_set[j][clist,1])
            perr[j,i] = np.sqrt( np.sum(dat_set[j][clist,2])/np.sum(dat_set[j][clist,1])**2 + 
                               np.sum(dat_set[j][clist,2])**2/np.sum(dat_set[j][clist,1])**3 )
            lmbin[i] = np.sum(dat_set[j][clist,1]*dat_set[j][clist,0])/np.sum(dat_set[j][clist,1])
        mass_lm[j] = np.log10( (lmbin/(1.+zvals[j])**1.547/np.exp(2.7226))**(1./0.857)*2.35e14)
    print mass_lm[0], bin_edge,pcount 
    #print mass_lm[2], pbcg[2], dat_set[2][:,0],dat_set[1][:,1]
        
    #Reading in the SHAM results
    dat_bl = np.loadtxt("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/test_galaxies/pbcg_sham_blanton.dat")
    dat_ber = np.loadtxt("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/test_galaxies/pbcg_sham_bernardi.dat")
    meanmass = (dat_bl[:,0]+dat_bl[:,1])/2.
        
    colorlist = ['b','r','g','purple','sienna']

    pyplot.plot(mtest,p,'k')
    for i in range(nzbins):
        pyplot.plot(mtest,p_dr8[i],color=colorlist[i])
        if do_errors:
            pyplot.plot(mtest,p_dr8_hi[i],color=colorlist[i],linestyle='--')
            pyplot.plot(mtest,p_dr8_lo[i],color=colorlist[i],linestyle='--')
        pyplot.plot(mass_lm[i],pbcg[i],color=colorlist[i],ls='None',marker='^')
        pyplot.errorbar(mass_lm[i],pbcg[i],perr[i],fmt=None,ecolor=colorlist[i])

    pyplot.plot(meanmass,dat_bl[:,2],'co-')
    pyplot.plot(meanmass,dat_ber[:,2],'mo-')
    pyplot.xlabel(r'log(Mvir) $[log(M_\odot)$]')
    pyplot.ylabel('P(BCG!=cen)')
    pyplot.ylim([0,0.8])

    return


#P(BNC) plotting routine for SVA1
def plot_bcg_sva(do_errors=True):    #Set up for test (Skibba+Sheth) version
    mtest = 13.+np.array(range(220))*0.01
    Ltest = np.log10(10.**((mtest-11.07)*3.273)*10.**9.935/(1+10.**(mtest-11.07))**3.018)
    atest = -2+0.501*(1-2./np.pi*np.arctan(2.106*(mtest-12.)))
    phtest = 10.**(-0.766 + 1.008*(mtest-12)-0.094*(mtest-12.)**2)

    p = np.zeros_like(Ltest)
    p_sva = np.copy(p)

    #Calculate pbcg for Skibba+Sheth version
    for i in range(len(Ltest)):
        p[i] = get_p_bcg_cen_basic(Ltest[i],0.14,Ltest[i]-0.25,phtest[i],atest[i]+1)

    #Getting the S82 fits + results
    indir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/fits_plm_full_sva1_v6.1.3/"

    chain_cen = np.loadtxt(indir+"chain_cen_all.dat")
    param_cen = chain_cen[np.argmax(chain_cen[:,-1])]

    chain_sat = np.loadtxt(indir+"chain_sat_ev_all.dat")
    param_sat = chain_sat[np.argmax(chain_sat[:,-1])]

    print param_sat
    zvals = [0.35, 0.85]
    nzbins = len(zvals)
    p_sva = np.zeros([nzbins,len(mtest)])
    
    for i in range(len(mtest)):
        for j in range(nzbins):
            p_sva[j,i] = get_p_bcg_sch_corr(param_cen,param_sat,z=zvals[j],mhalo=10.**mtest[i],beta=1.0,use_sch_fac=False)

    if do_errors:
        print "Starting SVA1 loop..."
        nsamples = 50
        p_sva_temp = np.zeros([nsamples,nzbins,len(mtest)])
        for i in range(nsamples):
            print i
            for j in range(len(mtest)):
                for k in range(nzbins):
                    p_sva_temp[i,k,j] = get_p_bcg_sch_corr(chain_cen[-1-i],chain_sat[-1-i],z=zvals[k],mhalo=10.**mtest[j],beta=1.0)
        print "Done with SVA loop"
        p_sva_hi = np.percentile(p_sva_temp,84,axis=0)
        p_sva_lo = np.percentile(p_sva_temp,16,axis=0)

    dat1 = np.loadtxt("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/sv1_gold_v6.1.3_uber/pbcg_cen_z_0.3_0.4.dat")
    dat2 = np.loadtxt("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/sv1_gold_v6.1.3_uber/pbcg_cen_z_0.8_0.9.dat")
    dat_set = [dat1, dat2]
    

    bin_edge = np.array([5,6,7,8,9,10,12,14,16,20,25,30,40,50,60,70,100.])
    lmbin = np.zeros(len(bin_edge)-1)
    pbcg = np.zeros([nzbins,len(lmbin)])
    pcount = 0*lmbin
    perr = np.zeros([nzbins,len(lmbin)])
    mass_lm = np.zeros([nzbins,len(lmbin)])
    for j in range(nzbins):
        for i in range(len(lmbin)):
            clist = np.where( (dat_set[j][:,0] >= bin_edge[i]) & (dat_set[j][:,0] < bin_edge[i+1]))[0]
            pbcg[j,i] = np.sum(dat_set[j][clist,2])/np.sum(dat_set[j][clist,1])
            pcount[i] = np.sum(dat_set[j][clist,1])
            perr[j,i] = np.sqrt( np.sum(dat_set[j][clist,2])/np.sum(dat_set[j][clist,1])**2 + 
                               np.sum(dat_set[j][clist,2])**2/np.sum(dat_set[j][clist,1])**3 )
            lmbin[i] = np.sum(dat_set[j][clist,1]*dat_set[j][clist,0])/np.sum(dat_set[j][clist,1])
        mass_lm[j] = np.log10( (lmbin/(1.+zvals[j])**1.547/np.exp(2.7226))**(1./0.857)*2.35e14)

    #Reading in the SHAM results
    dat_bl = np.loadtxt("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/test_galaxies/pbcg_sham_blanton.dat")
    dat_ber = np.loadtxt("/nfs/slac/g/ki/ki10/rmredd/redmapper_data/test_galaxies/pbcg_sham_bernardi.dat")
    meanmass = (dat_bl[:,0]+dat_bl[:,1])/2.

    colorlist = ['b','r','g','purple','sienna']
    marklist = ['v','^','x']
    
    #pyplot.plot(mtest,p,'k')
    for i in range(nzbins):
        pyplot.plot(mtest,p_sva[i],color=colorlist[i])
        if do_errors:
            pyplot.plot(mtest,p_sva_hi[i],color=colorlist[i],linestyle='--')
            pyplot.plot(mtest,p_sva_lo[i],color=colorlist[i],linestyle='--')
        pyplot.plot(mass_lm[i],pbcg[i],color=colorlist[i],ls='None',marker=marklist[i])
        pyplot.errorbar(mass_lm[i],pbcg[i],perr[i],fmt=None,ecolor=colorlist[i])

    pyplot.plot(meanmass,dat_bl[:,2],'mo-')
    pyplot.plot(meanmass,dat_ber[:,2],'ko-')
    pyplot.xlabel(r'log(Mvir) $[log(M_\odot)$]')
    pyplot.ylabel('P(BCG!=cen)')
    pyplot.ylim([0,0.8])

    return


#Item to submit to queue in order to get stupid long plots to plot...
if __name__ == "__main__":
    outdir = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/plots_main_dr8_v5.10/"
    
    plot_bcg_dr8()
    pyplot.savefig(outdir+"pbcg_comp_dr8.ps",orientation='landscape')

    outdir_s82 = "/nfs/slac/g/ki/ki10/rmredd/redmapper_data/plots_main_s82_v5.10/"
    
    pyplot.clf()
    plot_bcg_s82()
    pyplot.savefig(outdir_s82+"pbcg_comp_s82.ps",orientation='landscape')

