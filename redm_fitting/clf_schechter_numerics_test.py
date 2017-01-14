#/usr/bin/python

import numpy as np

#Tests of numerical summing of Schechter function -- check for bright-end noise
#Note draws in range of log(L/L*)=[-0.7,2]

#Draw a single galaxy from a Schechter function
def draw_single_galaxy(yrange=[-0.5,2],alpha=-0.85):

    sch_max = 10.**(yrange[0]*(alpha+1))*np.exp(-10.**yrange[0])
    p = 1.1
    L = np.copy(yrange[0])
    while p > 10.**(L*(alpha+1))*np.exp(-10.**L)/sch_max:
        L = np.random.random_sample()*(yrange[1]-yrange[0])+yrange[0]
        p = np.random.random_sample()

    return L

def sum_random_gals(ngals,alpha=-0.85,cutoff=1.):
    L = np.zeros(ngals)

    for i in range(ngals):
        L[i] = draw_single_galaxy(alpha=alpha)

    print "Done drawing galaxies"
    dL = 0.01
    lumbins = np.array(range(250))*dL-0.5+dL/2.
    weight = 10.**(L*(alpha+1))*np.exp(-10.**L)/np.sum( 10.**(lumbins*(-0.85+1))*np.exp(-10.**lumbins) )/0.01
    cut_val = np.random.normal(loc=1.,scale=0.1,size=ngals)
    weight = weight*cut_val
    L = L + np.random.normal(loc=0.,scale=0.01,size=ngals)

    clf = 0*lumbins
    clf_w = 0*lumbins
    clf_w_cut = 0*lumbins

    bin = np.floor((L+0.5)/dL)
    print np.max(L), np.max(bin), (np.max(L)+0.5)/dL, len(np.where(bin < 0)[0])

    for i in range(ngals):
        if bin[i] < 0:
            continue
        clf[bin[i]] = clf[bin[i]] + 1.
        clf_w[bin[i]] = clf_w[bin[i]] + 1./weight[i]
        if cut_val[i] > cutoff:
            clf_w_cut[bin[i]] = clf_w_cut[bin[i]] + 1./weight[i]

    clf = clf/ngals/dL
    clf_w = clf_w/ngals/dL
    clf_w_cut = clf_w_cut/len(np.where(cut_val > cutoff)[0])/dL
    print len(np.where(cut_val > cutoff)[0])

    return lumbins, clf, clf_w, clf_w_cut

def func_sat_test(lumbins,deviate=0.5,alpha=-0.85):
    clf = 10.**(lumbins*(alpha+1))*np.exp(-10.**lumbins)

    slope_d = (alpha+1)-10.**deviate

    clist = np.where(lumbins > deviate)[0]
    if len(clist) > 0:
        clf[clist] = 10.**( np.log10(10.**(deviate*(alpha+1))*np.exp(-10.**deviate)) + slope_d * (lumbins[clist]-deviate) - 10.**deviate/np.log(10)*(lumbins[clist]-deviate)**2/2. - 10.**deviate/np.log(10)**2*(lumbins[clist]-deviate)**3/6. )

    return clf
