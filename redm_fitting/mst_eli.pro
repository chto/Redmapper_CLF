function mst_eli,zbins

mst = 0*zbins

zlist = where(zbins le 0.5, complement=alist)
mst[zlist] = 22.44 + 3.36*alog(zbins[zlist]) + 0.273*alog(zbins[zlist])^2$
             -0.0618*alog(zbins[zlist])^3 - 0.0227*alog(zbins[zlist])^4
mst[alist] = 22.94+3.08*alog(zbins[alist]) - 11.22*alog(zbins[alist])^2$
-27.11*alog(zbins[alist])^3-18.02*alog(zbins[alist])^4

return,mst
end

function mst_lum,zbins,mst_mag

  kcorrect,mst_mag,0*mst_mag+0.01,zbins,kcorr,band_shift=0.3,$
           /mag,/stddev,absmag=mst,filterlist=['DES_i.par'],vname='lrg1'

return,mst
end
