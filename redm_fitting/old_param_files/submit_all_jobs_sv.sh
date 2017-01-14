#!/bin/bash

#Main runs
#bsub -q kipac-ibq -oo batch_sva1_cen.log "python run_emcee_cen.py param_sv_sat.dat"

#bsub -q kipac-ibq -oo batch_sva1_sat.log "python run_emcee_sat_ev.py param_sv_sat.dat"

#Individual redshift bins part
bsub -q kipac-ibq -oo batch_sva1_cen_z1.log "python run_emcee_cen_noev.py param_sv_z1.dat"
bsub -q kipac-ibq -oo batch_sva1_cen_z2.log "python run_emcee_cen_noev.py param_sv_z2.dat"
bsub -q kipac-ibq -oo batch_sva1_cen_z3.log "python run_emcee_cen_noev.py param_sv_z3.dat"
bsub -q kipac-ibq -oo batch_sva1_cen_z4.log "python run_emcee_cen_noev.py param_sv_z4.dat"
bsub -q kipac-ibq -oo batch_sva1_cen_z5.log "python run_emcee_cen_noev.py param_sv_z5.dat"
bsub -q kipac-ibq -oo batch_sva1_cen_z6.log "python run_emcee_cen_noev.py param_sv_z6.dat"
bsub -q kipac-ibq -oo batch_sva1_cen_z7.log "python run_emcee_cen_noev.py param_sv_z7.dat"
bsub -q kipac-ibq -oo batch_sva1_cen_z8.log "python run_emcee_cen_noev.py param_sv_z8.dat"

bsub -q kipac-ibq -oo batch_sva1_sat_z1.log "python run_emcee_sat_noev.py param_sv_z1.dat"
bsub -q kipac-ibq -oo batch_sva1_sat_z2.log "python run_emcee_sat_noev.py param_sv_z2.dat"
bsub -q kipac-ibq -oo batch_sva1_sat_z3.log "python run_emcee_sat_noev.py param_sv_z3.dat"
bsub -q kipac-ibq -oo batch_sva1_sat_z4.log "python run_emcee_sat_noev.py param_sv_z4.dat"
bsub -q kipac-ibq -oo batch_sva1_sat_z5.log "python run_emcee_sat_noev.py param_sv_z5.dat"
bsub -q kipac-ibq -oo batch_sva1_sat_z6.log "python run_emcee_sat_noev.py param_sv_z6.dat"
bsub -q kipac-ibq -oo batch_sva1_sat_z7.log "python run_emcee_sat_noev.py param_sv_z7.dat"
bsub -q kipac-ibq -oo batch_sva1_sat_z8.log "python run_emcee_sat_noev.py param_sv_z8.dat"