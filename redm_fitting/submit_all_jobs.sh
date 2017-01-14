#!/bin/bash

#Main runs
#bsub -q kipac-ibq -oo batch_dr8_cen.log "python run_emcee_cen.py param_dr8_v6.2.dat"
bsub -q kipac-ibq -oo batch_s82_cen.log "python run_emcee_cen.py param_s82_v6.2.dat"

#bsub -q kipac-ibq -oo batch_dr8_sat.log "python run_emcee_sat_ev.py param_dr8_v6.2.dat"
bsub -q kipac-ibq -oo batch_s82_sat.log "python run_emcee_sat_ev.py param_s82_v6.2.dat"

#Individual redshift bins part
#bsub -q kipac-ibq -oo batch_dr8_cen_z1.log "python run_emcee_cen_noev.py param_dr8_v6.2_z1.dat"
#bsub -q kipac-ibq -oo batch_dr8_cen_z2.log "python run_emcee_cen_noev.py param_dr8_v6.2_z2.dat"
#bsub -q kipac-ibq -oo batch_dr8_cen_z3.log "python run_emcee_cen_noev.py param_dr8_v6.2_z3.dat"
#bsub -q kipac-ibq -oo batch_dr8_cen_z4.log "python run_emcee_cen_noev.py param_dr8_v6.2_z4.dat"
#bsub -q kipac-ibq -oo batch_dr8_cen_z5.log "python run_emcee_cen_noev.py param_dr8_v6.2_z5.dat"

#bsub -q kipac-ibq -oo batch_dr8_sat_z1.log "python run_emcee_sat_noev.py param_dr8_v6.2_z1.dat"
#bsub -q kipac-ibq -oo batch_dr8_sat_z2.log "python run_emcee_sat_noev.py param_dr8_v6.2_z2.dat"
#bsub -q kipac-ibq -oo batch_dr8_sat_z3.log "python run_emcee_sat_noev.py param_dr8_v6.2_z3.dat"
#bsub -q kipac-ibq -oo batch_dr8_sat_z4.log "python run_emcee_sat_noev.py param_dr8_v6.2_z4.dat"
#bsub -q kipac-ibq -oo batch_dr8_sat_z5.log "python run_emcee_sat_noev.py param_dr8_v6.2_z5.dat"

bsub -q kipac-ibq -oo batch_s82_sat_z1.log "python run_emcee_sat_noev.py param_s82_v6.2_z1.dat"
bsub -q kipac-ibq -oo batch_s82_sat_z2.log "python run_emcee_sat_noev.py param_s82_v6.2_z2.dat"
bsub -q kipac-ibq -oo batch_s82_sat_z3.log "python run_emcee_sat_noev.py param_s82_v6.2_z3.dat"
bsub -q kipac-ibq -oo batch_s82_sat_z4.log "python run_emcee_sat_noev.py param_s82_v6.2_z4.dat"
bsub -q kipac-ibq -oo batch_s82_sat_z5.log "python run_emcee_sat_noev.py param_s82_v6.2_z5.dat"

bsub -q kipac-ibq -oo batch_s82_cen_z1.log "python run_emcee_cen_noev.py param_s82_v6.2_z1.dat"
bsub -q kipac-ibq -oo batch_s82_cen_z2.log "python run_emcee_cen_noev.py param_s82_v6.2_z2.dat"
bsub -q kipac-ibq -oo batch_s82_cen_z3.log "python run_emcee_cen_noev.py param_s82_v6.2_z3.dat"
bsub -q kipac-ibq -oo batch_s82_cen_z4.log "python run_emcee_cen_noev.py param_s82_v6.2_z4.dat"
bsub -q kipac-ibq -oo batch_s82_cen_z5.log "python run_emcee_cen_noev.py param_s82_v6.2_z5.dat"