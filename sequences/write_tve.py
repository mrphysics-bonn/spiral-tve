# Spiral Pulseq Sequence for TVE

#%%

import numpy as np
import matplotlib.pyplot as plt
import ismrmrd
import os
import stat
import datetime
from copy import copy
import subprocess
import re
import math
import random

from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.Sequence.sequence import Sequence
from pypulseq.make_adc import make_adc
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_gauss_pulse import make_gauss_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.make_delay import make_delay
from pypulseq.make_digital_output_pulse import make_digital_output_pulse
from pypulseq.opts import Opts
from pypulseq.calc_duration import calc_duration
from pypulseq.make_sigpy_pulse import sigpy_n_seq
from pypulseq.sigpy_pulse_opts import SigpyPulseOpts
from pypulseq.points_to_waveform import points_to_waveform

import spiraltraj
from sigpy.mri import rf as rfsig
import helper.pulseq_helper as ph
from helper.diffusion import diff_params, DISCSo, computeEncodingSpectrum
from helper.prot import create_hdr
from gre_refscan_B0 import gre_refscan_B0

from scipy.io import savemat, loadmat
from dipy.core.geometry import vec2vec_rotmat
from scipy.integrate import cumulative_trapezoid

#%% Parameters 
"""
PyPulseq units (SI): 
time:       [s] (not [ms] as in documentation)
spatial:    [m]
gradients:  [Hz/m] (gamma*T/m)
grad area:  [1/m]
flip angle: [rad]

SLR pulses should always be used for the diffusion sequence!!!
Some units get converted below, others have to stay in non-SI units as spiral calculation needs different units.
Use custom version of Pypulseq (important for delays): https://github.com/mavel101/pypulseq (branch dev_mv)
Change to Pulseq 1.4 (PyPulseq is still on 1.3.1 though) (no sign change for trapezoidal gradients on x-axis)

"""
# General
B0              = 6.983       # field strength [T]
scanner         = 'connectom' # scanner for acoustic resonance check ('connectom', 'skyra')
grads_off       = False       # turn off gradients in reference scan (just for simulation of ECC)
seq_name        = 'seqname'   # sequence/protocol filename

# Sequence - Contrast and Geometry
fov             = 210          # field of view [mm]
TR              = 156.0        # repetition time [ms]
TE              = 119.2          # echo time [ms]
res             = 1.5          # in plane resolution [mm]
slice_res       = 1.5          # slice thickness [mm]
dist_fac        = 0            # distance factor for slices [%]
slices          = 70           # number of slices
averages        = 1            # number of averages
inner_avg       = True         # do averages in inner loop
repetitions     = 1            # number of repetitions

refscan         = 2            # 0: no refscan, 1: normal refscan, 2: B0 mapping refscan
res_refscan     = 3            # resolution of refscan, if B0 mapping is performed, 2mm is typically sufficient
bw_refscan      = 1000         # Bandwidth of the reference scan [Hz]
flip_refscan    = 25           # reference scan flip angle
half_refscan    = False        # collect only half number of slices in refscan (with doubled slice thickness) - for 1mm datasets
separate_tr     = False        # separate TRs for echoes of B0 mapping refscans
n_TEref         = 3            # number of echoes in the refscan
prepscans       = 5            # number of preparation/dummy scans
noisescans      = 16           # number of noise scans

# ADC
os_factor       = 2            # oversampling factor (automatic 2x os from Siemens is not applied)
max_adc         = 32768        # maximum number of samples per ADC (originally 8192 was max, but VE12U seems to accept higher values)

# RF
flip_angle      = 90           # flip angle of excitation pulse [°]
slr_rf          = True         # True: use Sigpys SLR pulse design, False: use sinc pulses
rf_dur          = 4            # RF duration [ms]
tbp_exc         = 6            # time bandwidth product excitation pulse
exc_fac         = 1            # factor for slice thickness of excitation pulse (=1 in Siemens diffusion seq)
rf_refoc_dur    = 8            # refocusing pulse duration [ms]
tbp_refoc       = 3            # time bandwidth product refocusing pulse
refoc_fac       = 1            # factor for slice thickness of refocusing pulse (=1 in Siemens diffusion seq)
rf_spoiling     = False        # RF spoiling
sms             = True         # multiband imaging?
sms_factor      = 2            # multiband factor
sms_type        = 2            # SMS type - 0: only SMS RF, 1: blipped spiral, 2: wave-like/sinus blips in z
kz_steps        = 1            # number of steps in slice direction blipped spiral (should be <sms_factor)
wave_periods    = 7            # only SMS type 2: number of wave periods

fatsat          = True         # Fat saturation pulse
fatsat_tbp      = 2.1          # tbp of fatsat pulse [ms] (BW is fixed at 1000Hz atm)
fatsat_fa       = 70           # flip angle of the fat saturation pulse [°]
extra_fatsat    = True         # Only diffusion: Increase length of refocusing pulse to improve fatsat (see Ivanov, MRM, 2010)
grad_reversal   = False        # gradient reversal for extra fat suppression

# Gradients
max_slew        = 182          # maximum slewrate [T/m/s] (system limit)
spiral_slew     = 145          # maximum slew rate of spiral gradients
max_grad        = 55           # maximum gradient amplitude [mT/m] (system limit)
max_grad_sp     = 42           # maximum gradient amplitude of spiral gradients

Nintl           = 3            # spiral interleaves
redfac          = 3            # reduction/acceleration factor
spiraltype      = 1            # 1: Spiral Out, 4: ROI
spiral_os       = 1            # variable density spiral oversampling in center
trans_beg       = 0.25         # variable density transition beginning (between 0 and 1)
trans_end       = 0.28         # transition end (>trans_beg & between 0 and 1)

skope           = True         # add trigger for skope measurement
measure_delay   = False        # if False start skope measurement directly before spirals, if True start at the beginning of echo time delay
sync_scans      = 10           # number of Skope sync_scans   

# Diffusion
diff_slewrate   = 190          # diffusion gradient slewrate [T/m/s]
diff_maxgrad    = 70           # diffusion gradient max grad strength [mT/m]
diffSeq         = 1            # diffusion sequence; 0: qMas with tuned LTE (PTE,STE,LTE), 1: gDOR - with 180° pulse and separate LTE and separate PTE (PTE,LTE)
qti_protocol    = "Q2"      # "Q2", "Q3"; None: the default protocol will be used for the corresponding NOW option
if diffSeq == 1 or qti_protocol == "Q2":
    # Q2 protocol (Morez et al., Hum. Brain Mapp., 2023)
    b_val           = [[0,1400],[0,100,800,1400]]             # b-values [s/mm^2] (b=0 has to be included) [PTE, LTE]
    directions      = [[9,36],[22,6,30,48]]                   # number of acquisitions for each b-value [PTE, LTE] short protocol: [[5,18],[11,3,15,24]]
    bDelta          = [-0.5,1]                                # shape of diffusion tensor [PTE, LTE]
elif qti_protocol == "Q3":
    # Q3 protocol (Morez et al., Hum. Brain Mapp., 2023)
    b_val           = [[0,100,800],[0,1400],[0,100,800,1400]] # b-values [s/mm^2] (b=0 has to be included) [PTE, STE, LTE]
    directions      = [[4,7,9],[8,30],[20,9,50,15]]           # number of acquisitions for each b-value [PTE, STE, LTE]
    bDelta          = [-0.5,0,1]                              # shape of diffusion tensor [PTE, STE, LTE]
    rotate_STE      = True                                    # STE eigenvectors should be rotated as well
btens_slice             = 1     # include slice selection gradient and crusher in the b-tensor calculation
vol_TR                  = None  # volume TR [s], if None take minimum volume TR
plotSpectrum            = False # plot encoding spectrum of diffusion train
testCrushingMoments     = False # test if crusher moments are above 2 k-spaces for each diffusion volume

#%% Limits, checks and preparations

# Set System limits
rf_dead_time = 100e-6 # lead time before rf can be applied
rf_ringdown_time = 30e-6 # coil hold time (20e-6) + frequency reset time (10e-6)
system = Opts(max_grad=max_grad, grad_unit='mT/m', max_slew=max_slew, slew_unit='T/m/s', 
                rf_dead_time=rf_dead_time, rf_ringdown_time=rf_ringdown_time, grad_raster_time=ph.dt_grad, rf_raster_time=ph.dt_rf)

# convert parameters to Pulseq units
TR          *= 1e-3 # [s]
TE          *= 1e-3 # [s]
rf_dur      *= 1e-3 # [s]
rf_refoc_dur*= 1e-3 # [s]
slice_res   *= 1e-3 # [m]
res_refscan *= 1e-3 # [m]

# calculate effective interleaves
intl_eff = int(Nintl/redfac)

# set spoiler area
spoiler_area = 2/min(slice_res,1e-3*res) # 2 k-spaces
amp_spoil, ftop_spoil, ramp_spoil = ph.trap_from_area(spoiler_area, system, slewrate=min(90,max_slew), max_grad=min(40e-3, 1e-3*max_grad)) # reduce slew rate to avoid stimulation

# averaging
if inner_avg:
    avgs_in = averages
    avgs_out = 1
else:
    avgs_in = 1
    avgs_out = averages

# do some checks
if sms:
    slices_eff = int(slices/sms_factor)
    slice_sep = slices/sms_factor*slice_res*(1+dist_fac*1e-2) # distance between multiband slices [m]
    if slices/sms_factor%1 != 0:
        raise ValueError('Number of slices is not multiple of sms factor')
    if slices/sms_factor%2 == 0:
        raise ValueError('Slices/sms_factor (= number of stacks) must be an odd number') # ref: Barth (08/2015)
    if sms_factor > 2:
        mb_phs = 'quad_mod' # saves SAR and peak amp
    else:
        mb_phs = 'None'
else:
    slices_eff = slices
    sms_factor = 1
    sms_type = 0

if (half_refscan) and slices%2 != 0:
    raise ValueError('half_refscan only possible for even slice number.')

if skope:
    skope_delay = 200e-6 # delay/gradient free interval after Skope trigger
else:
    skope_delay = 0
    sync_scans = 0

if Nintl/redfac%1 != 0:
    raise ValueError('Number of interleaves is not multiple of reduction factor')

if int(fov/res+0.5) % 2:
    raise ValueError(f'Matrix size {int(fov/res+0.5)} (FOV/resolution) is not even.') 

if spiraltype!=1 and spiraltype!=4:
    ValueError('Right now only spiraltype 1 (spiral out) and 4 (ROI) possible.')

if redfac > 1 and refscan==0:
    print("WARNING: Cartesian reference scan is not activated.")

if diffSeq == 1 or qti_protocol=="Q2":
    if len(bDelta)!=2:
        raise ValueError('PTE and LTE have to be measured.')
else:
    if len(bDelta)!=3:
        raise ValueError('PTE, STE and LTE have to be measured.')

if len(b_val) != len(directions):
    raise ValueError('Select same PTE, STE and LTE option for b-value and direction lists.')

for k, b in enumerate(b_val):
    if len(b) != len(directions[k]):
        raise ValueError('b-value and direction lists have to be the same length.')
    if not b:
        raise ValueError('Select at least one b-value.')
    if 0 not in b:
        print("WARNING: No b=0 axquisition selected.")

if rf_dur==rf_refoc_dur and tbp_exc==tbp_refoc:
    raise ValueError('Do not choose same duration and TBP for excitation and refocusing pulse. Crashes the sequence due to a Pulseq or Pypulseq bug.')

if extra_fatsat:
    if B0 < 6:
        extra_fatsat = False
        print("WARNING: Extra fatsat is not compatible at high fields since refusing time will be too long. Use grad_reversal option instead.")
        grad_reversal = True

#%% RF Pulse and slab/slice selection gradient

# make rf pulse and calculate duration of excitation and rewinding
if slr_rf:
    sigpy_cfg = SigpyPulseOpts(pulse_type='slr', ptype='st')
    if flip_angle == 90:
        sigpy_cfg.ptype = 'ex'
        sigpy_cfg.cancel_alpha_phs = True
    rf, gz, gz_rew, rf_del = sigpy_n_seq(flip_angle=flip_angle*np.pi/180, system=system, duration=rf_dur, slice_thickness=slice_res*exc_fac,
                        time_bw_product=tbp_exc, pulse_cfg=sigpy_cfg, use='excitation', return_gz=True, return_delay = True, disp=False)
else:
    rf, gz, gz_rew, rf_del = make_sinc_pulse(flip_angle=flip_angle*np.pi/180, system=system, duration=rf_dur, slice_thickness=slice_res*exc_fac,
                            apodization=0.5, time_bw_product=tbp_exc, use='excitation', return_gz=True, return_delay = True)

if sms:
    band_sep  = slice_sep/slice_res/exc_fac*tbp_exc # normalized distance between slices
    rf.signal = rfsig.multiband.mb_rf(rf.signal, n_bands=sms_factor, band_sep=band_sep, phs_0_pt=mb_phs)

if grad_reversal:
    gz_rew.amplitude *= -1
    gz.amplitude *= -1

# timing
exc_to_rew = calc_duration(rf, gz, rf_del) - rf.delay - ph.round_up_to_raster(rf_dur/2, decimals=5) # time from middle of rf pulse to rewinder
rew_dur = calc_duration(gz_rew)

# refocusing pulse - increase slice thickness slightly for better refocusing
slc_diff = 0
if slr_rf:
    sigpy_cfg_ref = SigpyPulseOpts(pulse_type='slr', ptype='se')
    while slc_diff < 1e3*slice_res*refoc_fac:
        rf_refoc, gz_refoc, _ = sigpy_n_seq(flip_angle=2*flip_angle*np.pi/180, system=system, duration=rf_refoc_dur, slice_thickness=slice_res*refoc_fac,
                            time_bw_product=tbp_refoc, pulse_cfg=sigpy_cfg_ref, use='refocusing', return_gz=True, disp=False)
        if extra_fatsat:
            slc_diff = abs(3.35*B0*(1/(1e3*gz.amplitude/system.gamma) - 1/(1e3*gz_refoc.amplitude/system.gamma)))
            rf_refoc_dur += 1e-3
        else:
            break
else:
    while slc_diff < 1e3*slice_res*refoc_fac:
        rf_refoc, gz_refoc, _  = make_sinc_pulse(flip_angle=2*flip_angle*np.pi/180, system=system, duration=rf_refoc_dur, slice_thickness=slice_res*refoc_fac,
                                apodization=0.5, time_bw_product=tbp_refoc, use='refocusing', return_gz=True)
        if extra_fatsat:
            slc_diff = abs(3.35*B0*(1/(1e3*gz.amplitude/system.gamma) - 1/(1e3*gz_refoc.amplitude/system.gamma)))
            rf_refoc_dur += 1e-3
        else:
            break
gz_refoc_area = gz_refoc.area
if extra_fatsat:
    rf_refoc_dur -= 1e-3
    print(f"New refocusing pulse duration: {1e3*rf_refoc_dur:.2f}ms")

if sms:
    band_sep_refoc  = slice_sep/slice_res/refoc_fac*tbp_refoc
    rf_refoc.signal = rfsig.multiband.mb_rf(rf_refoc.signal, n_bands=sms_factor, band_sep=band_sep_refoc, phs_0_pt=mb_phs)

# define crusher - only for b=0
#  rf pulse                  #########
#  slice gradient          #############
#  crusher gradients   ######         ######

crusher_z1 = make_trapezoid(channel='z',system=system, amplitude=amp_spoil, flat_time=ftop_spoil, rise_time=ramp_spoil)
crusher_z2 = copy(crusher_z1)
crusher_dur = calc_duration(crusher_z1)

# merge crushers with refocusing gradient
rf_refoc.delay = crusher_dur
gz_refoc = make_trapezoid(channel='z', system=system, amplitude=gz_refoc.amplitude, flat_time=gz_refoc.flat_time, rise_time=ramp_spoil)
grad_refoc1 = ph.merge_ramps([crusher_z1, gz_refoc, crusher_z2], system=system) # cumulative area is preserved
grad_refoc2 = gz_refoc
grad_refoc2.delay = crusher_dur-ramp_spoil
refoc_dur  = calc_duration(grad_refoc1)

# define crushers also on x- and y channels
crusher_xy1 = copy(crusher_z1)
crusher_xy2 = copy(crusher_z1)
crusher_xy2.delay = round(calc_duration(grad_refoc1) - crusher_dur, ndigits=5)
crusher_xy1.channel = 'x'
crusher_xy2.channel = 'x'
crusher_x = ph.add_gradients([crusher_xy1,crusher_xy2], system=system)
crusher_xy1.channel = 'y'
crusher_xy2.channel = 'y'
crusher_y = ph.add_gradients([crusher_xy1,crusher_xy2], system=system)
crusher_wf = crusher_x.waveform

# check, if amplitudes of slice gradients differ at least by 20%
if not grad_reversal and not extra_fatsat:
    gz_exc_wf = ph.waveform_from_seqblock(gz)
    gz_refoc_wf = ph.waveform_from_seqblock(gz_refoc)
    gz_exc_amp = gz_exc_wf[len(gz_exc_wf)//2]
    gz_refoc_amp = gz_refoc_wf[len(gz_refoc_wf)//2]
    gz_ratio = gz_exc_amp/gz_refoc_amp
    if gz_ratio > 0.8 and gz_ratio < 1.25:
        raise ValueError(f'Ratio of slice gradient amplitudes of excitation and refocusing pulse is {gz_ratio:.2f}. Should differ by minimum 25%.')
    slc_diff = abs(1e6*3.35e-6*B0*(1/(1e3*gz_exc_amp/system.gamma) - 1/(1e3*gz_refoc_amp/system.gamma)))
    if slc_diff < 1e3*(slice_res*(refoc_fac+exc_fac)/2):
        print(f"WARNING: Fat slices of excitation and refocusing pulse only differ by {slc_diff:.2f} mm. Fat artifacts might occur.")

# RF spoiling parameters
rf_spoiling_inc = 50 # increment of RF spoiling [°]
rf_phase        = 0 
rf_inc          = 0

# Fat saturation
fw_shift = 3.35e-6 # unsigned fat water shift [ppm]
if fatsat:
    offset = -1 * int(B0*system.gamma*fw_shift)
    fatsat_bw = abs(offset) # bandwidth [Hz]
    fatsat_dur = ph.round_up_to_raster(fatsat_tbp/fatsat_bw, decimals=5)
    rf_fatsat, fatsat_del = make_gauss_pulse(flip_angle=fatsat_fa*np.pi/180, duration=fatsat_dur, bandwidth=fatsat_bw, freq_offset=offset, system=system, return_delay = True)

#%% Diffusion gradients and delay calculation 

# Delays in diffusion sequence

#  RF  rew refoc_delay  diffgrad1   RF_refoc   diffgrad2   te_delay   readout
# #### ### ###########  #########   ########   #########   ########   #######
#   -------------- TE/2 ----------------
#   -------------------------------------------- TE ------------------

# make diffusion volume list
diff_list = []

for enc, b in enumerate(b_val):

    # sort b-values and directions
    ix = np.argsort(b)
    b = (np.array(b)[ix]).tolist()
    directions[enc] = (np.array(directions[enc])[ix]).tolist()

    # calculate directions from discoball scheme
    bval_list_wo0 = []
    dir_list_wo0 = []
    for k,dirs in enumerate(directions[enc]):
        if b[k] != 0:
            disco_dirs = DISCSo(dirs, bNisNtheta=False)
            bval_list_wo0.extend(dirs*[b[k]])
            if (qti_protocol=="Q3") and (not rotate_STE) and (enc == 1):
                dir_list_wo0.append(np.asarray(dirs*[[1,0,0]]))
            else:
                dir_list_wo0.append(np.round(disco_dirs, decimals=5))
    dir_list_wo0 = np.concatenate(dir_list_wo0,axis=0)
    bval_list_wo0 = np.asarray(bval_list_wo0)

    # create random distribution of b-volumes reducing drift and thermal load
    diff_list_w0 = [{"bval": bval_list_wo0[k], "dir": dir_list_wo0[k], "bDelta":bDelta[enc]} for k in range(len(bval_list_wo0))]
    random.seed(enc)
    diff_list.append(random.sample(diff_list_w0,len(diff_list_w0)))

    # distribute lowest b-value across volume acquisitions
    b0_dir = directions[enc][0]-2 # ensure first and the last diffusion volume as b0 image
    n_skip = len(diff_list[enc]) // (b0_dir + 1)
    for i in range(b0_dir):
        index = (i + 1) * n_skip + i
        diff_list[enc].insert(index, {"bval": 0, "dir": np.zeros([3]), "bDelta": bDelta[enc]})
    diff_list[enc].insert(0, {"bval": 0, "dir": np.zeros([3]), "bDelta": bDelta[enc]})
    diff_list[enc].append({"bval": 0, "dir": np.zeros([3]), "bDelta": bDelta[enc]})

diff_list = [volume for encoding in diff_list for volume in encoding]

# save lists as arrays
bval_list = []
dir_list = []
bDelta_list = []
for elem in diff_list:
    bval_list.append(elem["bval"])
    dir_list.append(elem["dir"])
    bDelta_list.append(elem["bDelta"])
bval_list = np.asarray(bval_list)
dir_list = np.asarray(dir_list)
bDelta_list = np.asarray(bDelta_list)

# set diffusion max. grad and max. slewrate
diff_maxgrad_Hz = 1e-3 * diff_maxgrad * system.gamma
diff_slewrate_Hz = diff_slewrate * system.gamma
system.max_slew = diff_slewrate_Hz
system.max_grad = diff_maxgrad_Hz

# calculate diffusion parameters
b_val_max = max(max(b_val))


# make diffusion TVE gradients

if diffSeq == 0: # qMas with tuned LTE (PTE,STE,LTE)

    # calculate diffusion parameters from Matlab-optimization
    if qti_protocol=="Q2":
        includePTE = 1
    else:
        includePTE = 0
    matlab_executable = 'matlab'
    matlab_script_name = 'qMasOptimization'
    maxgrad = 1e-3 * diff_maxgrad
    maxslew = diff_slewrate
    arguments = f"{b_val_max}, {refoc_dur}, {includePTE}, {maxgrad}, {maxslew}"
    command = f"{matlab_executable} -nodisplay -r \"{matlab_script_name}({arguments}); exit\""
    completed_process = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    count = completed_process.stdout.decode('utf-8').find('min_spacing')
    input_string = completed_process.stdout.decode('utf-8')[count:-1]
    numbers = re.findall(r'\d+\.\d+|\d+', input_string)
    numbers = [float(num) for num in numbers]

    # set diffusion parameters
    system_max_grad = numbers[1]
    system_max_slew = numbers[2]
    factor_delta = 10 ** 5
    numbers[3] = math.floor(numbers[3] * factor_delta) / factor_delta # to eliminate rounding errors
    params_diff = diff_params(b_val=b_val_max, delta=numbers[3], spacing=numbers[0])

    # make spherical gradients
    diffgrad_ramp = ph.round_up_to_raster(system_max_grad/system_max_slew, 5)
    diffgrad_flat = params_diff.delta - diffgrad_ramp
    diffgrad_dur = diffgrad_flat + 2*diffgrad_ramp
    trap_diff_z = make_trapezoid(channel='z', system=system, flat_time=diffgrad_flat, rise_time=diffgrad_ramp, amplitude=system_max_grad, max_slew=system_max_slew)

    x = ph.round_up_to_raster((maxgrad * system.gamma)/((maxslew*0.94) * system.gamma), 5)
    tp = params_diff.spacing - diffgrad_dur - refoc_dur - 4*x
    tv = (params_diff.spacing+diffgrad_dur-refoc_dur)/2
    tn = (params_diff.spacing+diffgrad_dur+refoc_dur)/2
    counter = params_diff.delta**2*(params_diff.spacing-params_diff.delta/3)+diffgrad_ramp**3/30-params_diff.delta*diffgrad_ramp**2/6
    alpha_x = np.sqrt(counter/(params_diff.delta**2*(params_diff.spacing-diffgrad_dur+5/3*refoc_dur-4/3*x)))
    alpha_y = np.sqrt(counter/(params_diff.delta**2*(tp+8*x+2*np.pi**2*x**2/tp+8*np.pi**2*x**3/(5*tp**2))))
    C = (45*params_diff.spacing*diffgrad_dur**4 - 180*params_diff.spacing*diffgrad_dur**3*params_diff.delta - 60*params_diff.spacing*diffgrad_dur**3*diffgrad_ramp + 270*params_diff.spacing*diffgrad_dur**2*params_diff.delta**2 + 120*params_diff.spacing*diffgrad_dur**2*params_diff.delta*diffgrad_ramp + 150*params_diff.spacing*diffgrad_dur**2*diffgrad_ramp**2 - 180*params_diff.spacing*diffgrad_dur*params_diff.delta**3 - 60*params_diff.spacing*diffgrad_dur*params_diff.delta**2*diffgrad_ramp - 180*params_diff.spacing*diffgrad_dur*params_diff.delta*diffgrad_ramp**2 - 60*params_diff.spacing*diffgrad_dur*diffgrad_ramp**3 + 45*params_diff.spacing*params_diff.delta**4 + 90*params_diff.spacing*params_diff.delta**2*diffgrad_ramp**2 + 45*params_diff.spacing*diffgrad_ramp**4 - 12*diffgrad_dur**5 + 60*diffgrad_dur**4*params_diff.delta - 60*diffgrad_dur**4*diffgrad_ramp - 120*diffgrad_dur**3*params_diff.delta**2 + 120*diffgrad_dur**3*params_diff.delta*diffgrad_ramp + 120*diffgrad_dur**3*diffgrad_ramp*tn - 120*diffgrad_dur**3*diffgrad_ramp*tv + 120*diffgrad_dur**2*params_diff.delta**3 - 30*diffgrad_dur**2*params_diff.delta**2*diffgrad_ramp + 90*diffgrad_dur**2*params_diff.delta*diffgrad_ramp**2 - 240*diffgrad_dur**2*params_diff.delta*diffgrad_ramp*tn + 240*diffgrad_dur**2*params_diff.delta*diffgrad_ramp*tv - 40*diffgrad_dur**2*diffgrad_ramp**2*tn + 40*diffgrad_dur**2*diffgrad_ramp**2*tv + 160*diffgrad_dur**2*diffgrad_ramp**2*x - 60*diffgrad_dur*params_diff.delta**4 - 60*diffgrad_dur*params_diff.delta**3*diffgrad_ramp + 120*diffgrad_dur*params_diff.delta**2*diffgrad_ramp*tn - 120*diffgrad_dur*params_diff.delta**2*diffgrad_ramp*tv - 60*diffgrad_dur*params_diff.delta*diffgrad_ramp**3 - 60*diffgrad_dur*diffgrad_ramp**4 + 120*diffgrad_dur*diffgrad_ramp**3*tn - 120*diffgrad_dur*diffgrad_ramp**3*tv + 12*params_diff.delta**5 + 30*params_diff.delta**4*diffgrad_ramp - 30*params_diff.delta**3*diffgrad_ramp**2 + 30*params_diff.delta**2*diffgrad_ramp**3 + 18*diffgrad_ramp**5)/(20*diffgrad_dur**2*diffgrad_ramp**2)
    alpha_z = np.sqrt(counter/(params_diff.delta**2*(C/3)))
    Gx = np.sqrt(2)/3 * system_max_grad * params_diff.delta * 2*np.pi/tp * alpha_x
    Gy = np.sqrt(2/3) * system_max_grad * params_diff.delta * 2*np.pi/tp * alpha_y
    Gz = 4/3 * system_max_grad * params_diff.delta * np.pi/tp * alpha_z

    trap_diff_z_ampl = trap_diff_z.amplitude *  alpha_z
    if trap_diff_z_ampl/(maxgrad * system.gamma) > 1:
        raise ValueError('trap_diff_z_ampl is too high')
    if (trap_diff_z_ampl/trap_diff_z.rise_time)/(maxslew * system.gamma) > 1:
        raise ValueError('trap_diff_z_slew is too high')
    
    time = np.arange(0,tp/2+system.grad_raster_time,system.grad_raster_time)
    
    time_ramp = np.arange(0,x+system.grad_raster_time,system.grad_raster_time)
    Gy_ramp_wf = Gy/x * time_ramp
    Gy_ramp_wf = points_to_waveform(amplitudes=Gy_ramp_wf, grad_raster_time=system.grad_raster_time, times=time_ramp)
    Gy_sinus_wf = Gy*np.cos(2*np.pi/tp*time)
    Gy_sinus_wf = points_to_waveform(amplitudes=Gy_sinus_wf, grad_raster_time=system.grad_raster_time, times=time)
    GySinus_wf_first = np.concatenate((Gy_ramp_wf,Gy_sinus_wf,(-1)*Gy_ramp_wf[::-1]))
    GySinus_wf_second = np.concatenate((Gy_ramp_wf,Gy_sinus_wf,(-1)*Gy_ramp_wf[::-1]))
    GySinus_first = make_arbitrary_grad(channel='y', waveform=GySinus_wf_first, system=system)
    GySinus_second = make_arbitrary_grad(channel='y', waveform=GySinus_wf_first, system=system)

    GxSinus_wf = -Gx*np.sin(2*np.pi/tp*time)
    GxSinus_wf_first = points_to_waveform(amplitudes=GxSinus_wf, grad_raster_time=system.grad_raster_time, times=time)
    GxSinus_wf_second = points_to_waveform(amplitudes=GxSinus_wf, grad_raster_time=system.grad_raster_time, times=time)
    GxSinus_first = make_arbitrary_grad(channel='x', waveform=GxSinus_wf_first, delay = x, system=system)
    GxSinus_second = make_arbitrary_grad(channel='x', waveform=GxSinus_wf_second, delay = x, system=system)
    
    GzSinus_wf = -Gz*np.sin(2*np.pi/tp*time)
    GzSinus_wf_first = points_to_waveform(amplitudes=GzSinus_wf, grad_raster_time=system.grad_raster_time, times=time)
    GzSinus_wf_second = points_to_waveform(amplitudes=GzSinus_wf, grad_raster_time=system.grad_raster_time, times=time)
    GzSinus_first = make_arbitrary_grad(channel='z', waveform=GzSinus_wf_first, delay = x, system=system)
    GzSinus_second = make_arbitrary_grad(channel='z', waveform=GzSinus_wf_first, delay = x, system=system)

    rf_longer = 1.01e-3 # [s] insert pause due to SLL-limit near crusher around 180° pulse
    sinusTime_first = make_delay(d=ph.round_up_to_raster(calc_duration(GxSinus_first,GySinus_first,GzSinus_first)+rf_longer,decimals=5))
    GxSinus_second.delay += ph.round_up_to_raster(rf_longer,decimals=5)
    GySinus_second.delay += ph.round_up_to_raster(rf_longer,decimals=5)
    GzSinus_second.delay += ph.round_up_to_raster(rf_longer,decimals=5)
    sinusTime_second = make_delay(d=ph.round_up_to_raster(calc_duration(GxSinus_second,GySinus_second,GzSinus_second),decimals=5))

    # make linear gradients out of spherical gradients
    GxSinus_lin_x_first = make_arbitrary_grad(channel='x', waveform=GxSinus_wf_first*np.sqrt(3), delay = GxSinus_first.delay, system=system)
    GxSinus_lin_y_first = make_arbitrary_grad(channel='y', waveform=GxSinus_wf_first*np.sqrt(3), delay = GxSinus_first.delay, system=system)
    GxSinus_lin_z_first = make_arbitrary_grad(channel='z', waveform=GxSinus_wf_first*np.sqrt(3), delay = GxSinus_first.delay, system=system)
    GxSinus_lin_first = [GxSinus_lin_x_first,GxSinus_lin_y_first,GxSinus_lin_z_first]
    GxSinus_lin_x_second = make_arbitrary_grad(channel='x', waveform=GxSinus_wf_first*np.sqrt(3), delay = GxSinus_second.delay, system=system)
    GxSinus_lin_y_second = make_arbitrary_grad(channel='y', waveform=GxSinus_wf_first*np.sqrt(3), delay = GxSinus_second.delay, system=system)
    GxSinus_lin_z_second = make_arbitrary_grad(channel='z', waveform=GxSinus_wf_first*np.sqrt(3), delay = GxSinus_second.delay, system=system)
    GxSinus_lin_second = [GxSinus_lin_x_second,GxSinus_lin_y_second,GxSinus_lin_z_second]

    # make planar gradients out of spherical gradients
    wf_GxSinus_sph_first = wf_GxSinus_sph_second = np.concatenate((np.zeros(int((x)/system.grad_raster_time)),GxSinus_wf_first,np.zeros(int((x)/system.grad_raster_time))))
    wf_GySinus_sph_first = wf_GySinus_sph_second = GySinus_wf_first.copy()

    # rotate planar gradients to test system gradients limits; all gradients will be created below again
    for vol_ix, diff in enumerate(diff_list):
        if diff['bDelta'] == -0.5:
            M = vec2vec_rotmat(np.array([1,0,0]), diff['dir'])
            
            # ensure crushing moment on slice axis
            g_slice_spoil = np.sqrt(diff['bval']/b_val_max)*np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_first*np.sqrt(3/2)*M[2,1] + wf_GySinus_sph_first*np.sqrt(3/2)*M[2,2])
            t_grad = (np.arange(g_slice_spoil.shape[0]) + 0.5) * system.grad_raster_time
            q_slice_spoil = system.gamma * cumulative_trapezoid(g_slice_spoil, t_grad,initial=0)
            if q_slice_spoil[-1]<0:
                M[:,1]*=-1

            GxSinus_plan_x_first = make_arbitrary_grad(channel='x', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_first*np.sqrt(3/2)*M[0,1] + wf_GySinus_sph_first*np.sqrt(3/2)*M[0,2]), delay=GySinus_first.delay, system=system) # consider sign change from Pulseq rotation matrix
            GxSinus_plan_y_first = make_arbitrary_grad(channel='y', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_first*np.sqrt(3/2)*M[1,1] + wf_GySinus_sph_first*np.sqrt(3/2)*M[1,2]), delay=GySinus_first.delay, system=system)
            GxSinus_plan_z_first = make_arbitrary_grad(channel='z', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_first*np.sqrt(3/2)*M[2,1] + wf_GySinus_sph_first*np.sqrt(3/2)*M[2,2]), delay=GySinus_first.delay, system=system)

            GxSinus_plan_x_second = make_arbitrary_grad(channel='x', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_second*np.sqrt(3/2)*M[0,1] + wf_GySinus_sph_second*np.sqrt(3/2)*M[0,2]), delay=GySinus_second.delay, system=system) # consider sign change from Pulseq rotation matrix
            GxSinus_plan_y_second = make_arbitrary_grad(channel='y', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_second*np.sqrt(3/2)*M[1,1] + wf_GySinus_sph_second*np.sqrt(3/2)*M[1,2]), delay=GySinus_second.delay, system=system)
            GxSinus_plan_z_second = make_arbitrary_grad(channel='z', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_second*np.sqrt(3/2)*M[2,1] + wf_GySinus_sph_second*np.sqrt(3/2)*M[2,2]), delay=GySinus_second.delay, system=system)
                    
    GxSinus_plan_first = [GxSinus_plan_x_first,GxSinus_plan_y_first,GxSinus_plan_z_first]
    GxSinus_plan_second = [GxSinus_plan_x_second,GxSinus_plan_y_second,GxSinus_plan_z_second]


elif diffSeq == 1: # gDOR - with 180° and separate LTE and PTE (PTE,LTE)

    # load optimized waveforms (effective waveform)
    gDORResultsLTE = loadmat('qti_waveforms/from_optimization_to_seq/gDOR_LTE_18Hz_tuned.mat')
    time_first_LTE = gDORResultsLTE['timeFirst'][0,:]  # [s]
    time_second_LTE = gDORResultsLTE['timeSecond'][0,:] # [s]
    time_third_LTE = gDORResultsLTE['timeZero'][0,:] # [s]

    gDORResultsPTE = loadmat('qti_waveforms/from_optimization_to_seq/gDOR_PTE_18Hz_isotropic.mat')
    time_first_PTE = gDORResultsPTE['timeFirst'][0,:]  # [s]
    time_second_PTE = gDORResultsPTE['timeSecond'][0,:] # [s]
    time_third_PTE = gDORResultsPTE['timeZero'][0,:] # [s]

    # make linear gradients
    wf_Glin_z_first = gDORResultsLTE['gwfFirst'][:,2] * system.gamma # [Hz/m]
    wf_Glin_z_first = points_to_waveform(amplitudes=wf_Glin_z_first, grad_raster_time=system.grad_raster_time, times=time_first_LTE)
    wf_Glin_z_second = -1*gDORResultsLTE['gwfSecond'][:,2] * system.gamma # [Hz/m]
    wf_Glin_z_second = points_to_waveform(amplitudes=wf_Glin_z_second, grad_raster_time=system.grad_raster_time, times=time_second_LTE)

    # ensure crushing moment
    t_grad = (np.arange(wf_Glin_z_first.shape[0]) + 0.5) * system.grad_raster_time
    q=system.gamma * cumulative_trapezoid(wf_Glin_z_first, t_grad,initial=0)
    if q[-1]<0:
        wf_Glin_z_first *= -1
        wf_Glin_z_second *= -1

    GxSinus_lin_x_first = make_arbitrary_grad(channel='x', waveform=wf_Glin_z_first, system=system)
    GxSinus_lin_y_first = make_arbitrary_grad(channel='y', waveform=wf_Glin_z_first, system=system)
    GxSinus_lin_z_first = make_arbitrary_grad(channel='z', waveform=wf_Glin_z_first, system=system)
    GxSinus_lin_x_second = make_arbitrary_grad(channel='x', waveform=wf_Glin_z_second, system=system)
    GxSinus_lin_y_second = make_arbitrary_grad(channel='y', waveform=wf_Glin_z_second, system=system)
    GxSinus_lin_z_second = make_arbitrary_grad(channel='z', waveform=wf_Glin_z_second, system=system)

    # calculate timing of the diffusion encoding parts
    rf_longer = ph.round_up_to_raster((np.round(time_third_LTE[-1],5) - refoc_dur)/2,decimals=5)
    sinusTime_first = make_delay(d=ph.round_up_to_raster(calc_duration(GxSinus_lin_x_first,GxSinus_lin_y_first,GxSinus_lin_z_first)+rf_longer,decimals=5))
    GxSinus_lin_x_second.delay = GxSinus_lin_y_second.delay = GxSinus_lin_z_second.delay = rf_longer
    sinusTime_second = make_delay(d=ph.round_up_to_raster(calc_duration(GxSinus_lin_x_second,GxSinus_lin_y_second,GxSinus_lin_z_second),decimals=5))
    diffgrad_dur = 0
    GxSinus_lin_first = [GxSinus_lin_x_first,GxSinus_lin_y_first,GxSinus_lin_z_first]
    GxSinus_lin_second = [GxSinus_lin_x_second,GxSinus_lin_y_second,GxSinus_lin_z_second]

    # make planar gradients out of standalone optimization and prolong to reach LTE duration
    wf_Gplan_x_first = gDORResultsPTE['gwfFirst'][:,0] * system.gamma # [Hz/m]
    wf_Gplan_x_first = points_to_waveform(amplitudes=wf_Gplan_x_first, grad_raster_time=system.grad_raster_time, times=time_first_PTE)
    wf_Gplan_x_second = -1*gDORResultsPTE['gwfSecond'][:,0] * system.gamma # [Hz/m]
    wf_Gplan_x_second = points_to_waveform(amplitudes=wf_Gplan_x_second, grad_raster_time=system.grad_raster_time, times=time_second_PTE)
    
    wf_Gplan_y_first = gDORResultsPTE['gwfFirst'][:,1] * system.gamma # [Hz/m]
    wf_Gplan_y_first = points_to_waveform(amplitudes=wf_Gplan_y_first, grad_raster_time=system.grad_raster_time, times=time_first_PTE)
    wf_Gplan_y_second = -1*gDORResultsPTE['gwfSecond'][:,1] * system.gamma # [Hz/m]
    wf_Gplan_y_second = points_to_waveform(amplitudes=wf_Gplan_y_second, grad_raster_time=system.grad_raster_time, times=time_second_PTE)

    # rotate planar gradients to test system gradients limits; all gradients will be created below again
    for vol_ix, diff in enumerate(diff_list):
        if diff['bDelta'] == -0.5:
            M = vec2vec_rotmat(np.array([0,0,1]), diff['dir'])

            # ensure crushing moment on slice axis
            g_slice_spoil = np.sqrt(diff['bval']/b_val_max)*(wf_Gplan_x_first*M[2,0] + wf_Gplan_y_first*M[2,1])
            t_grad = (np.arange(g_slice_spoil.shape[0]) + 0.5) * system.grad_raster_time
            q_slice_spoil = system.gamma * cumulative_trapezoid(g_slice_spoil, t_grad,initial=0)
            if q_slice_spoil[-1]<0:
                M[:,1]*=-1
            
            GxSinus_plan_x_first = make_arbitrary_grad(channel='x', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_Gplan_x_first*M[0,0] + wf_Gplan_y_first*M[0,1]), system=system, delay = GxSinus_lin_x_first.delay)
            GxSinus_plan_y_first = make_arbitrary_grad(channel='y', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_Gplan_x_first*M[1,0] + wf_Gplan_y_first*M[1,1]), system=system, delay = GxSinus_lin_y_first.delay)
            GxSinus_plan_z_first = make_arbitrary_grad(channel='z', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_Gplan_x_first*M[2,0] + wf_Gplan_y_first*M[2,1]), system=system, delay = GxSinus_lin_z_first.delay)
            GxSinus_plan_x_second = make_arbitrary_grad(channel='x', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_Gplan_x_second*M[0,0] + wf_Gplan_y_second*M[0,1]), system=system, delay = GxSinus_lin_x_second.delay)
            GxSinus_plan_y_second = make_arbitrary_grad(channel='y', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_Gplan_x_second*M[1,0] + wf_Gplan_y_second*M[1,1]), system=system, delay = GxSinus_lin_y_second.delay)
            GxSinus_plan_z_second = make_arbitrary_grad(channel='z', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_Gplan_x_second*M[2,0] + wf_Gplan_y_second*M[2,1]), system=system, delay = GxSinus_lin_z_second.delay)
    
    GxSinus_plan_first = [GxSinus_plan_x_first,GxSinus_plan_y_first,GxSinus_plan_z_first]
    GxSinus_plan_second = [GxSinus_plan_x_second,GxSinus_plan_y_second,GxSinus_plan_z_second]   

# reset the maximum slewrate and gradient
system.max_slew = max_slew * system.gamma
system.max_grad = 1e-3 * max_grad * system.gamma


# calculate minimum TE
min_te_first = exc_to_rew + rew_dur + diffgrad_dur + sinusTime_first.delay + refoc_dur/2
min_te_second = sinusTime_second.delay + diffgrad_dur + refoc_dur/2
min_te = 2*max(min_te_first,min_te_second)
if skope:
    min_te += skope_delay
if TE < min_te:
    raise ValueError(f'Minimum TE: {min_te*1e3}')

# make delays
refoc_delay = make_delay(d = ph.round_up_to_raster(TE/2 - exc_to_rew - rew_dur - diffgrad_dur - sinusTime_first.delay - refoc_dur/2, decimals=5))
te_delay = make_delay(d = ph.round_up_to_raster(TE/2 - refoc_dur/2 - sinusTime_second.delay - diffgrad_dur, decimals=5))

# check spirals for acoustic resonances
if B0 > 4:
    resonances = [(500,600), (930, 1280)] # 7T resonances
else:
    if scanner == 'skyra':
        resonances = [(535, 635), (1010,1230)] # 3T Skyra resonances
    elif scanner == 'connectom':
        resonances = [(280,340), (546, 646), (1000,1500)] # 3T Connectom resonances
    else:
        raise ValueError('Unknown scanner name for 3T, select either skyra or connectom.')
if diffSeq == 1:
    freq_max = ph.check_resonances([GxSinus_lin_x_first.waveform,GxSinus_lin_y_first.waveform,GxSinus_lin_z_first.waveform, GxSinus_lin_x_second.waveform,
                                GxSinus_lin_y_second.waveform,GxSinus_lin_z_second.waveform,wf_Gplan_x_first,wf_Gplan_y_first,wf_Gplan_x_second,wf_Gplan_y_second], resonances)
else:
    freq_max = ph.check_resonances([GxSinus_first.waveform,GySinus_first.waveform,GzSinus_first.waveform,
                                GxSinus_second.waveform,GySinus_second.waveform,GzSinus_second.waveform], resonances) 

#%% Spiral Readout Gradients

# Parameters spiral trajectory:

# parameter         description               default value
# ---------        -------------              --------------

# nitlv:      number of spiral interleaves        15
# res:        resolution                          1 mm
# fov:        target field of view                192 mm
# max_amp:    maximum gradient amplitude          42 mT/m
# min_rise:   minimum gradient risetime           5 us/(mT/m)
# spiraltype: 1: spiral out                   
#             2: spiral in                        
#             3: double spiral                    x
#             4: ROI
#             5: RIO
# spiral_os:  spiral oversampling in center       1

# Maximum rotation angle for spirals
if spiraltype==3:
    max_rot     = np.pi
else:
    max_rot     = 2*np.pi  

# read in Spirals [T/m]
min_rise_sp = 1/spiral_slew * 1e3
spiral_calc = spiraltraj.calc_traj(nitlv=Nintl, fov=fov, res=res, spiraltype=spiraltype,
                             min_rise=min_rise_sp, max_amp=max_grad_sp, spiral_os=spiral_os, 
                             vd_transition_begin=trans_beg, vd_transition_end=trans_end)
spiral_calc = np.asarray(spiral_calc)
spiral_x = 1e-3*spiral_calc[:,0]
spiral_y = 1e-3*spiral_calc[:,1]

N_spiral = len(spiral_x)
readout_dur = N_spiral*system.grad_raster_time # readout duration [s]

# write spiral readout blocks to list
spirals = [{'deph': [None, None], 'spiral': [None, None], 'reph': [None, None]} for k in range(Nintl)]
reph_dur = []
save_sp = np.zeros((Nintl, 2, N_spiral)) # save gradients for FIRE reco
rot_angle = np.linspace(0, max_rot, Nintl, endpoint=False)
for k in range(Nintl):
    # rotate spiral gradients for shot selection
    sp_x, sp_y = ph.rot_grad(spiral_x, spiral_y, rot_angle[k])

    save_sp[k,0,:] = sp_x
    save_sp[k,1,:] = sp_y

    # unit to [Hz/m], make spiral gradients
    sp_x *= system.gamma
    sp_y *= system.gamma

    spiral_delay = 20e-6 # delay to avoid ADC artifact (first few points of ADC might be corrupted)
    spirals[k]['spiral'][0] = make_arbitrary_grad(channel='x', waveform=sp_x, delay=spiral_delay, system=system)
    spirals[k]['spiral'][1] = make_arbitrary_grad(channel='y', waveform=sp_y, delay=spiral_delay, system=system)

    # calculate rephaser area
    area_x = sp_x.sum()*system.grad_raster_time
    area_y = sp_y.sum()*system.grad_raster_time

    # calculate rephasers and make gradients - add spoiler area to rephaser
    # reduce slew rate & max_grad to to avoid stimulation
    if fatsat:
        amp_x, ftop_x, ramp_x = ph.trap_from_area(-area_x, system, slewrate=min(100,max_slew), max_grad=min(30e-3,1e-3*max_grad))
        amp_y, ftop_y, ramp_y = ph.trap_from_area(-area_y, system, slewrate=min(100,max_slew), max_grad=min(30e-3,1e-3*max_grad))
    else:
        amp_x, ftop_x, ramp_x = ph.trap_from_area(-area_x+spoiler_area, system, slewrate=min(100,max_slew), max_grad=min(30e-3,1e-3*max_grad))
        amp_y, ftop_y, ramp_y = ph.trap_from_area(-area_y+spoiler_area, system, slewrate=min(100,max_slew), max_grad=min(30e-3,1e-3*max_grad))
    spirals[k]['reph'][0] = make_trapezoid(channel='x', system=system, amplitude=amp_x, flat_time=ftop_x, rise_time=ramp_x)
    spirals[k]['reph'][1] = make_trapezoid(channel='y', system=system, amplitude=amp_y, flat_time=ftop_y, rise_time=ramp_y)
    reph_dur.append(max(ftop_x+2*ramp_x, ftop_y+2*ramp_y))

# spoiler (after fatsat or before next excitation)
spoiler_x  = make_trapezoid(channel='x',system=system, amplitude=amp_spoil, flat_time=ftop_spoil, rise_time=ramp_spoil)
spoiler_y  = make_trapezoid(channel='y',system=system, amplitude=amp_spoil, flat_time=ftop_spoil, rise_time=ramp_spoil)
spoiler_z  = make_trapezoid(channel='z',system=system, amplitude=amp_spoil, flat_time=ftop_spoil, rise_time=ramp_spoil)
spoiler_x_neg  = make_trapezoid(channel='x',system=system, amplitude=-1*amp_spoil, flat_time=ftop_spoil, rise_time=ramp_spoil)
spoiler_y_neg  = make_trapezoid(channel='y',system=system, amplitude=-1*amp_spoil, flat_time=ftop_spoil, rise_time=ramp_spoil)
spoiler_z_neg  = make_trapezoid(channel='z',system=system, amplitude=-1*amp_spoil, flat_time=ftop_spoil, rise_time=ramp_spoil)
spoiler_dur = calc_duration(spoiler_x)

# Blipped (and stacked) spiral as defined in Zahneisen (2014)
blip_dims = 0
if sms and sms_type:
    # calculate delta k in slice direction
    fov_z = sms_factor * slice_sep
    dkz = 1/fov_z

    # blipped spiral
    if sms_type == 1:
        blip = ph.trap_from_area(dkz, system)
        blip = ph.calc_triang_wf(blip[0], blip[1], blip[2])
        blip_big = ph.trap_from_area(kz_steps*dkz, system)
        blip_big = ph.calc_triang_wf(-1*blip_big[0], blip_big[1], blip_big[2])

        # place blips after every 2nd sign change of spiral x-gradient
        sp = save_sp[0,0]
        blips = np.zeros_like(sp)
        signchange = (np.diff(np.sign(sp)) != 0) * 1
        ctr = 0
        step = 0
        end_ix = 0
        for ix,change in enumerate(signchange):
            if change and ix>end_ix:
                ctr += 1
                if ctr%2 == 0 and ctr!=0:
                    if step == kz_steps//2:
                        if (len(blip_big) > len(sp)-ix-1): break
                        blips[ix+1:ix+1+len(blip_big)] = blip_big
                        step -= kz_steps
                        end_ix = ix+len(blip_big)
                    else:
                        if (len(blip) > len(sp)-ix-1): break
                        blips[ix+1:ix+1+len(blip)] = blip
                        step += 1
                        end_ix = ix+len(blip)
        
        sms_blips = make_arbitrary_grad(channel='z', waveform=blips, system=system, delay=spiral_delay)
        blip_dims = 1

    elif sms_type == 2:
        sp_len = len(save_sp[0,0])
        scale = kz_steps * dkz / 2 /  (sp_len*system.grad_raster_time) # sinus has area 2 from 0 to pi
        blips = scale * 2*np.pi*wave_periods * np.sin(np.linspace(0,2*np.pi*wave_periods,sp_len))
        blips_slew = np.diff(blips) / system.grad_raster_time
        if max(abs(blips_slew)) > system.max_slew:
            raise ValueError("Slewrate violation for sinus waves, reduce kz_steps.")

        sms_blips = make_arbitrary_grad(channel='z', waveform=blips, system=system, delay=spiral_delay)
        blip_dims = 1

    # stacked spiral (only GRE sequence, not really tested)
    elif sms_type == 3:
            blip = ph.trap_from_area(dkz*kz_steps//2, system)
            blip_pre = make_trapezoid(channel='z', amplitude=blip[0], flat_time=blip[1], rise_time=blip[2], system=system)
            blip_rew = make_trapezoid(channel='z', amplitude=-1*blip[0], flat_time=blip[1], rise_time=blip[2], system=system)
            if calc_duration(blip_pre) > te_delay.delay:
                raise ValueError(f"TE too short for sms blip. Increase by at least {1e3*(calc_duration(blip)-te_delay.delay)} ms")
else:
    fov_z = slice_res

# check spirals for acoustic resonances
if B0 > 4:
    resonances = [(500,600), (930, 1280)] # 7T resonances
else:
    if scanner == 'skyra':
        resonances = [(535, 635), (1010,1230)] # 3T Skyra resonances
    elif scanner == 'connectom':
        resonances = [(280,340), (546, 646), (1000,1500)] # 3T Connectom resonances
    else:
        raise ValueError('Unknown scanner name for 3T, select either skyra or connectom.')
freq_max = ph.check_resonances([spiral_x,spiral_y], resonances) 

#%% ADC

max_grad_sp_cmb = 1e3*np.max(np.sqrt(abs(spiral_x)**2+abs(spiral_y)**2))
dwelltime = 1/(system.gamma*max_grad_sp_cmb*fov*os_factor)*1e6 # ADC dwelltime [s]
dwelltime = ph.trunc_to_raster(dwelltime, decimals=7) # truncate dwelltime to 100 ns (scanner limit)
min_dwelltime = 1e-6
if dwelltime < min_dwelltime:
    dwelltime = min_dwelltime
print(f"ADC dwelltime: {1e6*dwelltime} us")

num_samples = round((readout_dur+spiral_delay)/dwelltime)
if num_samples%2==1:
    num_samples += 1 # even number of samples

if num_samples <= max_adc:
    num_segments = 1
    print('Number of ADCs: {}.'.format(num_samples))
else:
    # the segment duration has to be on the gradient raster
    # increase number of segments or samples/segments to achieve this
    # number of samples and number of samples per segment should always be an even number
    num_segments = 2
    if (num_samples/num_segments % 2 != 0):
        num_samples += 2
    segm_dur = 1e5 * dwelltime * num_samples/num_segments # segment duration [10us - gradient raster]
    while (not round(segm_dur,ndigits=5).is_integer() or num_samples/num_segments > 8192):
        if num_samples/num_segments > 8192:
            num_segments += 1
            while (num_samples/num_segments % 2 != 0):
                num_samples += 2
        else:
            num_samples += 2*num_segments
        segm_dur = 1e5 * dwelltime * num_samples/num_segments 
    print('ADC has to be segmented!! Number of ADCs: {}. Per segment: {}. Segments: {}.'.format(num_samples,num_samples/num_segments,num_segments))

    # self check
    if (num_samples/num_segments % 2 != 0 or num_samples % 2 != 0 or not round(segm_dur,ndigits=5).is_integer()):
        raise ValueError("Check if number of samples and number of samples per segment are even. Check if segment duration is on gradient raster time.")

if num_samples > 65535: # max of uint16 used by ISMRMRD
    raise ValueError("Too many samples for ISMRMRD format - lower the oversampling factor or take more interleaves")

adc = make_adc(system=system, num_samples=num_samples, dwell=dwelltime)
adc_dur = calc_duration(adc)
adc_delay = ph.round_up_to_raster(adc_dur+200e-6, decimals=5) # add small delay after readout for ADC frequency reset event and to avoid stimulation by rephaser
adc_delay = make_delay(d=adc_delay)
if skope:
    t_skope = (adc_dur+1e-3)*1e3 # add 1 ms to be safe
    if measure_delay:
        t_skope += te_delay.delay * 1e3
    print('Minimum Skope acquisition time: {:.2f} ms'.format(t_skope))

    min_dist = max(160e-3, (4*t_skope+1)*1e-3) # minimum trigger distance due to relaxation is 160ms, distance has to be bigger than 4x acqusition duration
    print('Minimum interleave TR: {:.2f} ms'.format(min_dist*1e3))
    trig_skip = int(np.ceil(min_dist/TR))
    if trig_skip > 1:
        print(f"TR too low to capture all triggers (minimum trigger distance 200ms). Only every {trig_skip}th trigger is captured.")
else:
    trig_skip = 0

#%% Set up protocol for FIRE reco and write header

date = datetime.date.today().strftime('%Y%m%d')
filename = date + '_' + seq_name

# set some parameters for the protocol
t_min = dwelltime/2

ismrmrd_file = f"{filename}.h5"

# set up protocol file and create header
if os.path.exists(ismrmrd_file):
    raise ValueError("Protocol name already exists. Choose different name")
prot = ismrmrd.Dataset(ismrmrd_file)
os.chmod(ismrmrd_file, os.stat(ismrmrd_file).st_mode | stat.S_IWOTH)
hdr = ismrmrd.xsd.ismrmrdHeader()
params_hdr = {"trajtype": "spiral", "fov": fov, "fov_z": fov_z, "res": res, "slices": slices, "slice_res": slice_res*(1+dist_fac*1e-2), 
                "nintl": intl_eff, "avg": averages, "rep": repetitions, "ncontrast": len(diff_list),
                "nsegments": num_segments, "dwelltime": dwelltime, "traj_delay": spiral_delay, "t_min": t_min, 
                "os_region": trans_beg, "os_factor": os_factor, "redfac": redfac, "sms_factor": sms_factor, "half_refscan": half_refscan}
create_hdr(hdr, params_hdr)
up_ho = ismrmrd.xsd.userParameterBase64Type('higher_order',1) # always use PowerGrid higher order recon
hdr.userParameters.userParameterBase64.append(up_ho)

# add pypulseq version
pulseq_version = ismrmrd.xsd.userParameterStringType('pulseq_version', "1.4.0")
hdr.userParameters.userParameterString.append(pulseq_version)

#%% Add sequence blocks to sequence & write acquisitions to protocol

# Set up the sequence
seq = Sequence()
trig_ctr = 0

# Definitions section in seq file
seq.set_definition("Name", filename) # protocol name is saved in Siemens header for FIRE reco
seq.set_definition("FOV", [1e-3*fov, 1e-3*fov, slice_res*(1+dist_fac*1e-2)*(slices-1)+slice_res]) # this sets the volume display in the UI
seq.set_definition("Slice_Thickness", "%f" % slice_res) # this sets the receive gain
if num_segments > 1:
    seq.set_definition("MaxAdcSegmentLength", "%d" % int(num_samples/num_segments+0.5)) # for automatic ADC segment length setting

# TokTokTok
tokx = make_trapezoid(channel='x', amplitude=1e-3*system.gamma, rise_time=1e-3, flat_time=4e-3)
toky = make_trapezoid(channel='y', amplitude=1e-3*system.gamma, rise_time=1e-3, flat_time=4e-3)
tokz = make_trapezoid(channel='z', amplitude=1e-3*system.gamma, rise_time=1e-3, flat_time=4e-3)
seq.add_block(tokx,toky,tokz,make_delay(d=0.5))
seq.add_block(tokx,toky,tokz,make_delay(d=0.5))
seq.add_block(tokx,toky,tokz,make_delay(d=0.5))

# Noise scans
noise_samples = 1024
noise_adc = make_adc(system=system, num_samples=noise_samples, dwell=dwelltime, delay=10e-6) # delay to be safe with pTx system (had crashes due to short NCO gaps)
noise_delay = make_delay(d=ph.round_up_to_raster(calc_duration(noise_adc)+1e-3,decimals=5)) # add some more time to the ADC delay to be safe
for k in range(noisescans):
    seq.add_block(noise_adc, noise_delay)
    acq = ismrmrd.Acquisition()
    acq.setFlag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)
    prot.append_acquisition(acq)

# Perform cartesian reference scan
if refscan:
    params_ref = {}
    d_te = round(1/(B0*system.gamma*ph.fw_shift),5) # fat/water "in-phase"
    if B0 > 4: # 7T
        rf_ref_dur = 2e-3
        tbp_ref = 4
        te_1 = 3*d_te
        te_ref = [round(te_1 + k*d_te, 5) for k in range(n_TEref*2)]
        if separate_tr:
            params_ref["TE"] = te_ref[:n_TEref]
        else:
            params_ref["TE"] = te_ref[::2]
    else: # 3T
        rf_ref_dur = 1.2e-3
        tbp_ref = 4
        te_1 = d_te
        params_ref["TE"] = [round(te_1 + k*d_te, 5) for k in range(n_TEref)]
    if half_refscan:
        slices_ref = slices // 2
        slice_res_ref = slice_res * 2
    else:
        slices_ref = slices
        slice_res_ref = slice_res
    
    params_ref.update({"fov": fov*1e-3, "slices":slices_ref, "slice_res":slice_res_ref, "dist_fac": dist_fac, "res":res_refscan,
                        "flip_angle":flip_refscan, "readout_bw": bw_refscan, "rf_dur": rf_ref_dur, "tbp": tbp_ref, "separate_tr": separate_tr})

    if refscan == 1:
        params_ref["ref_lines"] = 30 # ecalib takes 24 lines as default
        params_ref["TE"] = [te_1]
    elif refscan == 2:
        params_ref["center_out"] = True # center out readout
    else:
        raise ValueError("Invalid refscan selection.")

    # make refscan
    gre_refscan_B0(seq, prot=prot, system=system, params=params_ref, grads_off=grads_off)

dur_until_ref = seq.duration()[0]
print(f"Sequence duration after reference scan: {dur_until_ref:.2f} s")

# Skope sync scans
if skope:
    n_sync = ismrmrd.xsd.userParameterLongType()
    n_sync.name = 'n_sync'
    n_sync.value = sync_scans # save number of sync scans
    hdr.userParameters.userParameterLong.append(n_sync)

    if measure_delay:
        trig_delay = 0 # measure at the beginning of the echo time delay
    else:
        trig_delay = te_delay.delay - skope_delay # measure 200us before spiral readout

    adc_sync = make_adc(system=system, num_samples=4000, dwell=dwelltime)
    adc_sync_delay = make_delay(d=ph.round_up_to_raster(calc_duration(adc_sync)+200e-6, decimals=5))
    trig = make_digital_output_pulse(channel='ext1', duration=system.grad_raster_time, delay=trig_delay)

    for j in range(sync_scans):
        seq.add_block(trig, te_delay)
        seq.add_block(adc_sync, adc_sync_delay)
        seq.add_block(make_delay(d=50e-3)) # some delay between triggers

        acq = ismrmrd.Acquisition()
        acq.setFlag(ismrmrd.ACQ_IS_DUMMYSCAN_DATA)
        prot.append_acquisition(acq)
        if j == sync_scans-1:
            ix_img = ismrmrd.xsd.userParameterLongType()
            ix_img.name = 'ix_img'
            ix_img.value = prot.number_of_acquisitions() # save the index of the first imaging acquistion
            hdr.userParameters.userParameterLong.append(ix_img)

    sync_scan_delay = make_delay(d=5)
    seq.add_block(sync_scan_delay) # Skope trigger receipt has dead time after sync scans

else:
    sync_scans = 0 

""" diffusion

The following code generates a Pulseq diffusion sequence.
Single-shot & multishot acquisitions are possible
For multishot, sufficient oversampling in kspace center has to be chosen, as a phase correction is needed.

"""
exc_phase = np.pi / 2 # set to 90° as in Siemens sequence

vol_TR_delay = vol_TR - (TR*slices_eff) if vol_TR is not None else None
if vol_TR is None:
    vol_TR = TR*slices_eff
print(f"Volume TR: {vol_TR:.3f} s.")

# save loop variables before prepscans
repetitions_ = repetitions
avgs_in_ = avgs_in
avgs_out_ = avgs_out
intl_eff_ = intl_eff
diff_list_ = diff_list.copy()

# run prepscans, then imaging scans
for prep in range(prepscans+1):
    if prep != prepscans:
        repetitions = avgs_out = avgs_in = intl_eff = 1
        diff_list = [{"bval": 0, "dir": np.zeros(3), "bDelta": 1}] # (a dummy scan with bval=0)
    else:
        repetitions = repetitions_
        avgs_in = avgs_in_
        avgs_out = avgs_out_
        intl_eff = intl_eff_
        diff_list = diff_list_.copy()
        g = []
        btens = []

    for rep in range(repetitions):
        for avg_out in range(avgs_out):
            for vol_ix, diff in enumerate(diff_list):
                
                if diff['bval'] == 0:
                    crusher_x.waveform = crusher_wf.copy()
                    crusher_y.waveform = crusher_wf.copy()
                    grad_refoc = copy(grad_refoc1)
                    
                    if diffSeq == 0:
                        trap_diff_z.amplitude = 0

                    if diffSeq == 1:
                        GxSinus_lin_first[0].waveform = np.zeros_like(GxSinus_lin_x_first.waveform)
                        GxSinus_lin_first[1].waveform = np.zeros_like(GxSinus_lin_y_first.waveform)
                        GxSinus_lin_first[2].waveform = np.zeros_like(GxSinus_lin_z_first.waveform)
                        GxSinus_lin_second[0].waveform = np.zeros_like(GxSinus_lin_x_second.waveform)
                        GxSinus_lin_second[1].waveform = np.zeros_like(GxSinus_lin_y_second.waveform)
                        GxSinus_lin_second[2].waveform = np.zeros_like(GxSinus_lin_z_second.waveform)

                        GxSinus_plan_first[0].waveform = np.zeros_like(GxSinus_plan_x_first.waveform)
                        GxSinus_plan_first[1].waveform = np.zeros_like(GxSinus_plan_y_first.waveform)
                        GxSinus_plan_first[2].waveform = np.zeros_like(GxSinus_plan_z_first.waveform)
                        GxSinus_plan_second[0].waveform = np.zeros_like(GxSinus_plan_x_second.waveform)
                        GxSinus_plan_second[1].waveform = np.zeros_like(GxSinus_plan_y_second.waveform)
                        GxSinus_plan_second[2].waveform = np.zeros_like(GxSinus_plan_z_second.waveform)
                    else:
                        GxSinus_first.waveform = np.zeros_like(GxSinus_first.waveform)
                        GySinus_first.waveform = np.zeros_like(GySinus_first.waveform)
                        GzSinus_first.waveform = np.zeros_like(GzSinus_first.waveform)
                        GxSinus_second.waveform = np.zeros_like(GxSinus_second.waveform)
                        GySinus_second.waveform = np.zeros_like(GySinus_second.waveform)
                        GzSinus_second.waveform = np.zeros_like(GzSinus_second.waveform)

                        GxSinus_plan_first[0].waveform = np.zeros_like(wf_GxSinus_sph_first)
                        GxSinus_plan_first[1].waveform = np.zeros_like(wf_GxSinus_sph_first)
                        GxSinus_plan_first[2].waveform = np.zeros_like(wf_GxSinus_sph_first)
                        GxSinus_plan_second[0].waveform = np.zeros_like(wf_GxSinus_sph_second)
                        GxSinus_plan_second[1].waveform = np.zeros_like(wf_GxSinus_sph_second)
                        GxSinus_plan_second[2].waveform = np.zeros_like(wf_GxSinus_sph_second)

                        GxSinus_lin_first[0].waveform = np.zeros_like(GxSinus_first.waveform)
                        GxSinus_lin_first[1].waveform = np.zeros_like(GxSinus_first.waveform)
                        GxSinus_lin_first[2].waveform = np.zeros_like(GxSinus_first.waveform)
                        GxSinus_lin_second[0].waveform = np.zeros_like(GxSinus_second.waveform)
                        GxSinus_lin_second[1].waveform = np.zeros_like(GxSinus_second.waveform)
                        GxSinus_lin_second[2].waveform = np.zeros_like(GxSinus_second.waveform)
                
                else:
                    crusher_x.waveform = crusher_wf.copy()
                    crusher_y.waveform = crusher_wf.copy()
                    grad_refoc = copy(grad_refoc1)

                    if diff['bDelta'] == -0.5: # planar
                        if diffSeq == 0:
                            trap_diff_z.amplitude = 0
                        if diffSeq == 1: # Attention: b_val_max is the maximal bvalue of all encodings and also has to be the maximum of the PTE
                            M = vec2vec_rotmat(np.array([0,0,1]), diff['dir'])
                            g_slice_spoil = np.sqrt(diff['bval']/b_val_max)*(wf_Gplan_x_first*M[2,0] + wf_Gplan_y_first*M[2,1])
                            t_grad = (np.arange(g_slice_spoil.shape[0]) + 0.5) * system.grad_raster_time
                            q_slice_spoil = system.gamma * cumulative_trapezoid(g_slice_spoil, t_grad,initial=0)
                            if q_slice_spoil[-1]<0:
                                M[:,1]*=-1
                            GxSinus_plan_first[0].waveform = np.sqrt(diff['bval']/b_val_max)*(wf_Gplan_x_first*M[0,0] + wf_Gplan_y_first*M[0,1])
                            GxSinus_plan_first[1].waveform = np.sqrt(diff['bval']/b_val_max)*(wf_Gplan_x_first*M[1,0] + wf_Gplan_y_first*M[1,1])
                            GxSinus_plan_first[2].waveform = np.sqrt(diff['bval']/b_val_max)*(wf_Gplan_x_first*M[2,0] + wf_Gplan_y_first*M[2,1])
                            GxSinus_plan_second[0].waveform = np.sqrt(diff['bval']/b_val_max)*(wf_Gplan_x_second*M[0,0] + wf_Gplan_y_second*M[0,1])
                            GxSinus_plan_second[1].waveform = np.sqrt(diff['bval']/b_val_max)*(wf_Gplan_x_second*M[1,0] + wf_Gplan_y_second*M[1,1])
                            GxSinus_plan_second[2].waveform = np.sqrt(diff['bval']/b_val_max)*(wf_Gplan_x_second*M[2,0] + wf_Gplan_y_second*M[2,1])
                        else:
                            M = vec2vec_rotmat(np.array([1,0,0]), diff['dir'])
                            g_slice_spoil = np.sqrt(diff['bval']/b_val_max)*np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_first*np.sqrt(3/2)*M[2,1] + wf_GySinus_sph_first*np.sqrt(3/2)*M[2,2])
                            t_grad = (np.arange(g_slice_spoil.shape[0]) + 0.5) * system.grad_raster_time
                            q_slice_spoil = system.gamma * cumulative_trapezoid(g_slice_spoil, t_grad,initial=0)
                            if q_slice_spoil[-1]<0:
                                M[:,1]*=-1
                            GxSinus_plan_first[0].waveform = np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_first*np.sqrt(3/2)*M[0,1] + wf_GySinus_sph_first*np.sqrt(3/2)*M[0,2])
                            GxSinus_plan_first[1].waveform = np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_first*np.sqrt(3/2)*M[1,1] + wf_GySinus_sph_first*np.sqrt(3/2)*M[1,2])
                            GxSinus_plan_first[2].waveform = np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_first*np.sqrt(3/2)*M[2,1] + wf_GySinus_sph_first*np.sqrt(3/2)*M[2,2])
                            GxSinus_plan_second[0].waveform = np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_second*np.sqrt(3/2)*M[0,1] + wf_GySinus_sph_second*np.sqrt(3/2)*M[0,2])
                            GxSinus_plan_second[1].waveform = np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_second*np.sqrt(3/2)*M[1,1] + wf_GySinus_sph_second*np.sqrt(3/2)*M[1,2])
                            GxSinus_plan_second[2].waveform = np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_second*np.sqrt(3/2)*M[2,1] + wf_GySinus_sph_second*np.sqrt(3/2)*M[2,2])
                    
                    elif diff['bDelta'] == 0: # spherical (not included for diffSeq==1)
                        trap_diff_z.amplitude = np.sqrt(diff['bval']/b_val_max)*trap_diff_z_ampl
                        GxSinus_first.waveform = np.sqrt(diff['bval']/b_val_max)*GxSinus_wf_first
                        GySinus_first.waveform = np.sqrt(diff['bval']/b_val_max)*GySinus_wf_first
                        GzSinus_first.waveform = np.sqrt(diff['bval']/b_val_max)*GzSinus_wf_first
                        GxSinus_second.waveform = np.sqrt(diff['bval']/b_val_max)*GxSinus_wf_second
                        GySinus_second.waveform = np.sqrt(diff['bval']/b_val_max)*GySinus_wf_second
                        GzSinus_second.waveform = np.sqrt(diff['bval']/b_val_max)*GzSinus_wf_second
                    
                    else: # linear
                        if diffSeq == 0:
                            trap_diff_z.amplitude = 0
                        if diffSeq == 1:
                            GxSinus_lin_first[0].waveform = np.sqrt(diff['bval']/b_val_max)*wf_Glin_z_first * diff['dir'][0]
                            GxSinus_lin_first[1].waveform = np.sqrt(diff['bval']/b_val_max)*wf_Glin_z_first * diff['dir'][1]
                            GxSinus_lin_first[2].waveform = np.sqrt(diff['bval']/b_val_max)*wf_Glin_z_first * diff['dir'][2]
                            GxSinus_lin_second[0].waveform = np.sqrt(diff['bval']/b_val_max)*wf_Glin_z_second * diff['dir'][0]
                            GxSinus_lin_second[1].waveform = np.sqrt(diff['bval']/b_val_max)*wf_Glin_z_second * diff['dir'][1]
                            GxSinus_lin_second[2].waveform = np.sqrt(diff['bval']/b_val_max)*wf_Glin_z_second * diff['dir'][2]
                        else:
                            GxSinus_lin_first[0].waveform = np.sqrt(diff['bval']/b_val_max)*GxSinus_wf_first*np.sqrt(3) * diff['dir'][0]
                            GxSinus_lin_first[1].waveform = np.sqrt(diff['bval']/b_val_max)*GxSinus_wf_first*np.sqrt(3) * diff['dir'][1]
                            GxSinus_lin_first[2].waveform = np.sqrt(diff['bval']/b_val_max)*GxSinus_wf_first*np.sqrt(3) * diff['dir'][2]
                            GxSinus_lin_second[0].waveform = np.sqrt(diff['bval']/b_val_max)*GxSinus_wf_second*np.sqrt(3) * diff['dir'][0]
                            GxSinus_lin_second[1].waveform = np.sqrt(diff['bval']/b_val_max)*GxSinus_wf_second*np.sqrt(3) * diff['dir'][1]
                            GxSinus_lin_second[2].waveform = np.sqrt(diff['bval']/b_val_max)*GxSinus_wf_second*np.sqrt(3) * diff['dir'][2]

                for avg_in in range(avgs_in):
                    for n in range(intl_eff):
                        # slice ordering acc to Siemens method
                        if slices_eff%2 == 1:
                            slc = 0
                        else:
                            slc = 1
                        for slc_ctr in range(slices_eff):
                            if slc_ctr==int(slices_eff/2+0.5):
                                if slices_eff%2 == 1:
                                    slc = 1
                                else:
                                    slc = 0

                            # initiate for b-tensor calculation
                            seq_btens = Sequence()

                            # Add (fatsat and) excitation pulse, always use RF spoiling for fat sat pulse
                            if fatsat:
                                seq.add_block(spoiler_x_neg, spoiler_y_neg, spoiler_z_neg)
                                rf_fatsat.phase_offset = rf_phase / 180 * np.pi - 2 * np.pi * rf_fatsat.freq_offset * fatsat_dur/2
                                seq.add_block(rf_fatsat, fatsat_del)
                                rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
                                rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
                                seq.add_block(spoiler_x, spoiler_y, spoiler_z)

                            rf.freq_offset = gz.amplitude * slice_res * (slc - (slices_eff - 1) / 2) * (1+dist_fac*1e-2)
                            rf_refoc.freq_offset = gz_refoc.amplitude * slice_res * (slc - (slices_eff - 1) / 2) * (1+dist_fac*1e-2)
                            rf.phase_offset = exc_phase - 2 * np.pi * rf.freq_offset * rf_dur/2
                            rf_refoc.phase_offset = - 2 * np.pi * rf_refoc.freq_offset * rf_refoc_dur/2

                            seq.add_block(rf,gz,rf_del)
                            seq.add_block(gz_rew)
                            
                            seq_btens_exc = Sequence()
                            if btens_slice and (gz is not None) and (gz_rew is not None):
                                seq_btens_exc.add_block(rf,gz,rf_del)
                                half_slice_idx = round((calc_duration(rf,gz,rf_del)-exc_to_rew)/system.grad_raster_time) #seq_btens_exc.gradient_waveforms().shape[1]//2
                                seq_btens_exc.add_block(gz_rew)
                            else:
                                seq_btens_exc.add_block(rf,rf_del)
                                half_slice_idx = round((calc_duration(rf,rf_del)-exc_to_rew)/system.grad_raster_time) #seq_btens_exc.gradient_waveforms().shape[1]//2
                                seq_btens_exc.add_block(make_delay(d=rew_dur))

                            # diffusion block
                            seq.add_block(refoc_delay)
                            seq_btens_exc.add_block(refoc_delay)
                            if diffSeq == 0:
                                seq.add_block(trap_diff_z)
                                seq_btens.add_block(trap_diff_z)
                            if diff['bDelta'] == -0.5: # planar
                                seq.add_block(GxSinus_plan_first[0],GxSinus_plan_first[1],GxSinus_plan_first[2],sinusTime_first)
                                seq_btens.add_block(GxSinus_plan_first[0],GxSinus_plan_first[1],GxSinus_plan_first[2],sinusTime_first)
                            elif diff['bDelta'] == 0: # spherical
                                seq.add_block(GxSinus_first,GySinus_first,GzSinus_first,sinusTime_first)
                                seq_btens.add_block(GxSinus_first,GySinus_first,GzSinus_first,sinusTime_first)
                            else: # linear
                                seq.add_block(GxSinus_lin_first[0],GxSinus_lin_first[1],GxSinus_lin_first[2],sinusTime_first)
                                seq_btens.add_block(GxSinus_lin_first[0],GxSinus_lin_first[1],GxSinus_lin_first[2],sinusTime_first)
                            
                            # compute 0th. moment after first diffusion gradients to ensure crsuher moment
                            g_spoil = seq_btens.gradient_waveforms()/system.gamma
                            g_spoil_exc = seq_btens_exc.gradient_waveforms()[:,half_slice_idx:]/system.gamma
                            g_spoil = np.concatenate((g_spoil_exc,g_spoil),axis=1)
                            g_spoil = g_spoil.transpose((1,0)) # [time,axis] [T/m]
                            t_grad = (np.arange(g_spoil.shape[0]) + 0.5) * system.grad_raster_time # [s]
                            q_t = np.zeros_like(g_spoil)
                            for i in range(g_spoil.shape[1]):
                                q_t[:,i] = system.gamma * cumulative_trapezoid(g_spoil[:,i], t_grad,initial=0) # [rad/m]                                        
                            if q_t[-1,0]<0:
                                crusher_x.waveform = crusher_wf.copy()*(-1)
                            if q_t[-1,1]<0:
                                crusher_y.waveform = crusher_wf.copy()*(-1)

                            # refocussing block
                            before_refoc_idx = seq_btens.gradient_waveforms().shape[1]
                            seq.add_block(rf_refoc,grad_refoc,crusher_x,crusher_y)
                            if btens_slice:
                                seq_btens.add_block(rf_refoc,grad_refoc,crusher_x,crusher_y)
                            else:
                                seq_btens.add_block(make_delay(d=refoc_dur))
                            after_refoc_idx = seq_btens.gradient_waveforms().shape[1]
                            
                            # diffusion block
                            if diff['bDelta'] == -0.5: # planar
                                seq.add_block(GxSinus_plan_second[0],GxSinus_plan_second[1],GxSinus_plan_second[2],sinusTime_second)
                                seq_btens.add_block(GxSinus_plan_second[0],GxSinus_plan_second[1],GxSinus_plan_second[2],sinusTime_second)
                            elif diff['bDelta'] == 0: # spherical
                                seq.add_block(GxSinus_second,GySinus_second,GzSinus_second,sinusTime_second)
                                seq_btens.add_block(GxSinus_second,GySinus_second,GzSinus_second,sinusTime_second)
                            else: # linear
                                seq.add_block(GxSinus_lin_second[0],GxSinus_lin_second[1],GxSinus_lin_second[2],sinusTime_second)
                                seq_btens.add_block(GxSinus_lin_second[0],GxSinus_lin_second[1],GxSinus_lin_second[2],sinusTime_second)
                            if diffSeq == 0:
                                seq.add_block(trap_diff_z)
                                seq_btens.add_block(trap_diff_z)

                            # Skope trigger - keep minimum distance of 200us between subsequent triggers
                            if skope and slc_ctr%trig_skip==0 and slices_eff-slc_ctr >= trig_skip and prep == prepscans:
                                trig_ctr += 1
                                seq.add_block(trig, te_delay)
                            else:
                                seq.add_block(te_delay)
                            seq_btens.add_block(te_delay)

                            # calculate b-tensor for TVE analysis
                            if slc == 0 and prep == prepscans:
                                g.append(seq_btens.gradient_waveforms()/system.gamma)
                                half_idx = before_refoc_idx + (after_refoc_idx-before_refoc_idx)//2
                                g[-1][:,half_idx:] = (-1)*g[-1][:,half_idx:]

                                g_exc = seq_btens_exc.gradient_waveforms()[:,half_slice_idx:]/system.gamma
                                g[-1] = np.concatenate((g_exc,g[-1]),axis=1)
                                
                                gamma = system.gamma * 2 * np.pi
                                btens.append(np.zeros((3,3)))
                                for i in range(3):
                                    for j in range(3):
                                        btens[-1][i,j] = 1e-6*gamma**2*(np.sum((np.cumsum(g[-1][i,:])*np.cumsum(g[-1][j,:])*system.grad_raster_time**2)*system.grad_raster_time)) # [3,3] [s/mm²]

                            # spiral readout block
                            spiral_block = [spirals[n*redfac]['spiral'][0], spirals[n*redfac]['spiral'][1], adc_delay]
                            if sms and (sms_type == 1 or sms_type == 2):
                                spiral_block.append(sms_blips)
                            if prep == prepscans:
                                spiral_block.append(adc)
                            seq.add_block(*spiral_block)

                            # delay for TR
                            if fatsat:
                                min_tr = rf.delay + ph.round_up_to_raster(rf_dur/2, decimals=5) + TE + adc_delay.delay + calc_duration(spirals[n*redfac]['reph'][0],spirals[n*redfac]['reph'][1]) + fatsat_del.delay + 2*spoiler_dur
                            else:
                                min_tr = rf.delay + ph.round_up_to_raster(rf_dur/2, decimals=5) + TE + adc_delay.delay + calc_duration(spirals[n*redfac]['reph'][0],spirals[n*redfac]['reph'][1],spoiler_z)
                            if TR < min_tr:
                                raise ValueError('Minimum TR is {} ms.'.format(min_tr*1e3))
                            tr_delay = make_delay(d=ph.round_up_to_raster(TR-min_tr, decimals=5))
                            seq.add_block(tr_delay)

                            # rephaser (with spoiler, if no fatsat)
                            if fatsat:
                                seq.add_block(spirals[n*redfac]['reph'][0],spirals[n*redfac]['reph'][1])
                            else:
                                seq.add_block(spirals[n*redfac]['reph'][0],spirals[n*redfac]['reph'][1], spoiler_z)

                            # add protocol information
                            if prep == prepscans:
                                for seg in range(num_segments):
                                    acq = ismrmrd.Acquisition()
                                    if (n == intl_eff-1) and (seg == num_segments-1):
                                        acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
                                        if slc_ctr == slices_eff-1 and avg_out == avgs_out-1 and avg_in == avgs_in-1 and rep == repetitions-1 and vol_ix == len(diff_list)-1 and prep == prepscans:
                                            acq.setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
                                    acq.idx.kspace_encode_step_1 = n
                                    acq.idx.slice = slc
                                    acq.idx.contrast = vol_ix
                                    acq.idx.average = max(avg_in, avg_out)
                                    acq.idx.repetition = rep
                                    acq.idx.segment = seg
                                    acq.user_int[0] = diff['bval']
                                    acq.user_float[:3] = diff['dir']
                                    acq.user_int[1] = int(np.floor(diff['bDelta']))
                                    
                                    # save gradient only in first segment to save space
                                    if seg == 0:
                                        # use the trajectory field for the gradient array
                                        acq.resize(trajectory_dimensions = save_sp.shape[1]+blip_dims, number_of_samples=save_sp.shape[2], active_channels=0)
                                        acq.traj[:,:2] = np.swapaxes(save_sp[n*redfac],0,1) # [samples, dims]
                                        if sms and (sms_type == 1 or sms_type == 2):
                                            acq.traj[:,2] = sms_blips.waveform / system.gamma
                                    prot.append_acquisition(acq)
                            
                            slc += 2 # interleaved slice acquisition

                if vol_TR_delay is not None:
                    seq.add_block(make_delay(d=vol_TR_delay))

                        # slices
                    # intl
                # avg_in
            # contrast
        # avg_out
    # reps
# prepscans

print(f"Sequence duration: {seq.duration()[0]:.2f} s")

# save b-values, directions and bDeltas as arrays
prot.append_array("b_values", np.asarray(bval_list, dtype=np.float32))
prot.append_array("Directions", np.asarray(dir_list, dtype=np.float32))
prot.append_array("bDeltas", np.asarray(bDelta_list, dtype=np.float32))

# save calculated b-tensors
btens = np.asarray(btens)

if not os.path.isdir('qti_waveforms/btens'):
    os.makedirs('qti_waveforms/btens')
np.save('qti_waveforms/btens/btens_'+date+'_'+seq_name, btens)   

if skope:
    print(f"Number of Skope triggers: {trig_ctr}.")
    seq.set_definition("SkopeNrDynamics", trig_ctr)
    seq.set_definition("SkopeNrSyncDynamics", sync_scans)
    seq.set_definition("SkopeAqDuration_ms", t_skope)
    seq.set_definition("SkopeInterleaveTR_ms", 1e3*min_dist)

#%% visualize gradient waveforms

# convert list of gradients into numpy array
g_t = np.zeros((g[0].shape[1],g[0].shape[0],len(g))) # [time,axis,volumes] [T/m]
for vol in range(len(g)):
    g_t[:,:,vol] = g[vol].transpose((1,0))

# plot gradient waveforms
if plotSpectrum:

    # use g-list for plotting of the gradients and encoding spectrum
    g_t = g[8].transpose((1,0)) # [time,axis] [T/m]
    dt = system.grad_raster_time # [s]
    freq_axis, b_w_cmplx, b_trace_w, b_fromSum, w_centroid = computeEncodingSpectrum(g_t,dt,gamma=2*np.pi*system.gamma,df=1,saveplots=seq_name+'_bw.png') # [1/s], [freq,3,3] [s²/m²], [s²/m²], [s/m²], [rad/s]

    # gradients
    g_t = g[8].transpose((1,0))
    t_grad = (np.arange(g_t.shape[0]) + 0.5) * dt # [s]
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    labels = [r"g$_x$(t) [mT/m]", r"g$_y$(t) [mT/m]", r"g$_z$(t) [mT/m]"]
    colors =['#1f77b4', '#ff7f0e', '#2ca02c']
    for i in range(3):
        axs[i].plot(t_grad, g_t[:, i]*1e3, color=colors[i], lw=3)
        axs[i].plot([t_grad[t_grad.shape[0]//2],t_grad[t_grad.shape[0]//2]],[-70,70],color='k', lw=2)
        axs[i].set_ylabel(labels[i], fontsize=20)
        axs[i].tick_params(axis='both', labelsize=20)
        axs[i].grid()
    axs[2].set_xlabel("t [ms]", fontsize=20)
    plt.suptitle(seq_name, fontsize=20)
    plt.tight_layout()
    plt.savefig(seq_name+'_grads.png',dpi=300)

    # compute q(t)
    q_t = np.zeros_like(g_t)
    for i in range(g_t.shape[1]):
        q_t[:,i] = system.gamma * np.cumsum(g_t[:,i])*dt # [rad/m]

    center_idx = int(t_grad.shape[0]/2)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(q_t[:center_idx,0], q_t[:center_idx,1] , q_t[:center_idx,2], 'r',alpha=0.7, label='before 180° pulse')
    ax.plot(q_t[center_idx:,0], q_t[center_idx:,1] , q_t[center_idx:,2], 'b',alpha=0.7, label='after 180° pulse')
    ax.set_xlabel("x [1/m]", fontsize=15)
    ax.set_ylabel("y [1/m]", fontsize=15)
    ax.set_zlabel("z [1/m]", fontsize=15,labelpad=0)
    plt.tick_params(axis='both', labelsize=10, pad=-1)
    plt.legend(fontsize=10)
    plt.title(r"$\vec{q}(t)$", fontsize=20)
    plt.grid()
    fig.subplots_adjust(left=0.0, right=0.99, bottom=0.1, top=0.9)
    ax.view_init(elev=30, azim=-60)
    plt.savefig(seq_name+'_q.png',dpi=300)

    # zoom
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(q_t[:center_idx,0], q_t[:center_idx,1] , q_t[:center_idx,2], 'r',alpha=0.7, label='before 180° pulse')
    ax.plot(q_t[center_idx:,0], q_t[center_idx:,1] , q_t[center_idx:,2], 'b',alpha=0.7, label='after 180° pulse')
    ax.set_xlabel("x [1/m]", fontsize=15)
    ax.set_ylabel("y [1/m]", fontsize=15)
    ax.set_zlabel("z [1/m]", fontsize=15,labelpad=0)
    plt.tick_params(axis='both', labelsize=10, pad=-1)
    ax.set_xlim([-5000,5000])
    ax.set_ylim([-5000,5000])
    ax.set_zlim([-5000,5000])
    plt.legend(fontsize=10)
    plt.title(r"$\vec{q}(t)$", fontsize=20)
    plt.grid()
    fig.subplots_adjust(left=0.0, right=0.99, bottom=0.1, top=0.9)
    ax.view_init(elev=30, azim=-60)
    plt.savefig(seq_name+'_qzoom.png',dpi=300)

# test crusher moment of 2 k-spaces
if testCrushingMoments:

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    firstPTE = True
    firstLTE = True
    firstSTE = True

    aboveMoment_dict = []

    for volume in range(len(diff_list)):
        g_t = g[volume].transpose((1,0)) # [time,axis] [T/m]
        dt = system.grad_raster_time # [s]
        t_grad = (np.arange(g_t.shape[0]) + 0.5) * dt # [s]
        te_half_idx = t_grad.shape[0]//2

        if bDelta_list[volume]==-0.5:
            col='b'
            lab='PTE'
            if firstPTE:
                showlabel=True
                firstPTE=False
            else:
                showlabel=False
        if bDelta_list[volume]==1:
            col='orange'
            lab='LTE'
            if firstLTE:
                showlabel=True
                firstLTE=False
            else:
                showlabel=False
        if bDelta_list[volume]==0:
            col='g'
            lab='STE'
            if firstSTE:
                showlabel=True
                firstSTE=False
            else:
                showlabel=False

        # compute dephasing waveforms vec(q(t))
        q_t = np.zeros_like(g_t)
        for i in range(g_t.shape[1]):
            q_t[:,i] = gamma * cumulative_trapezoid(g_t[:,i], t_grad,initial=0) # [rad/m]
            q_t[:,i] /= 2*np.pi # [1/m]
            
            if np.round(np.abs(q_t[te_half_idx,i]),2)<np.round(spoiler_area,2):
                coltrue='r'
                aboveMoment_dict.append({'volume':volume,'bval':diff_list[volume]['bval'],'bDelta':diff_list[volume]['bDelta'],'axis':i,'moment':q_t[te_half_idx,i]})
                print(f'{np.abs(q_t[te_half_idx,i])} < {spoiler_area}')
            else:
                coltrue=col
            
            if showlabel:
                axs[i].plot(volume, q_t[te_half_idx,i], 'x',color=coltrue, label=lab)
            else:
                axs[i].plot(volume, q_t[te_half_idx,i], 'x',color=coltrue)

    for i in range(g_t.shape[1]):
        axs[i].plot([0,len(diff_list)], [spoiler_area,spoiler_area],color='k', label=r'+2$\cdot$kmax')
        axs[i].plot([0,len(diff_list)], [-spoiler_area,-spoiler_area],color='k', label=r'-2$\cdot$kmax')

    axs[0].set_ylabel("x-axis", fontsize=20)
    axs[1].set_ylabel("y-axis", fontsize=20)
    axs[2].set_ylabel("z-axis", fontsize=20)
    axs[2].set_xlabel("diffusion volume", fontsize=20)

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    plt.suptitle('Crusher moment', fontsize=20)
    plt.tight_layout()
    plt.savefig(seq_name+'_crusherMoments.png',dpi=300)

# test if still trace and shape is correct after imaging gradients
for elem in btens:
    evals, evecs = np.linalg.eig(elem)
    idx = evals.argsort()   
    evals = evals[idx] 
    evecs = evecs[:,idx]

    b_comp = sum(evals)
    # ensure Haeberlen convention |bZZ − b/3|>|bXX − b/3|>|bYY − b/3|
    evals_subtract = abs(evals-b_comp/3)
    evals_subtract = np.argsort(evals_subtract)[::-1] # ordered with the maximum value as the first element
    evals = evals[evals_subtract]
    b_delta_comp = (evals[0] - (evals[1] + evals[2]) / 2) / b_comp
    b_eta_comp = 3/2 * (evals[2]-evals[1])/(b_comp*b_delta_comp) # not precise for b_delta --> 0
    print(f'b-value = {b_comp} s/m²')
    print(f'bdelta = {b_delta_comp}')
    print(f'beta = {b_eta_comp}')

#%% save sequence and write sequence & protocol
    
# save seq
seqfile = f"{filename}.seq"
seq.write(seqfile)
prot.close()
