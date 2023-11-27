# Spiral Pulseq Sequence

#%%

import numpy as np
import matplotlib.pyplot as plt
import json
import ismrmrd
import os
import stat
import datetime
from pathlib import Path # create directories
import shutil # copy files
from copy import copy
import sys
import subprocess
import re

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
from pypulseq.make_arbitrary_rf import make_arbitrary_rf
from pypulseq.make_sigpy_pulse import sigpy_n_seq
from pypulseq.sigpy_pulse_opts import SigpyPulseOpts

from pypulseq.make_ptx_pulse import make_ptx_pulse

from pypulseq.points_to_waveform import points_to_waveform

import spiraltraj
from sigpy.mri import rf as rfsig
import pulseq_helper as ph
from diffusion import diff_params, calc_bval, DISCSo
from prot import create_hdr
from pulses.ext_pulses import Ext_pulse
from gre_refscan_B0 import gre_refscan_B0
import random
from scipy.io import savemat, loadmat
from divide_helper import get_ismrmrd_arrays
from dipy.core.geometry import vec2vec_rotmat

#%% Parameters 
"""
PyPulseq units (SI): 
time:       [s] (not [ms] as in documentation)
spatial:    [m]
gradients:  [Hz/m] (gamma*T/m)
grad area:  [1/m]
flip angle: [rad]

Some units get converted below, others have to stay in non-SI units as spiral calculation needs different units.

SLR pulses should always be used for the diffusion sequence!!!
For multiband imaging, the use of VERSE pulses is recommended (minimum-time VERSE with energy constraint).
It is also currently recommended to use the external pulses "90_SLR_4.90_SLR_4" and "180_SLR_4.180_SLR_4", which are also used in the MGH diffusion sequence
The pulse energy of the VERSE pulses will be printed. For a TR of 100ms, the combined energy of exc&ref should not exceed 100, if fatsat is used

Add new parameters also to the protocol parameter json file at the end of this script.
Use custom version of Pypulseq (important for delays): https://github.com/mavel101/pypulseq (branch dev_mv)
WIP: Change to Pulseq 1.4 (PyPulseq is still on 1.3.1 though)

Caution: Trapezoidal gradients will get a sign change in the physical coordinate system in old/compat rotation matrix mode

PTX support only for universal ini-file PTX pulses in unity orientation (no_rot ptx pulse parameter is always True).
Update: With the new Pulseq ptx spokes sequence, the no_rot parameter is not used anymore as blips are automatically rotated back.

"""
# General
B0              = 6.983   # field strength [T]
seq_dest        = 'None'  # save parameter: 'None': save nothing, 'scanner': export to scanner 'sim': export to simulation 'scantest': export seq only to scanner
scanner         = 'skyra'   # scanner for acoustic resonance check - only 
grads_off       = False   # turn off gradients in reference scan (just for simulation of ECC)
seq_name        = 'NowLteComp' # sequence/protocol filename
meas_date       = None # measurement date [YYYYMMDD] - if None use todays date 
plot            = False      # plot sequence
check_TE        = False      # check correct RF spacing of diffusion sequence
test_report     = False      # print the Pypulseq test report

# Sequence - Contrast and Geometry
seq_type        = 'diffusion' # sequence type (diffusion or gre)
fov             = 210         # field of view [mm]
TR              = 139        # repetition time [ms]
TE              = 108.5          # echo time [ms]
res             = 1.5         # in plane resolution [mm]
slice_res       = 1.5         # slice thickness [mm]
dist_fac        = 0      # distance factor for slices [%]
slices          = 6          # number of slices
averages        = 1           # number of averages
inner_avg       = True      # do averages in inner loop
repetitions     = 1         # number of repetitions

refscan         = 2       # 0: no refscan, 1: normal refscan, 2: B0 mapping refscan, 3: B0 mapping 3D refscan
res_refscan     = 3         # resolution of refscan, if B0 mapping is performed, 2mm is typically sufficient
bw_refscan      = 1000       # Bandwidth of the reference scan [Hz]
flip_refscan    = 25        # reference scan flip angle
half_refscan    = False     # collect only half number of slices in refscan (with doubled slice thickness) - for 1mm datasets
separate_tr     = False     # separate TRs for echoes of B0 mapping refscans
prepscans       = 5          # number of preparation/dummy scans
noisescans      = 16          # number of noise scans

# ADC
os_factor       = 2         # oversampling factor (automatic 2x os from Siemens is not applied)
max_adc         = 32768       # maximum number of samples per ADC (originally 8192 was max, but VE12U seems to accept higher values)

# RF
if seq_type == 'gre':
    flip_angle  = 10          # flip angle of excitation pulse [°]
if seq_type == 'diffusion':
    flip_angle  = 90

ptx             = False        # use pTx pulses
ptx_path_exc    = "pulses/ptx/" # path of the ptx excitation pulse (used for slice rewinder calculation), set to None if rewinder is part of ptx pulse
slr_rf          = False       # True: use Sigpys SLR pulse design, False: use external or sinc pulses
ext_rf          = True        # True: read external RF, only if SLR_rf is false
ext_path_exc    = 'pulses/slr_ext/90_SLR_4.90_SLR_4.pta' # path of external excitation pulse
ext_path_ref    = 'pulses/slr_ext/180_SLR_4.180_SLR_4.pta' # path of external refocusing pulse
verse_rf        = 0           # 0: no VERSE, 1: min SAR VERSE, 2: min time verse (recommended), 3: mintverse (hargreaves)
b1_max_fac      = [1,1]        # only verse_rf=2/3: factor to limit maximum B1 amplitude (in units of initial RF) [excitation, refocusing]
energy_fac      = [0.8,0.8]    # only verse_rf=2/3: factor to limit maximum energy (in units of initial RF) [excitation, refocusing]
rf_dur          = 6            # RF duration [ms]
tbp_exc         = 3           # time bandwidth product excitation pulse
rf_refoc_dur    = 8           # refocusing pulse duration [ms]
tbp_refoc       = 3           # time bandwidth product refocusing pulse
refoc_fac       = 1           # factor for slice thickness of refocusing pulse (=1 in Siemens diffusion seq)
rf_spoiling     = False       # RF spoiling
sms             = True        # multiband imaging?
sms_factor      = 2           # multiband factor
sms_type        = 0           # SMS type - 0: only SMS RF, 1: blipped spiral, 2: wave-like/sinus blips in z, 2: stacked spiral (only multishot, only GRE, not really tested) 
kz_steps        = 1           # number of steps in slice direction blipped spiral (should be <sms_factor)
wave_periods    = 7           # only SMS type 2: number of wave periods

fatsat          = True        # Fat saturation pulse
fatsat_tbp      = 2.1         # tbp of fatsat pulse [ms] (BW is fixed at 1000Hz atm)

# Gradients
max_slew        = 182         # maximum slewrate [T/m/s] (system limit)
spiral_slew     = 145         # maximum slew rate of spiral gradients - for pre Emph: set lower than max_slew
max_grad        = 55          # maximum gradient amplitude [mT/m] (system limit)
max_grad_sp     = 42         # maximum gradient amplitude of spiral gradients - for pre_emph: set lower than max_grad

Nintl           = 3           # spiral interleaves
redfac          = 3           # reduction/acceleration factor
spiraltype      = 1           # 1: Spiral Out, 4: ROI
spiral_os       = 1           # variable density spiral oversampling in center
trans_beg       = 0.25        # variable density transition beginning (between 0 and 1)
trans_end       = 0.28         # transition end (>trans_beg & between 0 and 1)

pre_emph        = False       # Gradient Pre-Emphasis based on GIRF
skope           = True       # add trigger for skope measurement
measure_delay   = False       # if False start skope measurement directly before spirals, if True start at the beginning of echo time delay
sync_scans      = 10          # number of Skope sync_scans   

# Diffusion
# number of b=0 images should be appr. directions/3.59 c.f. Kingsley: Intr. to DTI Part III (2006)
diff_slewrate   = 180          # diffusion gradient slewrate
diff_maxgrad    = 68         # diffusion gradient max grad strength
read_axon       = False        # read b-values and directions for axon diameter measurement (requires dipy)
b_val           = [[0,100,800],[0,2117],[0,100,800,2117]] # b-values [s/mm^2] (b=0 has to be included) [PTE, STE, LTE]
directions      = [[4,7,9],[8,30],[20,9,50,15]]  # number of acquisitions for each b-value [PTE, STE, LTE]
bDelta          = [-0.5,0,1] # shape of diffusion tensor [PTE, STE, LTE]
btens_slice     = 1             # include slice selection gradient and crusher in the b-tensor calculation
delta           = 9          # not used; duration of diffusion gradients [ms], spacing is automatically calculated
vol_TR          = None       # volume TR [s], if None take minimum volume TR

NOW             = 2          # 0: No use of NOW, use qMas instead with LTE tuned, 1: use NOW with LTE tuned, 2: use NOW with LTE detuned (choose file Detuned or Comp below), 3: get spacings for NOW optimization
use_prot        = False       # if you want to use an old protocol
prot_file       = '/home/niesens/Documents/Master/Protocols/20230720/20230720_qMas_LTE_2b283.h5' # old protocol path

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
delta       *= 1e-3 # [s]

# calculate effective interleaves
intl_eff = int(Nintl/redfac)

# set spoiler area
spoiler_area = 2/min(slice_res,1e-3*res) # 2 k-spaces
amp_spoil, ftop_spoil, ramp_spoil = ph.trap_from_area(spoiler_area, system, slewrate=90, max_grad=40e-3) # reduce slew rate to avoid stimulation

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
    if seq_type == 'diffusion' and sms_type == 3:
        sms_type = 0
        print("Set SMS type to 0, as type 3 are only available for GRE sequence")
    if sms_type == 3 and intl_eff == 1:
        raise ValueError('SMS type 3 only valid for multishot acquisitions.')
    if sms_factor > 2:
        mb_phs = 'quad_mod' # saves SAR and peak amp
    else:
        mb_phs = 'None'
else:
    slices_eff = slices
    sms_factor = 1
    sms_type = 0

if ptx:
    verse_rf = False
    slr_rf = False

if (half_refscan or refscan == 3) and slices%2 != 0:
    raise ValueError('3D refscan and half_refscan only possible for even slice number.')

if skope:
    skope_delay = 200e-6 # delay/gradient free interval after Skope trigger
    min_dist = 150e-3 # minimum trigger distance due to relaxation
    if seq_type == 'diffusion':
        trig_skip = int(np.ceil(min_dist/TR))
        if trig_skip > 1:
            print(f"TR too low to capture all triggers (minimum trigger distance 200ms). Only every {trig_skip}th trigger is captured.")
    else:
        trig_skip = 1 # trigger skipping not implemented in GRE
        if TR < min_dist:
            print('Warning: TR should be at least 200 ms for Skope measurements')
else:
    skope_delay = 0
    trig_skip = 0

if seq_type != 'gre' and seq_type != 'diffusion':
    raise ValueError('Choose gre or diffusion as sequence type')

if Nintl/redfac%1 != 0:
    raise ValueError('Number of interleaves is not multiple of reduction factor')

if int(fov/res+0.5) % 2:
    raise ValueError(f'Matrix size {int(fov/res+0.5)} (FOV/resolution) is not even.') 

if spiraltype!=1 and spiraltype!=4:
    ValueError('Right now only spiraltype 1 (spiral out) and 4 (ROI) possible.')

if redfac > 1 and refscan==0:
    print("WARNING: Cartesian reference scan is not activated.")

if plot or check_TE:
    refscan = False
    prepscans = 0
    noisescans = 0
    print("We do plotting or a TE check, so refscan,prepscans and noisescan are disabled.")
    if check_TE:
        fatsat = False
        print("We do a TE check, so fatsat is disabled.")

if grads_off:
    seq_dest = "sim"
    print("Refscan gradients turned off, set seq_dest to sim.")

if seq_type=='diffusion':
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
    if not ptx and rf_dur==rf_refoc_dur and tbp_exc==tbp_refoc:
        raise ValueError('Do not choose same duration and TBP for excitation and refocusing pulse. Crashes the sequence due to a Pulseq or Pypulseq bug.')

#%% RF Pulse and slab/slice selection gradient

# make rf pulse and calculate duration of excitation and rewinding
if slr_rf:
    sigpy_cfg = SigpyPulseOpts(pulse_type='slr', ptype='st')
    if flip_angle == 90:
        sigpy_cfg.ptype = 'ex'
        sigpy_cfg.cancel_alpha_phs = True
    rf, gz, gz_rew, rf_del = sigpy_n_seq(flip_angle=flip_angle*np.pi/180, system=system, duration=rf_dur, slice_thickness=slice_res,
                        time_bw_product=tbp_exc, pulse_cfg=sigpy_cfg, use='excitation', return_gz=True, return_delay = True, disp=False)
elif ext_rf:
    ext_p = Ext_pulse(ext_path_exc)
    ext_p.calc_rf_grad(duration=rf_dur, slc_thickness=slice_res)
    gz = make_trapezoid(channel='z', amplitude=ext_p.grad_amp, flat_time=rf_dur, system=system)
    if gz.rise_time < system.rf_dead_time:
        gz = make_trapezoid(channel='z', amplitude=ext_p.grad_amp, flat_time=rf_dur, rise_time=system.rf_dead_time, system=system)
    rf = make_arbitrary_rf(signal=ext_p.rf, flip_angle=flip_angle*np.pi/180, delay=gz.rise_time, system=system)
    gz_rew = make_trapezoid(channel='z', area=-gz.area/2,system=system)
    rf_del = make_delay(d=calc_duration(gz))
    tbp_exc = gz.amplitude * slice_res * rf_dur # calculate tbp of external pulse for multiband modulation
else:
    rf, gz, gz_rew, rf_del = make_sinc_pulse(flip_angle=flip_angle*np.pi/180, system=system, duration=rf_dur, slice_thickness=slice_res,
                            apodization=0.5, time_bw_product=tbp_exc, use='excitation', return_gz=True, return_delay = True)

if sms and not verse_rf:
    band_sep  = slice_sep/slice_res*tbp_exc # normalized distance between slices
    rf.signal = rfsig.multiband.mb_rf(rf.signal, n_bands=sms_factor, band_sep=band_sep, phs_0_pt=mb_phs)       

# refocusing pulse - increase slice thickness slightly for better refocusing
if seq_type=='diffusion':
    if slr_rf:
        sigpy_cfg_ref = SigpyPulseOpts(pulse_type='slr', ptype='se')
        rf_refoc, gz_refoc, _ = sigpy_n_seq(flip_angle=2*flip_angle*np.pi/180, system=system, duration=rf_refoc_dur, slice_thickness=slice_res*refoc_fac,
                            time_bw_product=tbp_refoc, pulse_cfg=sigpy_cfg_ref, use='refocusing', return_gz=True, disp=False)
    elif ext_rf:
        ext_p_ref = Ext_pulse(ext_path_ref)
        ext_p_ref.calc_rf_grad(duration=rf_refoc_dur, slc_thickness=slice_res*refoc_fac)
        gz_refoc = make_trapezoid(channel='z', amplitude=ext_p_ref.grad_amp, flat_time=rf_refoc_dur, system=system)
        rf_refoc = make_arbitrary_rf(signal=ext_p_ref.rf, flip_angle=2*flip_angle*np.pi/180, delay=gz_refoc.rise_time, system=system)
        tbp_refoc = gz_refoc.amplitude * slice_res*refoc_fac * rf_refoc_dur # calculate tbp of external pulse for multiband modulation
    else:
        rf_refoc, gz_refoc, _  = make_sinc_pulse(flip_angle=2*flip_angle*np.pi/180, system=system, duration=rf_refoc_dur, slice_thickness=slice_res*refoc_fac,
                                apodization=0.5, time_bw_product=tbp_refoc, use='refocusing', return_gz=True)
    gz_refoc_area = gz_refoc.area

    if sms and not verse_rf:
        band_sep_refoc  = slice_sep/slice_res*tbp_refoc
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

# timing
if ptx:
    exc_to_rew = calc_duration(rf, rf_del) - rf.delay - ph.round_up_to_raster(rf_dur/2, decimals=5)
    if gz_rew is not None:
        rew_dur = calc_duration(gz_rew)
    else:
        rew_dur = 0
else:
    exc_to_rew = calc_duration(rf, gz, rf_del) - rf.delay - ph.round_up_to_raster(rf_dur/2, decimals=5) # time from middle of rf pulse to rewinder
    rew_dur = calc_duration(gz_rew)

# RF spoiling parameters
rf_spoiling_inc = 50 # increment of RF spoiling [°]
rf_phase        = 0 
rf_inc          = 0

# Fat saturation
fw_shift = 3.35e-6 # unsigned fat water shift [ppm]
if fatsat:
    fatsat_fa = 90 # flip angle [°]
    offset = -1 * int(B0*system.gamma*fw_shift)
    fatsat_bw = abs(offset) # bandwidth [Hz]
    fatsat_dur = ph.round_up_to_raster(fatsat_tbp/fatsat_bw, decimals=5)
    rf_fatsat, fatsat_del = make_gauss_pulse(flip_angle=fatsat_fa*np.pi/180, duration=fatsat_dur, bandwidth=fatsat_bw, freq_offset=offset, system=system, return_delay = True)

#%% Diffusion gradients and delay calculation 

# Delays in diffusion sequence

#  RF  rew refoc_delay  diffgrad1   RF_refoc   diffgrad2   te_delay   readout
# #### ### ###########  #########   ########   #########   ########   #######
#   -------------- TE/2 ----------------
#                       --------    spacing   --------
#   -------------------------------------------- TE ------------------

if seq_type=='diffusion':

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
                if enc == 1:
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
        b0_dir = directions[enc][0]
        n_skip = len(diff_list[enc]) // (b0_dir + 1)
        for i in range(b0_dir):
            index = (i + 1) * n_skip + i
            diff_list[enc].insert(index, {"bval": 0, "dir": np.zeros([3]), "bDelta": bDelta[enc]})

    diff_list = [volume for encoding in diff_list for volume in encoding]

    # use old protocol (b-parameters)
    if use_prot:
        prot = ismrmrd.Dataset(prot_file, create_if_needed=False)
        prot_arrays = get_ismrmrd_arrays(prot_file)

        for i,elem in enumerate(diff_list):
            diff_list[i]['bval'] = int(prot_arrays['b_values'][i])
            diff_list[i]['dir'] = prot_arrays['Directions'][i].astype('float64')
            diff_list[i]['bDelta'] = int(prot_arrays['bDeltas'][i])

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

    if NOW == 0: # qMas with LTE tuned

        # calculate diffusion parameters from Matlab-optimization
        matlab_executable = 'matlab'
        matlab_script_name = 'qMasOptimization'
        arguments = f"{b_val_max}, {refoc_dur}"
        command = f"{matlab_executable} -nodisplay -r \"{matlab_script_name}({arguments}); exit\""
        completed_process = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
        count = completed_process.stdout.decode('utf-8').find('min_spacing')
        input_string = completed_process.stdout.decode('utf-8')[count:-1]
        numbers = re.findall(r'\d+\.\d+|\d+', input_string)
        numbers = [float(num) for num in numbers]

        # set diffusion parameters
        system.max_grad = numbers[1]
        system.max_slew = numbers[2]
        params_diff = diff_params(b_val=b_val_max, delta=numbers[3], spacing=numbers[0])

        # make spherical gradients
        diffgrad_ramp = ph.round_up_to_raster(system.max_grad/system.max_slew, 5)
        diffgrad_flat = params_diff.delta - diffgrad_ramp
        diffgrad_dur = diffgrad_flat + 2*diffgrad_ramp
        trap_diff_z = make_trapezoid(channel='z', system=system, flat_time=diffgrad_flat, rise_time=diffgrad_ramp, amplitude=system.max_grad, max_slew=system.max_slew)

        x = ph.round_up_to_raster((1e-3 * 70 * system.gamma)/(180 * system.gamma), 5)
        tp = params_diff.spacing - diffgrad_dur - refoc_dur - 4*x
        tv = (params_diff.spacing+diffgrad_dur-refoc_dur)/2
        tn = (params_diff.spacing+diffgrad_dur+refoc_dur)/2
        counter = params_diff.delta**2*(params_diff.spacing-params_diff.delta/3)+diffgrad_ramp**3/30-params_diff.delta*diffgrad_ramp**2/6
        alpha_x = np.sqrt(counter/(params_diff.delta**2*(params_diff.spacing-diffgrad_dur+5/3*refoc_dur-4/3*x)))
        alpha_y = np.sqrt(counter/(params_diff.delta**2*(tp+8*x+2*np.pi**2*x**2/tp+8*np.pi**2*x**3/(5*tp**2))))
        C = (45*params_diff.spacing*diffgrad_dur**4 - 180*params_diff.spacing*diffgrad_dur**3*params_diff.delta - 60*params_diff.spacing*diffgrad_dur**3*diffgrad_ramp + 270*params_diff.spacing*diffgrad_dur**2*params_diff.delta**2 + 120*params_diff.spacing*diffgrad_dur**2*params_diff.delta*diffgrad_ramp + 150*params_diff.spacing*diffgrad_dur**2*diffgrad_ramp**2 - 180*params_diff.spacing*diffgrad_dur*params_diff.delta**3 - 60*params_diff.spacing*diffgrad_dur*params_diff.delta**2*diffgrad_ramp - 180*params_diff.spacing*diffgrad_dur*params_diff.delta*diffgrad_ramp**2 - 60*params_diff.spacing*diffgrad_dur*diffgrad_ramp**3 + 45*params_diff.spacing*params_diff.delta**4 + 90*params_diff.spacing*params_diff.delta**2*diffgrad_ramp**2 + 45*params_diff.spacing*diffgrad_ramp**4 - 12*diffgrad_dur**5 + 60*diffgrad_dur**4*params_diff.delta - 60*diffgrad_dur**4*diffgrad_ramp - 120*diffgrad_dur**3*params_diff.delta**2 + 120*diffgrad_dur**3*params_diff.delta*diffgrad_ramp + 120*diffgrad_dur**3*diffgrad_ramp*tn - 120*diffgrad_dur**3*diffgrad_ramp*tv + 120*diffgrad_dur**2*params_diff.delta**3 - 30*diffgrad_dur**2*params_diff.delta**2*diffgrad_ramp + 90*diffgrad_dur**2*params_diff.delta*diffgrad_ramp**2 - 240*diffgrad_dur**2*params_diff.delta*diffgrad_ramp*tn + 240*diffgrad_dur**2*params_diff.delta*diffgrad_ramp*tv - 40*diffgrad_dur**2*diffgrad_ramp**2*tn + 40*diffgrad_dur**2*diffgrad_ramp**2*tv + 160*diffgrad_dur**2*diffgrad_ramp**2*x - 60*diffgrad_dur*params_diff.delta**4 - 60*diffgrad_dur*params_diff.delta**3*diffgrad_ramp + 120*diffgrad_dur*params_diff.delta**2*diffgrad_ramp*tn - 120*diffgrad_dur*params_diff.delta**2*diffgrad_ramp*tv - 60*diffgrad_dur*params_diff.delta*diffgrad_ramp**3 - 60*diffgrad_dur*diffgrad_ramp**4 + 120*diffgrad_dur*diffgrad_ramp**3*tn - 120*diffgrad_dur*diffgrad_ramp**3*tv + 12*params_diff.delta**5 + 30*params_diff.delta**4*diffgrad_ramp - 30*params_diff.delta**3*diffgrad_ramp**2 + 30*params_diff.delta**2*diffgrad_ramp**3 + 18*diffgrad_ramp**5)/(20*diffgrad_dur**2*diffgrad_ramp**2)
        alpha_z = np.sqrt(counter/(params_diff.delta**2*(C/3)))
        Gx = np.sqrt(2)/3 * system.max_grad * params_diff.delta * 2*np.pi/tp * alpha_x
        Gy = np.sqrt(2/3) * system.max_grad * params_diff.delta * 2*np.pi/tp * alpha_y
        Gz = 4/3 * system.max_grad * params_diff.delta * np.pi/tp * alpha_z

        trap_diff_z_ampl = trap_diff_z.amplitude *  alpha_z
        if trap_diff_z_ampl/(1e-3 * 70 * system.gamma) > 1:
            raise ValueError('trap_diff_z_ampl is too high')
        if (trap_diff_z_ampl/trap_diff_z.rise_time)/(200 * system.gamma) > 1:
            raise ValueError('trap_diff_z_slew is too high')
        
        time = np.arange(0,tp/2+system.grad_raster_time,system.grad_raster_time)
        
        time_ramp = np.arange(0,x+system.grad_raster_time,system.grad_raster_time)
        Gy_ramp_wf = Gy/x * time_ramp
        Gy_ramp_wf = points_to_waveform(amplitudes=Gy_ramp_wf, grad_raster_time=system.grad_raster_time, times=time_ramp)
        Gy_sinus_wf = Gy*np.cos(2*np.pi/tp*time)
        Gy_sinus_wf = points_to_waveform(amplitudes=Gy_sinus_wf, grad_raster_time=system.grad_raster_time, times=time)
        GySinus_wf_first = GySinus_wf_second = np.concatenate((Gy_ramp_wf,Gy_sinus_wf,(-1)*Gy_ramp_wf[::-1]))
        GySinus_first = GySinus_second = make_arbitrary_grad(channel='y', waveform=GySinus_wf_first, system=system)

        GxSinus_wf = -Gx*np.sin(2*np.pi/tp*time)
        GxSinus_wf_first = GxSinus_wf_second = points_to_waveform(amplitudes=GxSinus_wf, grad_raster_time=system.grad_raster_time, times=time)
        GxSinus_first = GxSinus_second = make_arbitrary_grad(channel='x', waveform=(-1)*GxSinus_wf_first, delay = x, system=system)
        
        GzSinus_wf = -Gz*np.sin(2*np.pi/tp*time)
        GzSinus_wf_first = GzSinus_wf_second = points_to_waveform(amplitudes=GzSinus_wf, grad_raster_time=system.grad_raster_time, times=time)
        GzSinus_first = GzSinus_second = make_arbitrary_grad(channel='z', waveform=GzSinus_wf_first, delay = x, system=system)

        sinusTime_first = make_delay(d=calc_duration(GxSinus_first,GySinus_first,GzSinus_first))
        sinusTime_second = make_delay(d=calc_duration(GxSinus_second,GySinus_second,GzSinus_second))

        # make linear gradients out of spherical gradients
        GxSinus_lin_x = make_arbitrary_grad(channel='x', waveform=(-1)*GxSinus_wf_first*np.sqrt(3), delay = x, system=system)
        GxSinus_lin_y = make_arbitrary_grad(channel='y', waveform=GxSinus_wf_first*np.sqrt(3), delay = x, system=system)
        GxSinus_lin_z = make_arbitrary_grad(channel='z', waveform=GxSinus_wf_first*np.sqrt(3), delay = x, system=system)
        GxSinus_lin_first = GxSinus_lin_second = [GxSinus_lin_x,GxSinus_lin_y,GxSinus_lin_z]

        # make planar gradients out of spherical gradients
        wf_GxSinus_sph_first = wf_GxSinus_sph_second = np.concatenate((np.zeros(int((x+system.grad_raster_time)/system.grad_raster_time)),GxSinus_wf_first,np.zeros(int((x)/system.grad_raster_time))))
        wf_GySinus_sph_first = wf_GySinus_sph_second = GySinus_wf_first
        wf_GzSinus_sph = np.concatenate((np.zeros(int((x+system.grad_raster_time)/system.grad_raster_time)),GzSinus_wf_first,np.zeros(int((x)/system.grad_raster_time))))

        # just for testing all gradients that will be created below
        for vol_ix, diff in enumerate(diff_list):
            if diff['bDelta'] == -0.5:
                M = vec2vec_rotmat(np.array([1,0,0]), diff['dir'])
                GxSinus_plan_x = make_arbitrary_grad(channel='x', waveform=(-1)*np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_first*np.sqrt(3/2)*M[0,1] + wf_GySinus_sph_first*np.sqrt(3/2)*M[0,2]), system=system) # consider sign change from Pulseq rotation matrix
                GxSinus_plan_y = make_arbitrary_grad(channel='y', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_first*np.sqrt(3/2)*M[1,1] + wf_GySinus_sph_first*np.sqrt(3/2)*M[1,2]), system=system)
                GxSinus_plan_z = make_arbitrary_grad(channel='z', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_first*np.sqrt(3/2)*M[2,1] + wf_GySinus_sph_first*np.sqrt(3/2)*M[2,2]), system=system)
                
                GxSinus_plan_x = make_arbitrary_grad(channel='x', waveform=(-1)*np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_second*np.sqrt(3/2)*M[0,1] + wf_GySinus_sph_second*np.sqrt(3/2)*M[0,2]), system=system) # consider sign change from Pulseq rotation matrix
                GxSinus_plan_y = make_arbitrary_grad(channel='y', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_second*np.sqrt(3/2)*M[1,1] + wf_GySinus_sph_second*np.sqrt(3/2)*M[1,2]), system=system)
                GxSinus_plan_z = make_arbitrary_grad(channel='z', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_second*np.sqrt(3/2)*M[2,1] + wf_GySinus_sph_second*np.sqrt(3/2)*M[2,2]), system=system)
        
        GxSinus_plan_first = GxSinus_plan_second = [GxSinus_plan_x,GxSinus_plan_y,GxSinus_plan_z]

    elif NOW == 1: # NOW with LTE tuned
        
        # load NOW Results
        NowResults = loadmat('NOW/NowResults_LteTuned.mat')
        NowResults = NowResults['NowResults']

        time_first = NowResults['timeFirst'][0][0][0] * 1e-3 # s
        time_second = NowResults['timeSecond'][0][0][0] * 1e-3 # s
        time_third = NowResults['timeZero'][0][0][0] * 1e-3 # s

        # make spherical gradients
        GxSinus_wf_first = NowResults['gwfFirst'][0][0][:,0] * system.gamma # [Hz/m]
        GxSinus_wf_first = points_to_waveform(amplitudes=GxSinus_wf_first, grad_raster_time=system.grad_raster_time, times=time_first)
        GxSinus_first = make_arbitrary_grad(channel='x', waveform=(-1)*GxSinus_wf_first, system=system)
        GxSinus_wf_second = NowResults['gwfSecond'][0][0][:,0] * system.gamma # [Hz/m]
        GxSinus_wf_second = points_to_waveform(amplitudes=GxSinus_wf_second, grad_raster_time=system.grad_raster_time, times=time_second)
        GxSinus_second = make_arbitrary_grad(channel='x', waveform=(-1)*GxSinus_wf_second, system=system)

        GySinus_wf_first = NowResults['gwfFirst'][0][0][:,1] * system.gamma # [Hz/m]
        GySinus_wf_first = points_to_waveform(amplitudes=GySinus_wf_first, grad_raster_time=system.grad_raster_time, times=time_first)
        GySinus_first = make_arbitrary_grad(channel='y', waveform=GySinus_wf_first, system=system)
        GySinus_wf_second = NowResults['gwfSecond'][0][0][:,1] * system.gamma # [Hz/m]
        GySinus_wf_second = points_to_waveform(amplitudes=GySinus_wf_second, grad_raster_time=system.grad_raster_time, times=time_second)
        GySinus_second = make_arbitrary_grad(channel='y', waveform=GySinus_wf_second, system=system)

        GzSinus_wf_first = NowResults['gwfFirst'][0][0][:,2] * system.gamma # [Hz/m]
        GzSinus_wf_first = points_to_waveform(amplitudes=GzSinus_wf_first, grad_raster_time=system.grad_raster_time, times=time_first)
        GzSinus_first = make_arbitrary_grad(channel='z', waveform=GzSinus_wf_first, system=system)
        GzSinus_wf_second = NowResults['gwfSecond'][0][0][:,2] * system.gamma # [Hz/m]
        GzSinus_wf_second = points_to_waveform(amplitudes=GzSinus_wf_second, grad_raster_time=system.grad_raster_time, times=time_second)
        GzSinus_second = make_arbitrary_grad(channel='z', waveform=GzSinus_wf_second, system=system)

        # calculate timing of the diffusion encoding parts
        rf_longer = (time_third[-1] - refoc_dur)/2
        sinusTime_first = make_delay(d=calc_duration(GxSinus_first,GySinus_first,GzSinus_first)+rf_longer)
        GxSinus_second.delay = GySinus_second.delay = GzSinus_second.delay = ph.round_up_to_raster(rf_longer,decimals=5)
        sinusTime_second = make_delay(d=ph.round_up_to_raster(calc_duration(GxSinus_second,GySinus_second,GzSinus_second),decimals=5))
        diffgrad_dur = 0

        # make linear gradients out of spherical gradients
        GxSinus_lin_x_first = make_arbitrary_grad(channel='x', waveform=(-1)*GxSinus_wf_first*np.sqrt(3), system=system)
        GxSinus_lin_y_first = make_arbitrary_grad(channel='y', waveform=GxSinus_wf_first*np.sqrt(3), system=system)
        GxSinus_lin_z_first = make_arbitrary_grad(channel='z', waveform=GxSinus_wf_first*np.sqrt(3), system=system)
        GxSinus_lin_first = [GxSinus_lin_x_first,GxSinus_lin_y_first,GxSinus_lin_z_first]
        GxSinus_lin_x_second = make_arbitrary_grad(channel='x', waveform=(-1)*GxSinus_wf_second*np.sqrt(3), system=system)
        GxSinus_lin_y_second = make_arbitrary_grad(channel='y', waveform=GxSinus_wf_second*np.sqrt(3), system=system)
        GxSinus_lin_z_second = make_arbitrary_grad(channel='z', waveform=GxSinus_wf_second*np.sqrt(3), system=system)
        GxSinus_lin_second = [GxSinus_lin_x_second,GxSinus_lin_y_second,GxSinus_lin_z_second]

        # make planar gradients out of spherical gradients
        wf_GxSinus_sph_first = GxSinus_wf_first
        wf_GxSinus_sph_second = GxSinus_wf_second
        wf_GySinus_sph_first = GySinus_wf_first
        wf_GySinus_sph_second = GySinus_wf_second

        # just for testing all gradients that will be created below
        for vol_ix, diff in enumerate(diff_list):
            if diff['bDelta'] == -0.5:
                M = vec2vec_rotmat(np.array([1,0,0]), diff['dir'])
                GxSinus_plan_x_first = make_arbitrary_grad(channel='x', waveform=(-1)*np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_first*np.sqrt(3/2)*M[0,1] + wf_GySinus_sph_first*np.sqrt(3/2)*M[0,2]), system=system) # consider sign change from Pulseq rotation matrix
                GxSinus_plan_y_first = make_arbitrary_grad(channel='y', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_first*np.sqrt(3/2)*M[1,1] + wf_GySinus_sph_first*np.sqrt(3/2)*M[1,2]), system=system)
                GxSinus_plan_z_first = make_arbitrary_grad(channel='z', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_first*np.sqrt(3/2)*M[2,1] + wf_GySinus_sph_first*np.sqrt(3/2)*M[2,2]), system=system)
                
                GxSinus_plan_x_second = make_arbitrary_grad(channel='x', waveform=(-1)*np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_second*np.sqrt(3/2)*M[0,1] + wf_GySinus_sph_second*np.sqrt(3/2)*M[0,2]), system=system) # consider sign change from Pulseq rotation matrix
                GxSinus_plan_y_second = make_arbitrary_grad(channel='y', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_second*np.sqrt(3/2)*M[1,1] + wf_GySinus_sph_second*np.sqrt(3/2)*M[1,2]), system=system)
                GxSinus_plan_z_second = make_arbitrary_grad(channel='z', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_second*np.sqrt(3/2)*M[2,1] + wf_GySinus_sph_second*np.sqrt(3/2)*M[2,2]), system=system)
        
        GxSinus_plan_first = [GxSinus_plan_x_first,GxSinus_plan_y_first,GxSinus_plan_z_first]
        GxSinus_plan_second = [GxSinus_plan_x_second,GxSinus_plan_y_second,GxSinus_plan_z_second]
            
    elif NOW == 2: # NOW with LTE detuned
        
        # load NOW Results
        NowResults = loadmat('NOW/NowResults_LteComp.mat') # LteDetuned ; LteComp
        NowResults = NowResults['NowResults']

        time_first = NowResults['timeFirst'][0][0][0] * 1e-3 # s
        time_second = NowResults['timeSecond'][0][0][0] * 1e-3 # s
        time_third = NowResults['timeZero'][0][0][0] * 1e-3 # s

        # make spherical gradients
        GxSinus_wf_first = NowResults['gwfFirst'][0][0][:,0] * system.gamma # [Hz/m]
        GxSinus_wf_first = points_to_waveform(amplitudes=GxSinus_wf_first, grad_raster_time=system.grad_raster_time, times=time_first)
        GxSinus_first = make_arbitrary_grad(channel='x', waveform=(-1)*GxSinus_wf_first, system=system)
        GxSinus_wf_second = NowResults['gwfSecond'][0][0][:,0] * system.gamma # [Hz/m]
        GxSinus_wf_second = points_to_waveform(amplitudes=GxSinus_wf_second, grad_raster_time=system.grad_raster_time, times=time_second)
        GxSinus_second = make_arbitrary_grad(channel='x', waveform=(-1)*GxSinus_wf_second, system=system)

        GySinus_wf_first = NowResults['gwfFirst'][0][0][:,1] * system.gamma # [Hz/m]
        GySinus_wf_first = points_to_waveform(amplitudes=GySinus_wf_first, grad_raster_time=system.grad_raster_time, times=time_first)
        GySinus_first = make_arbitrary_grad(channel='y', waveform=GySinus_wf_first, system=system)
        GySinus_wf_second = NowResults['gwfSecond'][0][0][:,1] * system.gamma # [Hz/m]
        GySinus_wf_second = points_to_waveform(amplitudes=GySinus_wf_second, grad_raster_time=system.grad_raster_time, times=time_second)
        GySinus_second = make_arbitrary_grad(channel='y', waveform=GySinus_wf_second, system=system)

        GzSinus_wf_first = NowResults['gwfFirst'][0][0][:,2] * system.gamma # [Hz/m]
        GzSinus_wf_first = points_to_waveform(amplitudes=GzSinus_wf_first, grad_raster_time=system.grad_raster_time, times=time_first)
        GzSinus_first = make_arbitrary_grad(channel='z', waveform=GzSinus_wf_first, system=system)
        GzSinus_wf_second = NowResults['gwfSecond'][0][0][:,2] * system.gamma # [Hz/m]
        GzSinus_wf_second = points_to_waveform(amplitudes=GzSinus_wf_second, grad_raster_time=system.grad_raster_time, times=time_second)
        GzSinus_second = make_arbitrary_grad(channel='z', waveform=GzSinus_wf_second, system=system)

        # calculate timing of the diffusion encoding parts
        rf_longer = (time_third[-1] - refoc_dur)/2
        sinusTime_first = make_delay(d=ph.round_up_to_raster(calc_duration(GxSinus_first,GySinus_first,GzSinus_first)+rf_longer,decimals=5))
        GxSinus_second.delay = GySinus_second.delay = GzSinus_second.delay = ph.round_up_to_raster(rf_longer,decimals=5)
        sinusTime_second = make_delay(d=ph.round_up_to_raster(calc_duration(GxSinus_second,GySinus_second,GzSinus_second),decimals=5))
        diffgrad_dur = 0

        # make linear gradients out of spherical gradients
        # calculate diffusion parameters
        encoding_duration = sinusTime_first.delay + sinusTime_second.delay + refoc_dur
        params_diff = diff_params(b_val=b_val_max, delta=None, spacing=None)
        params_diff.calc_params(system, encoding_duration=encoding_duration)
        
        diffgrad_ramp = ph.round_up_to_raster(system.max_grad/system.max_slew, 5)
        diffgrad_flat = params_diff.delta - diffgrad_ramp
        diffgrad_dur_NOW2 = diffgrad_flat + 2*diffgrad_ramp

        if params_diff.spacing < refoc_dur+diffgrad_dur_NOW2:
            raise ValueError('Spacing between diffusion gradients too low. Increase spacing or decrease duration (delta) of diffusion gradients.')

        spacing_delay = make_delay(d = params_diff.spacing - sinusTime_first.delay - refoc_dur)

        # make diffusion gradients - x-channel will get sign change from Pulseq rotation matrix (in the old/compat Pulseq mode), if gradient is trapezoidal
        trap_diff_x = make_trapezoid(channel='x', system=system, flat_time=diffgrad_flat, rise_time=diffgrad_ramp, amplitude=-1*system.max_grad, max_slew=system.max_slew)
        trap_diff_y = make_trapezoid(channel='y', system=system, flat_time=diffgrad_flat, rise_time=diffgrad_ramp, amplitude=system.max_grad, max_slew=system.max_slew)
        trap_diff_z = make_trapezoid(channel='z', system=system, flat_time=diffgrad_flat, rise_time=diffgrad_ramp, amplitude=system.max_grad, max_slew=system.max_slew)
        diff_gradients = [trap_diff_x,trap_diff_y,trap_diff_z]

        # make planar gradients out of spherical gradients
        wf_GxSinus_sph_first = GxSinus_wf_first
        wf_GxSinus_sph_second = GxSinus_wf_second
        wf_GySinus_sph_first = GySinus_wf_first
        wf_GySinus_sph_second = GySinus_wf_second

        # just for testing all gradients that will be created below
        for vol_ix, diff in enumerate(diff_list):
            if diff['bDelta'] == -0.5:
                M = vec2vec_rotmat(np.array([1,0,0]), diff['dir'])
                GxSinus_plan_x_first = make_arbitrary_grad(channel='x', waveform=(-1)*np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_first*np.sqrt(3/2)*M[0,1] + wf_GySinus_sph_first*np.sqrt(3/2)*M[0,2]), system=system) # consider sign change from Pulseq rotation matrix
                GxSinus_plan_y_first = make_arbitrary_grad(channel='y', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_first*np.sqrt(3/2)*M[1,1] + wf_GySinus_sph_first*np.sqrt(3/2)*M[1,2]), system=system)
                GxSinus_plan_z_first = make_arbitrary_grad(channel='z', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_first*np.sqrt(3/2)*M[2,1] + wf_GySinus_sph_first*np.sqrt(3/2)*M[2,2]), system=system)
                
                GxSinus_plan_x_second = make_arbitrary_grad(channel='x', waveform=(-1)*np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_second*np.sqrt(3/2)*M[0,1] + wf_GySinus_sph_second*np.sqrt(3/2)*M[0,2]), system=system) # consider sign change from Pulseq rotation matrix
                GxSinus_plan_y_second = make_arbitrary_grad(channel='y', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_second*np.sqrt(3/2)*M[1,1] + wf_GySinus_sph_second*np.sqrt(3/2)*M[1,2]), system=system)
                GxSinus_plan_z_second = make_arbitrary_grad(channel='z', waveform=np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_second*np.sqrt(3/2)*M[2,1] + wf_GySinus_sph_second*np.sqrt(3/2)*M[2,2]), system=system)
        
        GxSinus_plan_first = [GxSinus_plan_x_first,GxSinus_plan_y_first,GxSinus_plan_z_first]
        GxSinus_plan_second = [GxSinus_plan_x_second,GxSinus_plan_y_second,GxSinus_plan_z_second]

    elif NOW == 3: # get spacings for NOW optimization
        firstEncDur = ph.round_up_to_raster(TE/2 - exc_to_rew - rew_dur - ph.round_up_to_raster(refoc_dur/2, decimals=5), decimals=5)
        secondEncDur = ph.round_up_to_raster(TE/2 - ph.round_up_to_raster(refoc_dur/2, decimals=5), decimals=5)
        print(f'firstEncDur: {firstEncDur} [s]')
        print(f'secondEncDur: {secondEncDur} [s]')
        sys.exit()

    # reset the maximum slewrate and gradient
    system.max_slew = max_slew * system.gamma
    system.max_grad = 1e-3 * max_grad * system.gamma
    
    # calculate minimum TE
    min_te_first = exc_to_rew + rew_dur + diffgrad_dur + sinusTime_first.delay + ph.round_up_to_raster(refoc_dur/2, decimals=5)
    min_te_second = sinusTime_second.delay + diffgrad_dur + ph.round_up_to_raster(refoc_dur/2, decimals=5)
    min_te = 2*max(min_te_first,min_te_second)
    if skope:
        min_te += skope_delay
    if TE < min_te:
        raise ValueError(f'Minimum TE: {min_te*1e3}')

    # make delays
    refoc_delay = make_delay(d = TE/2 - exc_to_rew - rew_dur - diffgrad_dur - sinusTime_first.delay - ph.round_up_to_raster(refoc_dur/2, decimals=5))
    te_delay = make_delay(d = TE/2 - ph.round_up_to_raster(refoc_dur/2, decimals=5) - sinusTime_second.delay - diffgrad_dur)

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
    freq_max = ph.check_resonances([GxSinus_first.waveform,GySinus_first.waveform,GzSinus_first.waveform,
                                    GxSinus_second.waveform,GySinus_second.waveform,GzSinus_second.waveform], resonances) 

if seq_type=='gre':
    # calculate minimum TE
    min_te = exc_to_rew + rew_dur
    if min_te+skope_delay > TE:
        raise ValueError('Minimum TE is {} ms.'.format((min_te+skope_delay)*1e3))
    te_delay = make_delay(d = TE - min_te)

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
    if fatsat:
        amp_x, ftop_x, ramp_x = ph.trap_from_area(-area_x, system, slewrate=100, max_grad=30e-3) # reduce slew rate & max_grad to to avoid stimulation
        amp_y, ftop_y, ramp_y = ph.trap_from_area(-area_y, system, slewrate=100, max_grad=30e-3)
    else:
        amp_x, ftop_x, ramp_x = ph.trap_from_area(-area_x+spoiler_area, system, slewrate=100, max_grad=30e-3) # reduce slew rate & max_grad to to avoid stimulation
        amp_y, ftop_y, ramp_y = ph.trap_from_area(-area_y+spoiler_area, system, slewrate=100, max_grad=30e-3)
    spirals[k]['reph'][0] = make_trapezoid(channel='x', system=system, amplitude=amp_x, flat_time=ftop_x, rise_time=ramp_x)
    spirals[k]['reph'][1] = make_trapezoid(channel='y', system=system, amplitude=amp_y, flat_time=ftop_y, rise_time=ramp_y)
    reph_dur.append(max(ftop_x+2*ramp_x, ftop_y+2*ramp_y))

# spoiler (after fatsat or before next excitation)
spoiler_x  = make_trapezoid(channel='x',system=system, amplitude=amp_spoil, flat_time=ftop_spoil, rise_time=ramp_spoil)
spoiler_y  = make_trapezoid(channel='y',system=system, amplitude=amp_spoil, flat_time=ftop_spoil, rise_time=ramp_spoil)
spoiler_z  = make_trapezoid(channel='z',system=system, amplitude=amp_spoil, flat_time=ftop_spoil, rise_time=ramp_spoil)
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
    if seq_type == 'diffusion':
        t_skope += te_delay.delay * 1e3
    print('Minimum Skope acquisition time: {:.2f} ms'.format(t_skope))

#%% Set up protocol for FIRE reco and write header

date = datetime.date.today().strftime('%Y%m%d')
if meas_date is not None:
    date = meas_date
filename = date + '_' + seq_name

if seq_dest == 'scanner':

    # set some parameters for the protocol
    if seq_type == "gre":
        t_min = TE + dwelltime/2
        n_dirs = 1
    if seq_type == "diffusion":
        t_min = dwelltime/2

    # create new directory if needed
    Path("../Protocols/"+date).mkdir(parents=True, exist_ok=True)
    ismrmrd_file = f"../Protocols/{date}/{filename}.h5"

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
    
else:
    prot = None

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
tokx = make_trapezoid(channel='x', amplitude=-1e-3*system.gamma, rise_time=1e-3, flat_time=4e-3)
toky = make_trapezoid(channel='y', amplitude=1e-3*system.gamma, rise_time=1e-3, flat_time=4e-3)
tokz = make_trapezoid(channel='z', amplitude=1e-3*system.gamma, rise_time=1e-3, flat_time=4e-3)
seq.add_block(tokx,toky,tokz,make_delay(d=0.5))
seq.add_block(tokx,toky,tokz,make_delay(d=0.5))
seq.add_block(tokx,toky,tokz,make_delay(d=0.5))

# Noise scans
noise_samples = 256
noise_adc = make_adc(system=system, num_samples=256, dwell=dwelltime, delay=10e-6) # delay to be safe with pTx system (had crashes due to short NCO gaps)
noise_delay = make_delay(d=ph.round_up_to_raster(calc_duration(noise_adc)+1e-3,decimals=5)) # add some more time to the ADC delay to be safe
for k in range(noisescans):
    seq.add_block(noise_adc, noise_delay)
    if seq_dest == 'scanner':
        acq = ismrmrd.Acquisition()
        acq.setFlag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)
        prot.append_acquisition(acq)

# Perform cartesian reference scan
if refscan:
    if B0 > 4: # 7T
        te_1 = 2.04e-3 # 2nd "in-phase" timestamp: 2*(981Hz)^-1
        te_2 = 3.06e-3
        te_3 = 4.08e-3
        rf_ref_dur = 0.8e-3
        tbp_ref = 2
    else: # 3T
        te_1 = 2.42e-3
        te_2 = 4.84e-3
        te_3 = 4.84e-3
        rf_ref_dur = 1.2e-3
        tbp_ref = 4
    if half_refscan:
        slices_ref = slices // 2
        slice_res_ref = slice_res * 2
    else:
        slices_ref = slices
        slice_res_ref = slice_res

    params_ref = {"fov": fov*1e-3, "slices":slices_ref, "slice_res":slice_res_ref, "dist_fac": dist_fac, "res":res_refscan,
                  "flip_angle":flip_refscan, "readout_bw": bw_refscan, "rf_dur": rf_ref_dur, "tbp": tbp_ref, "separate_tr": separate_tr}
    if refscan == 1:
        params_ref["ref_lines"] = 30 # ecalib takes 24 lines as default
        params_ref["TE"] = [te_1]
    elif refscan == 2:
        params_ref["center_out"] = True # center out readout
        if separate_tr:
            params_ref["TE"] = [te_1, te_2]
        else:
            params_ref["TE"] = [te_1, te_3]
    elif refscan == 3: # 3D refscan
        params_ref["rf_dur"] = 1.8e-3
        params_ref["dim3"] = True
        params_ref["tbp"] = 32
        params_ref["slice_os"] = 0.2 # slice oversampling (0.1=10%)
        if separate_tr:
            params_ref["TE"] = [te_1, te_2]
        else:
            params_ref["TE"] = [te_1, te_3]
    else:
        raise ValueError("Invalid refscan selection.")

    # make refscan
    gre_refscan_B0(seq, prot=prot, system=system, params=params_ref, grads_off=grads_off)

dur_until_ref = seq.duration()[0]
print(f"Sequence duration after reference scan: {dur_until_ref:.2f} s")

# Skope sync scans
if skope:
    if seq_dest == 'scanner':
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

        if seq_dest == 'scanner':
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

if seq_type=='diffusion':

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
            diff_list = [{"bval": 0, "dir": np.zeros(3), "bDelta": 0}]
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
                        crusher_x.waveform = crusher_wf
                        crusher_y.waveform = crusher_wf
                        grad_refoc = grad_refoc1
                        
                        if NOW == 0:
                            trap_diff_z.amplitude = 0

                        GxSinus_first.waveform = np.zeros_like(GxSinus_first.waveform)
                        GySinus_first.waveform = np.zeros_like(GySinus_first.waveform)
                        GzSinus_first.waveform = np.zeros_like(GzSinus_first.waveform)
                        GxSinus_second.waveform = np.zeros_like(GxSinus_second.waveform)
                        GySinus_second.waveform = np.zeros_like(GySinus_second.waveform)
                        GzSinus_second.waveform = np.zeros_like(GzSinus_second.waveform)

                        GxSinus_plan_first[0].waveform = np.zeros_like(GxSinus_first.waveform)
                        GxSinus_plan_first[1].waveform = np.zeros_like(GxSinus_first.waveform)
                        GxSinus_plan_first[2].waveform = np.zeros_like(GxSinus_first.waveform)
                        GxSinus_plan_second[0].waveform = np.zeros_like(GxSinus_second.waveform)
                        GxSinus_plan_second[1].waveform = np.zeros_like(GxSinus_second.waveform)
                        GxSinus_plan_second[2].waveform = np.zeros_like(GxSinus_second.waveform)

                        if NOW == 2:
                            diff_gradients[0].amplitude = 0
                            diff_gradients[1].amplitude = 0
                            diff_gradients[2].amplitude = 0
                        else:
                            GxSinus_lin_first[0].waveform = np.zeros_like(GxSinus_first.waveform)
                            GxSinus_lin_first[1].waveform = np.zeros_like(GxSinus_first.waveform)
                            GxSinus_lin_first[2].waveform = np.zeros_like(GxSinus_first.waveform)
                            GxSinus_lin_second[0].waveform = np.zeros_like(GxSinus_second.waveform)
                            GxSinus_lin_second[1].waveform = np.zeros_like(GxSinus_second.waveform)
                            GxSinus_lin_second[2].waveform = np.zeros_like(GxSinus_second.waveform)
                    else:
                        crusher_x.waveform *= 0
                        crusher_y.waveform *= 0
                        grad_refoc = grad_refoc2

                        if diff['bDelta'] == -0.5: # planar
                            if NOW == 0:
                                trap_diff_z.amplitude = 0
                            M = vec2vec_rotmat(np.array([1,0,0]), diff['dir'])
                            GxSinus_plan_first[0].waveform = (-1)*np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_first*np.sqrt(3/2)*M[0,1] + wf_GySinus_sph_first*np.sqrt(3/2)*M[0,2]) # consider sign change from Pulseq rotation matrix
                            GxSinus_plan_first[1].waveform = np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_first*np.sqrt(3/2)*M[1,1] + wf_GySinus_sph_first*np.sqrt(3/2)*M[1,2])
                            GxSinus_plan_first[2].waveform = np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_first*np.sqrt(3/2)*M[2,1] + wf_GySinus_sph_first*np.sqrt(3/2)*M[2,2])
                            GxSinus_plan_second[0].waveform = (-1)*np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_second*np.sqrt(3/2)*M[0,1] + wf_GySinus_sph_second*np.sqrt(3/2)*M[0,2]) # consider sign change from Pulseq rotation matrix
                            GxSinus_plan_second[1].waveform = np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_second*np.sqrt(3/2)*M[1,1] + wf_GySinus_sph_second*np.sqrt(3/2)*M[1,2])
                            GxSinus_plan_second[2].waveform = np.sqrt(diff['bval']/b_val_max)*(wf_GxSinus_sph_second*np.sqrt(3/2)*M[2,1] + wf_GySinus_sph_second*np.sqrt(3/2)*M[2,2])
                        elif diff['bDelta'] == 0: # spherical
                            if NOW == 0:
                                trap_diff_z.amplitude = np.sqrt(diff['bval']/b_val_max)*trap_diff_z_ampl
                            GxSinus_first.waveform = (-1)*np.sqrt(diff['bval']/b_val_max)*GxSinus_wf_first # consider sign change from Pulseq rotation matrix
                            GySinus_first.waveform = np.sqrt(diff['bval']/b_val_max)*GySinus_wf_first
                            GzSinus_first.waveform = np.sqrt(diff['bval']/b_val_max)*GzSinus_wf_first
                            GxSinus_second.waveform = (-1)*np.sqrt(diff['bval']/b_val_max)*GxSinus_wf_second # consider sign change from Pulseq rotation matrix
                            GySinus_second.waveform = np.sqrt(diff['bval']/b_val_max)*GySinus_wf_second
                            GzSinus_second.waveform = np.sqrt(diff['bval']/b_val_max)*GzSinus_wf_second
                        else: # linear
                            if NOW == 0:
                                trap_diff_z.amplitude = 0
                            if NOW == 2:
                                diff_gradients[0].amplitude = -1 * np.sqrt(diff['bval']/b_val_max) * diff['dir'][0] * diff_maxgrad_Hz # consider sign change from Pulseq rotation matrix
                                diff_gradients[1].amplitude = np.sqrt(diff['bval']/b_val_max) * diff['dir'][1] * diff_maxgrad_Hz
                                diff_gradients[2].amplitude = np.sqrt(diff['bval']/b_val_max) * diff['dir'][2] * diff_maxgrad_Hz
                            else:
                                GxSinus_lin_first[0].waveform = (-1)*np.sqrt(diff['bval']/b_val_max)*GxSinus_wf_first*np.sqrt(3) * diff['dir'][0] # consider sign change from Pulseq rotation matrix
                                GxSinus_lin_first[1].waveform = np.sqrt(diff['bval']/b_val_max)*GxSinus_wf_first*np.sqrt(3) * diff['dir'][1]
                                GxSinus_lin_first[2].waveform = np.sqrt(diff['bval']/b_val_max)*GxSinus_wf_first*np.sqrt(3) * diff['dir'][2]
                                GxSinus_lin_second[0].waveform = (-1)*np.sqrt(diff['bval']/b_val_max)*GxSinus_wf_second*np.sqrt(3) * diff['dir'][0] # consider sign change from Pulseq rotation matrix
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

                                seq_btens = Sequence() # initiate for b-tensor calculation

                                # Add (fatsat and) excitation pulse
                                if fatsat:
                                    rf_fatsat.phase_offset = rf_phase / 180 * np.pi # always use RF spoiling for fat sat pulse
                                    seq.add_block(rf_fatsat, fatsat_del)
                                    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
                                    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
                                    seq.add_block(spoiler_x, spoiler_y, spoiler_z)

                                rf.freq_offset = gz.amplitude * slice_res * (slc - (slices_eff - 1) / 2) * (1+dist_fac*1e-2)
                                rf_refoc.freq_offset = gz_refoc.amplitude * slice_res * (slc - (slices_eff - 1) / 2) * (1+dist_fac*1e-2)

                                seq.add_block(rf,gz,rf_del)
                                seq.add_block(gz_rew)
                                if btens_slice:
                                    seq_btens_exc = Sequence()
                                    seq_btens_exc.add_block(rf,gz,rf_del)
                                    half_slice_idx = seq_btens_exc.gradient_waveforms().shape[1]//2
                                    seq_btens_exc.add_block(gz_rew)
                                    seq_btens_exc.add_block(refoc_delay)

                                # diffusion block
                                seq.add_block(refoc_delay)
                                if NOW == 0:
                                    seq.add_block(trap_diff_z)
                                    seq_btens.add_block(trap_diff_z)
                                if diff['bDelta'] == -0.5: # planar
                                    seq.add_block(GxSinus_plan_first[0],GxSinus_plan_first[1],GxSinus_plan_first[2],sinusTime_first)
                                    seq_btens.add_block(GxSinus_plan_first[0],GxSinus_plan_first[1],GxSinus_plan_first[2],sinusTime_first)
                                elif diff['bDelta'] == 0: # spherical
                                    seq.add_block(GxSinus_first,GySinus_first,GzSinus_first,sinusTime_first)
                                    seq_btens.add_block(GxSinus_first,GySinus_first,GzSinus_first,sinusTime_first)
                                else: # linear
                                    if NOW == 2:
                                        seq.add_block(diff_gradients[0],diff_gradients[1],diff_gradients[2],sinusTime_first)
                                        seq_btens.add_block(diff_gradients[0],diff_gradients[1],diff_gradients[2],sinusTime_first)
                                    else:
                                        seq.add_block(GxSinus_lin_first[0],GxSinus_lin_first[1],GxSinus_lin_first[2],sinusTime_first)
                                        seq_btens.add_block(GxSinus_lin_first[0],GxSinus_lin_first[1],GxSinus_lin_first[2],sinusTime_first)
                                before_refoc_idx = seq_btens.gradient_waveforms().shape[1]
                                seq.add_block(rf_refoc,grad_refoc,crusher_x,crusher_y)
                                if btens_slice:
                                    seq_btens.add_block(rf_refoc,grad_refoc,crusher_x,crusher_y)
                                else:
                                    seq_btens.add_block(make_delay(d=refoc_dur))
                                after_refoc_idx = seq_btens.gradient_waveforms().shape[1]
                                if diff['bDelta'] == -0.5: # planar
                                    seq.add_block(GxSinus_plan_second[0],GxSinus_plan_second[1],GxSinus_plan_second[2],sinusTime_second)
                                    seq_btens.add_block(GxSinus_plan_second[0],GxSinus_plan_second[1],GxSinus_plan_second[2],sinusTime_second)
                                elif diff['bDelta'] == 0: # spherical
                                    seq.add_block(GxSinus_second,GySinus_second,GzSinus_second,sinusTime_second)
                                    seq_btens.add_block(GxSinus_second,GySinus_second,GzSinus_second,sinusTime_second)
                                else: # linear
                                    if NOW == 2:
                                        seq.add_block(spacing_delay)
                                        seq.add_block(diff_gradients[0],diff_gradients[1],diff_gradients[2])
                                        seq_btens.add_block(spacing_delay)
                                        seq_btens.add_block(diff_gradients[0],diff_gradients[1],diff_gradients[2])
                                    else:
                                        seq.add_block(GxSinus_lin_second[0],GxSinus_lin_second[1],GxSinus_lin_second[2],sinusTime_second)
                                        seq_btens.add_block(GxSinus_lin_second[0],GxSinus_lin_second[1],GxSinus_lin_second[2],sinusTime_second)
                                if NOW == 0:
                                    seq.add_block(trap_diff_z)
                                    seq_btens.add_block(trap_diff_z)

                                # calculate b-tensor for QTI analysis
                                if slc == 0 and prep == prepscans:
                                    g.append(seq_btens.gradient_waveforms()/system.gamma)
                                    g[-1][0,:] = (-1)*g[-1][0,:] # Pulseq rotation matrix inverts x-axis
                                    half_idx = before_refoc_idx + (after_refoc_idx-before_refoc_idx)//2
                                    g[-1][:,half_idx:] = (-1)*g[-1][:,half_idx:]

                                    if not ptx and btens_slice:
                                        g_exc = seq_btens_exc.gradient_waveforms()[:,half_slice_idx:]/system.gamma
                                        g[-1] = np.concatenate((g_exc,g[-1]),axis=1)

                                    gamma = system.gamma * 2 * np.pi
                                    btens.append(np.zeros((3,3)))
                                    for i in range(3):
                                        for j in range(3):
                                            btens[-1][i,j] = 1e-6*gamma**2*(np.sum((np.cumsum(g[-1][i,:])*np.cumsum(g[-1][j,:])*system.grad_raster_time**2)*system.grad_raster_time))

                                # Skope trigger - keep minimum distance of 200us between subsequent triggers
                                if skope and slc_ctr%trig_skip==0 and slices_eff-slc_ctr >= trig_skip and prep == prepscans:
                                    trig_ctr += 1
                                    seq.add_block(trig, te_delay)
                                else:
                                    seq.add_block(te_delay)

                                # spiral readout block
                                spiral_block = [spirals[n*redfac]['spiral'][0], spirals[n*redfac]['spiral'][1], adc_delay]
                                if sms and (sms_type == 1 or sms_type == 2):
                                    spiral_block.append(sms_blips)
                                if prep == prepscans:
                                    spiral_block.append(adc)
                                seq.add_block(*spiral_block)

                                # delay for TR
                                if fatsat:
                                    min_tr = rf.delay + ph.round_up_to_raster(rf_dur/2, decimals=5) + TE + adc_delay.delay + calc_duration(spirals[n*redfac]['reph'][0],spirals[n*redfac]['reph'][1]) + fatsat_del.delay + spoiler_dur
                                else:
                                    min_tr = rf.delay + ph.round_up_to_raster(rf_dur/2, decimals=5) + TE + adc_delay.delay + calc_duration(spirals[n*redfac]['reph'][0],spirals[n*redfac]['reph'][1],spoiler_z)
                                if TR < min_tr:
                                    raise ValueError('Minimum TR is {} ms.'.format(min_tr*1e3))
                                tr_delay = make_delay(d=TR-min_tr)
                                seq.add_block(tr_delay)

                                # rephaser (with spoiler, if no fatsat)
                                if fatsat:
                                    seq.add_block(spirals[n*redfac]['reph'][0],spirals[n*redfac]['reph'][1])
                                else:
                                    seq.add_block(spirals[n*redfac]['reph'][0],spirals[n*redfac]['reph'][1], spoiler_z)

                                # add protocol information
                                if seq_dest == 'scanner' and prep == prepscans:
                                    for seg in range(num_segments):
                                        acq = ismrmrd.Acquisition()
                                        if (n == intl_eff-1) and (seg == num_segments-1):
                                            acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
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

""" gradient echo

The following code generates a spoiled gradient echo (GRE) sequence used for testing purposes.
"""

if seq_type=='gre':

    # save loop variables before prepscans
    averages_ = averages
    intl_eff_ = intl_eff

    # imaging
    # slice ordering acc to Siemens method
    if slices_eff%2 == 1:
        slc = 0
    else:
        slc = 1
    for s in range(slices_eff):
        if s==int(slices_eff/2+0.5): 
            if slices_eff%2 == 1:
                slc = 1
            else:
                slc = 0
        
        rf.freq_offset = gz.amplitude * slice_res * (slc - (slices_eff - 1) / 2) * (1+dist_fac*1e-2)

        for prep in range(prepscans+1):
            if prep != prepscans:
                averages = intl_eff = 1
            else:
                averages = averages_
                intl_eff = intl_eff_
            for avg in range(averages):
                for n in range(intl_eff):
                    if rf_spoiling:
                        rf.phase_offset  = rf_phase / 180 * np.pi
                        adc.phase_offset = rf_phase / 180 * np.pi
                    if fatsat:
                        rf_fatsat.phase_offset = rf_phase / 180 * np.pi # always use RF spoiling for fat sat pulse
                        seq.add_block(rf_fatsat, fatsat_del)
                        seq.add_block(spoiler_x, spoiler_y, spoiler_z)
                    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
                    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

                    # excitation
                    if ptx:
                        seq.add_block(rf, rf_del)
                        if gz_rew is not None:
                            seq.add_block(gz_rew)
                    else:
                        seq.add_block(rf,gz,rf_del)
                        seq.add_block(gz_rew)

                    # spiral readout block with spoiler gradient
                    if skope and prep == prepscans:
                        trig_ctr += 1
                        if sms_type == 3:
                            blip_pre.amplitude = blip[0] * (n%kz_steps-kz_steps//2) / (kz_steps//2)
                            seq.add_block(trig, te_delay, blip_pre)
                        else:
                            seq.add_block(trig, te_delay)
                    else:
                        if sms_type == 3:
                            blip_pre.amplitude = blip[0] * (n%kz_steps-kz_steps//2) / (kz_steps//2)
                            seq.add_block(te_delay, blip_pre)
                        else:
                            seq.add_block(te_delay)
                    if (sms_type == 1 or sms_type == 2):
                        if prep != prepscans:
                            seq.add_block(spirals[n*redfac]['spiral'][0], spirals[n*redfac]['spiral'][1], sms_blips, adc_delay)
                        else:
                            seq.add_block(spirals[n*redfac]['spiral'][0], spirals[n*redfac]['spiral'][1], sms_blips, adc, adc_delay)
                    else:
                        if prep != prepscans:
                            seq.add_block(spirals[n*redfac]['spiral'][0], spirals[n*redfac]['spiral'][1], adc_delay)
                        else:
                            seq.add_block(spirals[n*redfac]['spiral'][0], spirals[n*redfac]['spiral'][1], adc, adc_delay)

                    if fatsat:
                        min_tr = rf.delay + ph.round_up_to_raster(rf_dur/2, decimals=5) + TE + adc_delay.delay + calc_duration(spirals[n*redfac]['reph'][0],spirals[n*redfac]['reph'][1]) + fatsat_del.delay + spoiler_dur
                    else:
                        min_tr = rf.delay + ph.round_up_to_raster(rf_dur/2, decimals=5) + TE + adc_delay.delay + calc_duration(spirals[n*redfac]['reph'][0],spirals[n*redfac]['reph'][1], spoiler_z)
                    if sms and sms_type == 3:
                        blip_rew.amplitude = -1*blip[0] * (n%kz_steps-kz_steps//2) / (kz_steps//2)
                        seq.add_block(blip_rew)
                        min_tr += calc_duration(blip_rew)

                    # delay for TR 
                    if TR < min_tr:
                        raise ValueError('Minimum TR is {} ms.'.format(min_tr*1e3))
                    tr_delay = make_delay(d=TR-min_tr)
                    seq.add_block(tr_delay)

                    # rephaser (with spoiler, if no fatsat)
                    if fatsat:
                        seq.add_block(spirals[n*redfac]['reph'][0], spirals[n*redfac]['reph'][1])
                    else:
                        seq.add_block(spirals[n*redfac]['reph'][0], spirals[n*redfac]['reph'][1], spoiler_z)

                    # add protocol information
                    if seq_dest == 'scanner' and prep == prepscans:
                        for seg in range(num_segments):
                            acq = ismrmrd.Acquisition()
                            if (n == intl_eff - 1) and (seg == num_segments - 1):
                                acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
                            acq.idx.kspace_encode_step_1 = n
                            acq.idx.slice = slc
                            acq.idx.average = avg
                            acq.idx.segment = seg
                            if sms_type == 3:
                                acq.idx.kspace_encode_step_2 = n%kz_steps

                            # save gradient only in first segment to save space
                            if seg == 0:
                                # use the trajectory field for the gradient array
                                if sms and sms_type > 0:
                                    acq.resize(trajectory_dimensions=3, number_of_samples=save_sp.shape[2], active_channels=0)
                                    acq.traj[:,:2] = np.swapaxes(save_sp[n*redfac],0,1) # [samples, dims]
                                    if sms_type == 3:
                                        acq.traj[0,2] = (blip_pre.area / system.gamma) * (n%kz_steps-kz_steps//2) / (kz_steps//2) / system.grad_raster_time
                                    if (sms_type == 1 or sms_type == 2):
                                        acq.traj[:,2] = sms_blips.waveform / system.gamma
                                else:
                                    acq.resize(trajectory_dimensions=save_sp.shape[1], number_of_samples=save_sp.shape[2], active_channels=0)
                                    acq.traj[:] = np.swapaxes(save_sp[n*redfac],0,1) # [samples, dims]

                            prot.append_acquisition(acq)
        
        slc += 2 # acquire every 2nd slice, afterwards fill slices inbetween

                # intl
            # avg
        # prepscans
    # slices

print(f"Sequence duration: {seq.duration()[0]:.2f} s")

# save b-values, directions and bDeltas as arrays
if seq_dest == 'scanner' and seq_type == 'diffusion':
    prot.append_array("b_values", np.asarray(bval_list, dtype=np.float32))
    prot.append_array("Directions", np.asarray(dir_list, dtype=np.float32))
    prot.append_array("bDeltas", np.asarray(bDelta_list, dtype=np.float32))

# save calculated b-tensors
btens = np.asarray(btens)
np.save('btens', btens)

if skope:
    print(f"Number of Skope triggers: {trig_ctr}.")
    seq.set_definition("SkopeNrDynamics", trig_ctr)
    seq.set_definition("SkopeNrSyncDynamics", sync_scans)
    seq.set_definition("SkopeAqDuration_ms", t_skope)
    seq.set_definition("SkopeInterleaveTR_ms", 1e3*min_dist)

#%% Plot gradient waveforms

if NOW == 0:
    wf_spiral_Gx = save_sp[0,0,:]
    wf_spiral_Gy = save_sp[0,1,:]
    rf_fatsat_signal = points_to_waveform(amplitudes=rf_fatsat.signal, grad_raster_time=system.grad_raster_time, times=rf_fatsat.t)
    rf_signal = points_to_waveform(amplitudes=rf.signal, grad_raster_time=system.grad_raster_time, times=rf.t)
    rf_refoc_signal = points_to_waveform(amplitudes=rf_refoc.signal, grad_raster_time=system.grad_raster_time, times=rf_refoc.t)

    # spherical
    trap_diff_z.amplitude = trap_diff_z_ampl
    wf_Gz_trapez_sph = ph.waveform_from_seqblock(trap_diff_z)/system.gamma
    wf_GxSinus_sph = np.concatenate((np.zeros(int((x+system.grad_raster_time)/system.grad_raster_time)),GxSinus_wf_first/system.gamma,np.zeros(int((x)/system.grad_raster_time))))
    wf_GySinus_sph = GySinus_wf_first/system.gamma
    wf_GzSinus_sph = np.concatenate((np.zeros(int((x+system.grad_raster_time)/system.grad_raster_time)),GzSinus_wf_first/system.gamma,np.zeros(int((x)/system.grad_raster_time))))
    rf_samples = round(refoc_dur/system.grad_raster_time)
    wf_GxSinus_sph_fin = np.concatenate((np.zeros_like(rf_fatsat_signal),np.zeros_like(rf_signal),np.zeros_like(ph.waveform_from_seqblock(gz_rew)),np.zeros(len(wf_Gz_trapez_sph)),wf_GxSinus_sph,np.zeros_like(rf_refoc_signal),wf_GxSinus_sph,np.zeros(len(wf_Gz_trapez_sph)),np.zeros(len(wf_spiral_Gx))))
    wf_GySinus_sph_fin = np.concatenate((np.zeros_like(rf_fatsat_signal),np.zeros_like(rf_signal),np.zeros_like(ph.waveform_from_seqblock(gz_rew)),np.zeros(len(wf_Gz_trapez_sph)),wf_GySinus_sph,np.zeros_like(rf_refoc_signal),wf_GySinus_sph,np.zeros(len(wf_Gz_trapez_sph)),np.zeros(len(wf_spiral_Gx))))
    wf_GzSinus_sph_fin = np.concatenate((np.zeros_like(rf_fatsat_signal),np.zeros_like(rf_signal),np.zeros_like(ph.waveform_from_seqblock(gz_rew)),wf_Gz_trapez_sph,wf_GzSinus_sph,np.zeros_like(rf_refoc_signal),wf_GzSinus_sph,wf_Gz_trapez_sph,np.zeros(len(wf_spiral_Gx))))
    wf_GSinus_sph_fin = np.zeros((len(wf_GxSinus_sph_fin),3))
    wf_GSinus_sph_fin[:,0] = wf_GxSinus_sph_fin
    wf_GSinus_sph_fin[:,1] = wf_GySinus_sph_fin
    wf_GSinus_sph_fin[:,2] = wf_GzSinus_sph_fin

    # linear
    wf_GxSinus_lin_x = np.sqrt(max(b_val[2])/b_val_max)*(wf_GxSinus_sph_fin)*np.sqrt(3)
    wf_GxSinus_lin = np.zeros((len(wf_GxSinus_lin_x),3))
    wf_GxSinus_lin[:,0] = wf_GxSinus_lin_x
    wf_GxSinus_lin[:,1] = np.zeros_like(wf_GxSinus_lin_x)
    wf_GxSinus_lin[:,2] = np.zeros_like(wf_GxSinus_lin_x)

    # planar
    wf_GxSinus_plan_y =  np.sqrt(max(b_val[0])/b_val_max)*(wf_GxSinus_sph_fin)*np.sqrt(3/2)
    wf_GxSinus_plan_z =  np.sqrt(max(b_val[0])/b_val_max)*(wf_GySinus_sph_fin)*np.sqrt(3/2)
    wf_GxSinus_plan = np.zeros((len(wf_GxSinus_plan_y),3))
    wf_GxSinus_plan[:,0] = np.zeros_like(wf_GxSinus_plan_y)
    wf_GxSinus_plan[:,1] = wf_GxSinus_plan_y
    wf_GxSinus_plan[:,2] = wf_GxSinus_plan_z

    # for spirals. adc, rf-pulses
    t_spiral = np.linspace(0, TR, len(wf_spiral_Gx))
    wf_rf = np.concatenate((rf_fatsat_signal,rf_signal,np.zeros_like(ph.waveform_from_seqblock(gz_rew)),np.zeros(len(wf_Gz_trapez_sph)),np.zeros(len(wf_GxSinus_sph)),rf_refoc_signal,np.zeros(len(wf_GxSinus_sph)),np.zeros(len(wf_Gz_trapez_sph)),np.zeros(len(wf_spiral_Gx))))
    t_rf = np.linspace(0, TR, len(wf_rf))
    wf_adc = np.concatenate((np.zeros(len(rf_fatsat_signal)),np.zeros(len(rf_signal)),np.zeros_like(ph.waveform_from_seqblock(gz_rew)),np.zeros(len(wf_Gz_trapez_sph)),np.zeros(len(wf_GxSinus_sph)),np.zeros(len(rf_refoc_signal)),np.zeros(len(wf_GxSinus_sph)),np.zeros(len(wf_Gz_trapez_sph)),np.ones(len(wf_spiral_Gx))))
    t_adc = np.linspace(0, TR, len(wf_adc))
    wf_read_Gx = np.concatenate((np.zeros(len(rf_fatsat_signal)),np.zeros(len(rf_signal)),np.zeros_like(ph.waveform_from_seqblock(gz_rew)),np.zeros(len(wf_Gz_trapez_sph)),np.zeros(len(wf_GxSinus_sph)),np.zeros(len(rf_refoc_signal)),np.zeros(len(wf_GxSinus_sph)),np.zeros(len(wf_Gz_trapez_sph)),wf_spiral_Gx))
    wf_read_Gy = np.concatenate((np.zeros(len(rf_fatsat_signal)),np.zeros(len(rf_signal)),np.zeros_like(ph.waveform_from_seqblock(gz_rew)),np.zeros(len(wf_Gz_trapez_sph)),np.zeros(len(wf_GxSinus_sph)),np.zeros(len(rf_refoc_signal)),np.zeros(len(wf_GxSinus_sph)),np.zeros(len(wf_Gz_trapez_sph)),wf_spiral_Gy))
    wf_read_Gz = np.concatenate((np.zeros_like(rf_fatsat_signal),np.ones_like(rf_signal)*0.009,np.ones_like(ph.waveform_from_seqblock(gz_rew))*(-0.054),np.zeros_like(wf_Gz_trapez_sph),np.zeros_like(wf_GzSinus_sph),np.ones_like(rf_refoc_signal)*0.009,np.zeros_like(wf_GzSinus_sph),np.zeros_like(wf_Gz_trapez_sph),np.zeros(len(wf_spiral_Gx))))
    t_read = t_adc

    fig,axes = plt.subplots(5,1,figsize=[30,30])
    t = np.linspace(0, TR, len(wf_GxSinus_sph_fin))
    axes[0].plot(t_adc*1e3, wf_adc, lw=10)
    axes[0].set_ylabel("ADC", fontsize=50)
    plt.yticks(fontsize=14)
    plt.grid('on')
    axes[1].plot(t_rf*1e3, wf_rf, lw=10)
    axes[1].set_ylabel("RF [V]", fontsize=50)
    plt.yticks(fontsize=14)
    plt.grid('on')
    axes[2].plot(t_read*1e3,wf_read_Gx*1e3,label="readout",color=(0, 0.4470, 0.7410), lw=10)
    axes[2].plot(t*1e3,wf_GxSinus_sph_fin*1e3,label="spherical",color=(0.9290, 0.6940 ,0.1250), lw=10)
    axes[2].plot(t*1e3,wf_GxSinus_lin_x*1e3,label="linear",color='green', lw=10)
    axes[2].set_ylabel(r"$g*_x(t)$ [mT/m]", fontsize=50)
    axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=50)
    plt.yticks(fontsize=14)
    plt.grid('on')
    axes[3].plot(t_read*1e3,wf_read_Gy*1e3,label="readout",color=(0, 0.4470, 0.7410), lw=10)
    axes[3].plot(t*1e3,wf_GySinus_sph_fin*1e3,label="spherical",color=(0.9290, 0.6940 ,0.1250), lw=10)
    axes[3].plot(t*1e3,wf_GxSinus_plan_y*1e3,label="planar",color=(0.4940, 0.1840 ,0.5560), lw=10)
    axes[3].set_ylabel(r"$g*_y(t)$ [mT/m]", fontsize=50)
    axes[3].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=50)
    plt.yticks(fontsize=14)
    plt.grid('on')
    axes[4].plot(t*1e3,wf_read_Gz*1e3,label="slice-selection",color=(0, 0.4470, 0.7410), lw=10)
    axes[4].plot(t*1e3,wf_GzSinus_sph_fin*1e3,label="spherical",color=(0.9290, 0.6940, 0.1250), lw=10)
    axes[4].plot(t*1e3,wf_GxSinus_plan_z*1e3,label="planar",color=(0.4940, 0.1840, 0.5560), lw=10)
    axes[4].set_ylabel(r"$g*_z(t)$ [mT/m]", fontsize=50)
    axes[4].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=50)
    for k in range(len(axes)):
        axes[k].tick_params(axis='both', labelsize=50)
        axes[k].grid('on')
    axes[2].set_ylim(-70,70)
    axes[3].set_ylim(-70,70)
    axes[4].set_ylim(-70,70)
    plt.xlabel("t [ms]", fontsize=50)
    plt.tight_layout()

elif NOW == 1:
    wf_spiral_Gx = save_sp[0,0,:]
    wf_spiral_Gy = save_sp[0,1,:]
    rf_fatsat_signal = points_to_waveform(amplitudes=rf_fatsat.signal, grad_raster_time=system.grad_raster_time, times=rf_fatsat.t)
    rf_signal = points_to_waveform(amplitudes=rf.signal, grad_raster_time=system.grad_raster_time, times=rf.t)
    rf_refoc_signal = points_to_waveform(amplitudes=rf_refoc.signal, grad_raster_time=system.grad_raster_time, times=rf_refoc.t)

    # spherical
    rf_samples = round(refoc_dur/system.grad_raster_time)
    sinusTime_first_samples = round((sinusTime_first.delay-calc_duration(GxSinus_first))/system.grad_raster_time)
    sinusTime_second_samples = round((rf_longer)/system.grad_raster_time)
    wf_GxSinus_sph_fin = np.concatenate((np.zeros_like(rf_fatsat_signal),np.zeros_like(rf_signal),np.zeros_like(ph.waveform_from_seqblock(gz_rew)),GxSinus_wf_first/system.gamma,np.zeros(int(sinusTime_first_samples)),np.zeros_like(rf_refoc_signal),np.zeros(int(sinusTime_second_samples)),GxSinus_wf_second/system.gamma,np.zeros(len(wf_spiral_Gx))))
    wf_GySinus_sph_fin = np.concatenate((np.zeros_like(rf_fatsat_signal),np.zeros_like(rf_signal),np.zeros_like(ph.waveform_from_seqblock(gz_rew)),GySinus_wf_first/system.gamma,np.zeros(int(sinusTime_first_samples)),np.zeros_like(rf_refoc_signal),np.zeros(int(sinusTime_second_samples)),GySinus_wf_second/system.gamma,np.zeros(len(wf_spiral_Gx))))
    wf_GzSinus_sph_fin = np.concatenate((np.zeros_like(rf_fatsat_signal),np.zeros_like(rf_signal),np.zeros_like(ph.waveform_from_seqblock(gz_rew)),GzSinus_wf_first/system.gamma,np.zeros(int(sinusTime_first_samples)),np.zeros_like(rf_refoc_signal),np.zeros(int(sinusTime_second_samples)),GzSinus_wf_second/system.gamma,np.zeros(len(wf_spiral_Gx))))
    wf_GSinus_sph_fin = np.zeros((len(wf_GxSinus_sph_fin),3))
    wf_GSinus_sph_fin[:,0] = wf_GxSinus_sph_fin
    wf_GSinus_sph_fin[:,1] = wf_GySinus_sph_fin
    wf_GSinus_sph_fin[:,2] = wf_GzSinus_sph_fin

    # linear
    wf_GxSinus_lin = np.zeros_like(wf_GSinus_sph_fin)
    wf_GxSinus_lin[:,0] = wf_GxSinus_sph_fin * np.sqrt(3)
    wf_GxSinus_lin[:,1] = np.zeros_like(wf_GxSinus_sph_fin)
    wf_GxSinus_lin[:,2] = np.zeros_like(wf_GxSinus_sph_fin)

    # planar
    wf_GxSinus_plan_y =  np.sqrt(max(b_val[0])/b_val_max)*(wf_GxSinus_sph_fin)*np.sqrt(3/2)
    wf_GxSinus_plan_z =  np.sqrt(max(b_val[0])/b_val_max)*(wf_GySinus_sph_fin)*np.sqrt(3/2)
    wf_GxSinus_plan = np.zeros((len(wf_GxSinus_plan_y),3))
    wf_GxSinus_plan[:,0] = np.zeros_like(wf_GxSinus_plan_y)
    wf_GxSinus_plan[:,1] = wf_GxSinus_plan_y
    wf_GxSinus_plan[:,2] = wf_GxSinus_plan_z

    # for spirals. adc, rf-pulses
    sinusTime_first_samples = round((sinusTime_first.delay-calc_duration(GxSinus_first))/system.grad_raster_time)
    sinusTime_second_samples = round((rf_longer)/system.grad_raster_time)
    t_spiral = np.linspace(0, TR, len(wf_spiral_Gx))
    wf_rf = np.concatenate((rf_fatsat_signal,rf_signal,np.zeros_like(ph.waveform_from_seqblock(gz_rew)),np.zeros_like(np.concatenate((GxSinus_wf_first/system.gamma,np.zeros(int(sinusTime_first_samples))))),rf_refoc_signal,np.zeros(int(sinusTime_second_samples)),np.zeros_like(GxSinus_wf_second/system.gamma),np.zeros(len(wf_spiral_Gx))))
    t_rf = np.linspace(0, TR, len(wf_rf))
    wf_adc = np.concatenate((np.zeros_like(np.concatenate((rf_fatsat_signal,rf_signal,np.ones_like(ph.waveform_from_seqblock(gz_rew))*0.054,np.zeros_like(np.concatenate((GxSinus_wf_first/system.gamma,np.zeros(int(sinusTime_first_samples))))),rf_refoc_signal))),np.zeros(int(sinusTime_second_samples)),np.zeros_like(GxSinus_wf_second/system.gamma),np.ones(len(wf_spiral_Gx))))
    t_adc = np.linspace(0, TR, len(wf_adc))
    wf_read_Gx = np.concatenate((np.zeros_like(np.concatenate((rf_fatsat_signal,rf_signal,np.ones_like(ph.waveform_from_seqblock(gz_rew))*0.054,np.zeros_like(np.concatenate((GxSinus_wf_first/system.gamma,np.zeros(int(sinusTime_first_samples))))),rf_refoc_signal))),np.zeros(int(sinusTime_second_samples)),np.zeros_like(GxSinus_wf_second/system.gamma),wf_spiral_Gx))
    wf_read_Gy = np.concatenate((np.zeros_like(np.concatenate((rf_fatsat_signal,rf_signal,np.ones_like(ph.waveform_from_seqblock(gz_rew))*0.054,np.zeros_like(np.concatenate((GxSinus_wf_first/system.gamma,np.zeros(int(sinusTime_first_samples))))),rf_refoc_signal))),np.zeros(int(sinusTime_second_samples)),np.zeros_like(GxSinus_wf_second/system.gamma),wf_spiral_Gy))
    wf_read_Gz = np.concatenate((np.zeros_like(rf_fatsat_signal),np.ones_like(rf_signal)*0.009,np.ones_like(ph.waveform_from_seqblock(gz_rew))*(-0.054),np.zeros_like(GzSinus_wf_first/system.gamma),np.zeros(int(sinusTime_first_samples)),np.ones_like(rf_refoc_signal)*0.009,np.zeros(int(sinusTime_second_samples)),np.zeros_like(GzSinus_wf_second/system.gamma),np.zeros(len(wf_spiral_Gx))))
    t_read = t_adc

    fig,axes = plt.subplots(5,1,figsize=[30,30])
    t = np.linspace(0, TR, len(wf_GxSinus_sph_fin))
    axes[0].plot(t_adc*1e3, wf_adc, lw=10)
    axes[0].set_ylabel("ADC", fontsize=50)
    plt.yticks(fontsize=14)
    plt.grid('on')
    axes[1].plot(t_rf*1e3, wf_rf, lw=10)
    axes[1].set_ylabel("RF [V]", fontsize=50)
    plt.yticks(fontsize=14)
    plt.grid('on')
    axes[2].plot(t_read*1e3,wf_read_Gx*1e3,label="readout",color=(0, 0.4470, 0.7410), lw=10)
    axes[2].plot(t*1e3,wf_GxSinus_sph_fin*1e3,label="spherical",color=(0.9290, 0.6940 ,0.1250), lw=10)
    axes[2].plot(t*1e3,wf_GxSinus_lin[:,0]*1e3,label="linear",color='green', lw=10)
    axes[2].set_ylabel(r"$g*_x(t)$ [mT/m]", fontsize=50)
    axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=50)
    plt.yticks(fontsize=14)
    plt.grid('on')
    axes[3].plot(t_read*1e3,wf_read_Gy*1e3,label="readout",color=(0, 0.4470, 0.7410), lw=10)
    axes[3].plot(t*1e3,wf_GySinus_sph_fin*1e3,label="spherical",color=(0.9290, 0.6940 ,0.1250), lw=10)
    axes[3].plot(t*1e3,wf_GxSinus_plan_y*1e3,label="planar",color=(0.4940, 0.1840 ,0.5560), lw=10)
    axes[3].set_ylabel(r"$g*_y(t)$ [mT/m]", fontsize=50)
    axes[3].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=50)
    plt.yticks(fontsize=14)
    plt.grid('on')
    axes[4].plot(t*1e3,wf_read_Gz*1e3,label="slice-selection",color=(0, 0.4470, 0.7410), lw=10)
    axes[4].plot(t*1e3,wf_GzSinus_sph_fin*1e3,label="spherical",color=(0.9290, 0.6940, 0.1250), lw=10)
    axes[4].plot(t*1e3,wf_GxSinus_plan_z*1e3,label="planar",color=(0.4940, 0.1840, 0.5560), lw=10)
    axes[4].set_ylabel(r"$g*_z(t)$ [mT/m]", fontsize=50)
    axes[4].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=50)
    for k in range(len(axes)):
        axes[k].tick_params(axis='both', labelsize=50)
        axes[k].grid('on')
    axes[2].set_ylim(-70,70)
    axes[3].set_ylim(-70,70)
    axes[4].set_ylim(-70,70)
    plt.xlabel("t [ms]", fontsize=50)
    plt.tight_layout()


elif NOW == 2:
    wf_spiral_Gx = save_sp[0,0,:]
    wf_spiral_Gy = save_sp[0,1,:]
    rf_fatsat_signal = points_to_waveform(amplitudes=rf_fatsat.signal, grad_raster_time=system.grad_raster_time, times=rf_fatsat.t)
    rf_signal = points_to_waveform(amplitudes=rf.signal, grad_raster_time=system.grad_raster_time, times=rf.t)
    rf_refoc_signal = points_to_waveform(amplitudes=rf_refoc.signal, grad_raster_time=system.grad_raster_time, times=rf_refoc.t)

    # spherical
    rf_samples = round(refoc_dur/system.grad_raster_time)
    sinusTime_first_samples = round((sinusTime_first.delay-calc_duration(GxSinus_first))/system.grad_raster_time)
    sinusTime_second_samples = round((rf_longer)/system.grad_raster_time)
    wf_GxSinus_sph = np.concatenate((np.zeros_like(rf_fatsat_signal),np.zeros_like(rf_signal),np.zeros_like(ph.waveform_from_seqblock(gz_rew)),GxSinus_wf_first/system.gamma,np.zeros(int(sinusTime_first_samples)),np.zeros_like(rf_refoc_signal),np.zeros(int(sinusTime_second_samples)),GxSinus_wf_second/system.gamma,np.zeros(len(wf_spiral_Gx))))
    wf_GySinus_sph = np.concatenate((np.zeros_like(rf_fatsat_signal),np.zeros_like(rf_signal),np.zeros_like(ph.waveform_from_seqblock(gz_rew)),GySinus_wf_first/system.gamma,np.zeros(int(sinusTime_first_samples)),np.zeros_like(rf_refoc_signal),np.zeros(int(sinusTime_second_samples)),GySinus_wf_second/system.gamma,np.zeros(len(wf_spiral_Gx))))
    wf_GzSinus_sph = np.concatenate((np.zeros_like(rf_fatsat_signal),np.zeros_like(rf_signal),np.zeros_like(ph.waveform_from_seqblock(gz_rew)),GzSinus_wf_first/system.gamma,np.zeros(int(sinusTime_first_samples)),np.zeros_like(rf_refoc_signal),np.zeros(int(sinusTime_second_samples)),GzSinus_wf_second/system.gamma,np.zeros(len(wf_spiral_Gx))))
    wf_GSinus_sph = np.zeros((len(wf_GxSinus_sph),3))
    wf_GSinus_sph[:,0] = wf_GxSinus_sph
    wf_GSinus_sph[:,1] = wf_GySinus_sph
    wf_GSinus_sph[:,2] = wf_GzSinus_sph

    # linear
    wf_Gx_trapez_lin = ph.waveform_from_seqblock(trap_diff_z)/system.gamma
    sinusTime_first_samples = round((sinusTime_first.delay-calc_duration(trap_diff_x))/system.grad_raster_time)
    spacing_samples = round(spacing_delay.delay/system.grad_raster_time)
    wf_Gx_lin = np.concatenate((np.zeros_like(rf_fatsat_signal),np.zeros_like(rf_signal),np.zeros_like(ph.waveform_from_seqblock(gz_rew)),wf_Gx_trapez_lin,np.zeros(int(sinusTime_first_samples)),np.zeros_like(rf_refoc_signal),np.zeros(int(spacing_samples)),wf_Gx_trapez_lin,np.zeros(len(wf_spiral_Gx))))
    wf_GxSinus_lin = np.zeros((len(wf_Gx_lin),3))
    wf_GxSinus_lin[:,0] = np.zeros_like(wf_Gx_lin)
    wf_GxSinus_lin[:,1] = np.zeros_like(wf_Gx_lin)
    wf_GxSinus_lin[:,2] = wf_Gx_lin

    # planar
    wf_GxSinus_plan_y =  np.sqrt(max(b_val[0])/b_val_max)*(wf_GxSinus_sph)*np.sqrt(3/2)
    wf_GxSinus_plan_z =  np.sqrt(max(b_val[0])/b_val_max)*(wf_GySinus_sph)*np.sqrt(3/2)
    wf_GxSinus_plan = np.zeros((len(wf_GxSinus_plan_y),3))
    wf_GxSinus_plan[:,0] = np.zeros_like(wf_GxSinus_plan_y)
    wf_GxSinus_plan[:,1] = wf_GxSinus_plan_y
    wf_GxSinus_plan[:,2] = wf_GxSinus_plan_z

    # for spirals. adc, rf-pulses
    sinusTime_first_samples = round((sinusTime_first.delay-calc_duration(GxSinus_first))/system.grad_raster_time)
    sinusTime_second_samples = round((rf_longer)/system.grad_raster_time)
    t_spiral = np.linspace(0, TR, len(wf_spiral_Gx))
    wf_rf = np.concatenate((rf_fatsat_signal,rf_signal,np.zeros_like(ph.waveform_from_seqblock(gz_rew)),np.zeros_like(np.concatenate((GxSinus_wf_first/system.gamma,np.zeros(int(sinusTime_first_samples))))),rf_refoc_signal,np.zeros(int(sinusTime_second_samples)),np.zeros_like(GxSinus_wf_second/system.gamma),np.zeros(len(wf_spiral_Gx))))
    t_rf = np.linspace(0, TR, len(wf_rf))
    wf_adc = np.concatenate((np.zeros_like(np.concatenate((rf_fatsat_signal,rf_signal,np.ones_like(ph.waveform_from_seqblock(gz_rew))*0.054,np.zeros_like(np.concatenate((GxSinus_wf_first/system.gamma,np.zeros(int(sinusTime_first_samples))))),rf_refoc_signal))),np.zeros(int(sinusTime_second_samples)),np.zeros_like(GxSinus_wf_second/system.gamma),np.ones(len(wf_spiral_Gx))))
    t_adc = np.linspace(0, TR, len(wf_adc))
    wf_read_Gx = np.concatenate((np.zeros_like(np.concatenate((rf_fatsat_signal,rf_signal,np.ones_like(ph.waveform_from_seqblock(gz_rew))*0.054,np.zeros_like(np.concatenate((GxSinus_wf_first/system.gamma,np.zeros(int(sinusTime_first_samples))))),rf_refoc_signal))),np.zeros(int(sinusTime_second_samples)),np.zeros_like(GxSinus_wf_second/system.gamma),wf_spiral_Gx))
    wf_read_Gy = np.concatenate((np.zeros_like(np.concatenate((rf_fatsat_signal,rf_signal,np.ones_like(ph.waveform_from_seqblock(gz_rew))*0.054,np.zeros_like(np.concatenate((GxSinus_wf_first/system.gamma,np.zeros(int(sinusTime_first_samples))))),rf_refoc_signal))),np.zeros(int(sinusTime_second_samples)),np.zeros_like(GxSinus_wf_second/system.gamma),wf_spiral_Gy))
    wf_read_Gz = np.concatenate((np.zeros_like(rf_fatsat_signal),np.ones_like(rf_signal)*0.009,np.ones_like(ph.waveform_from_seqblock(gz_rew))*(-0.054),np.zeros_like(GzSinus_wf_first/system.gamma),np.zeros(int(sinusTime_first_samples)),np.ones_like(rf_refoc_signal)*0.009,np.zeros(int(sinusTime_second_samples)),np.zeros_like(GzSinus_wf_second/system.gamma),np.zeros(len(wf_spiral_Gx))))
    t_read = t_adc

    fig,axes = plt.subplots(5,1,figsize=[30,30])
    t_sph = np.linspace(0, TR, len(wf_GxSinus_sph))
    t_lin = np.linspace(0, TR, len(wf_Gx_lin))
    axes[0].plot(t_adc*1e3, wf_adc, lw=10)
    axes[0].set_ylabel("ADC", fontsize=50)
    plt.yticks(fontsize=14)
    plt.grid('on')
    axes[1].plot(t_rf*1e3, wf_rf, lw=10)
    axes[1].set_ylabel("RF [V]", fontsize=50)
    plt.yticks(fontsize=14)
    plt.grid('on')
    axes[2].plot(t_read*1e3,wf_read_Gx*1e3,label="readout",color=(0, 0.4470, 0.7410), lw=10)
    axes[2].plot(t_sph*1e3,wf_GxSinus_sph*1e3,label="spherical",color=(0.9290, 0.6940 ,0.1250), lw=10)
    axes[2].plot(t_lin*1e3,wf_Gx_lin*1e3,label="linear",color='green', lw=10)
    axes[2].set_ylabel(r"$g*_x(t)$ [mT/m]", fontsize=50)
    axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=50)
    plt.yticks(fontsize=14)
    plt.grid('on')
    axes[3].plot(t_read*1e3,wf_read_Gy*1e3,label="readout",color=(0, 0.4470, 0.7410), lw=10)
    axes[3].plot(t_sph*1e3,wf_GySinus_sph*1e3,label="spherical",color=(0.9290, 0.6940 ,0.1250), lw=10)
    axes[3].plot(t_sph*1e3,wf_GxSinus_plan_y*1e3,label="planar",color=(0.4940, 0.1840 ,0.5560), lw=10)
    axes[3].set_ylabel(r"$g*_y(t)$ [mT/m]", fontsize=50)
    axes[3].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=50)
    plt.yticks(fontsize=14)
    plt.grid('on')
    axes[4].plot(t_sph*1e3,wf_read_Gz*1e3,label="slice-selection",color=(0, 0.4470, 0.7410), lw=10)
    axes[4].plot(t_sph*1e3,wf_GzSinus_sph*1e3,label="spherical",color=(0.9290, 0.6940, 0.1250), lw=10)
    axes[4].plot(t_sph*1e3,wf_GxSinus_plan_z*1e3,label="planar",color=(0.4940, 0.1840, 0.5560), lw=10)
    axes[4].set_ylabel(r"$g*_z(t)$ [mT/m]", fontsize=50)
    axes[4].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=50)
    for k in range(len(axes)):
        axes[k].tick_params(axis='both', labelsize=50)
        axes[k].grid('on')
    axes[2].set_ylim(-70,70)
    axes[3].set_ylim(-70,70)
    axes[4].set_ylim(-70,70)
    plt.xlabel("t [ms]", fontsize=50)
    plt.tight_layout()

#%% Make power spectrum

if NOW == 0:
    # spherical
    trap_diff_z.amplitude = trap_diff_z_ampl
    wf_Gz_trapez_sph = ph.waveform_from_seqblock(trap_diff_z)/system.gamma
    wf_GxSinus_sph = np.concatenate((np.zeros(int((x+system.grad_raster_time)/system.grad_raster_time)),GxSinus_wf_first/system.gamma,np.zeros(int((x)/system.grad_raster_time))))
    wf_GySinus_sph = GySinus_wf_first/system.gamma
    wf_GzSinus_sph = np.concatenate((np.zeros(int((x+system.grad_raster_time)/system.grad_raster_time)),GzSinus_wf_first/system.gamma,np.zeros(int((x)/system.grad_raster_time))))
    rf_samples = round(refoc_dur/system.grad_raster_time)
    wf_GxSinus_sph = np.concatenate((np.zeros(len(wf_Gz_trapez_sph)),wf_GxSinus_sph))
    wf_GySinus_sph = np.concatenate((np.zeros(len(wf_Gz_trapez_sph)),wf_GySinus_sph))
    wf_GzSinus_sph = np.concatenate((wf_Gz_trapez_sph,wf_GzSinus_sph))
    wf_GSinus_sph = np.zeros((len(wf_GxSinus_sph),3))
    wf_GSinus_sph[:,0] = wf_GxSinus_sph
    wf_GSinus_sph[:,1] = wf_GySinus_sph
    wf_GSinus_sph[:,2] = wf_GzSinus_sph

    # linear
    wf_GxSinus_lin_x = np.sqrt(max(b_val[2])/b_val_max)*(wf_GxSinus_sph)*np.sqrt(3)
    wf_GSinus_lin = np.zeros((len(wf_GxSinus_lin_x),3))
    wf_GSinus_lin[:,0] = wf_GxSinus_lin_x
    wf_GSinus_lin[:,1] = np.zeros_like(wf_GxSinus_lin_x)
    wf_GSinus_lin[:,2] = np.zeros_like(wf_GxSinus_lin_x)

    # planar
    wf_GxSinus_plan_y =  np.sqrt(max(b_val[0])/b_val_max)*(wf_GxSinus_sph)*np.sqrt(3/2)
    wf_GySinus_plan_z =  np.sqrt(max(b_val[0])/b_val_max)*(wf_GySinus_sph)*np.sqrt(3/2)
    wf_GSinus_plan = np.zeros((len(wf_GxSinus_plan_y),3))
    wf_GSinus_plan[:,0] = np.zeros_like(wf_GxSinus_plan_y)
    wf_GSinus_plan[:,1] = wf_GxSinus_plan_y
    wf_GSinus_plan[:,2] = wf_GySinus_plan_z

elif NOW == 1:
    # spherical
    rf_samples = round(refoc_dur/system.grad_raster_time)
    wf_GxSinus_sph = GxSinus_wf_first/system.gamma
    wf_GySinus_sph = GySinus_wf_first/system.gamma
    wf_GzSinus_sph = GzSinus_wf_first/system.gamma
    wf_GSinus_sph = np.zeros((len(wf_GxSinus_sph),3))
    wf_GSinus_sph[:,0] = wf_GxSinus_sph
    wf_GSinus_sph[:,1] = wf_GySinus_sph
    wf_GSinus_sph[:,2] = wf_GzSinus_sph

    # linear
    wf_GSinus_lin = np.zeros_like(wf_GSinus_sph)
    wf_GSinus_lin[:,0] = wf_GxSinus_sph * np.sqrt(3)
    wf_GSinus_lin[:,1] = np.zeros_like(wf_GySinus_sph)
    wf_GSinus_lin[:,2] = np.zeros_like(wf_GzSinus_sph)

    # planar
    wf_GxSinus_plan_y =  np.sqrt(max(b_val[0])/b_val_max)*(wf_GxSinus_sph)*np.sqrt(3/2)
    wf_GySinus_plan_z =  np.sqrt(max(b_val[0])/b_val_max)*(wf_GySinus_sph)*np.sqrt(3/2)
    wf_GSinus_plan = np.zeros((len(wf_GxSinus_plan_y),3))
    wf_GSinus_plan[:,0] = np.zeros_like(wf_GxSinus_plan_y)
    wf_GSinus_plan[:,1] = wf_GxSinus_plan_y
    wf_GSinus_plan[:,2] = wf_GySinus_plan_z

elif NOW == 2:
    # spherical
    rf_samples = round(refoc_dur/system.grad_raster_time)
    sinusTime_first_samples = round((sinusTime_first.delay-calc_duration(GxSinus_first))/system.grad_raster_time)
    sinusTime_second_samples = round((rf_longer)/system.grad_raster_time)
    wf_GxSinus_sph = np.concatenate((GxSinus_wf_first/system.gamma,np.zeros(sinusTime_first_samples)))
    wf_GySinus_sph = np.concatenate((GySinus_wf_first/system.gamma,np.zeros(sinusTime_first_samples)))
    wf_GzSinus_sph = np.concatenate((GzSinus_wf_first/system.gamma,np.zeros(sinusTime_first_samples)))
    wf_GSinus_sph = np.zeros((len(wf_GxSinus_sph),3))
    wf_GSinus_sph[:,0] = wf_GxSinus_sph
    wf_GSinus_sph[:,1] = wf_GySinus_sph
    wf_GSinus_sph[:,2] = wf_GzSinus_sph

    # linear
    wf_Gx_trapez_lin = ph.waveform_from_seqblock(trap_diff_z)/system.gamma
    sinusTime_first_samples = round((sinusTime_first.delay-calc_duration(trap_diff_x))/system.grad_raster_time)
    spacing_samples = round(spacing_delay.delay/system.grad_raster_time)
    wf_Gx_lin = np.concatenate((wf_Gx_trapez_lin,np.zeros(sinusTime_first_samples)))
    wf_GSinus_lin = np.zeros((len(wf_Gx_lin),3))
    wf_GSinus_lin[:,0] = wf_Gx_lin
    wf_GSinus_lin[:,1] = np.zeros_like(wf_Gx_lin)
    wf_GSinus_lin[:,2] = np.zeros_like(wf_Gx_lin)

    # planar
    wf_GxSinus_plan_y =  np.sqrt(max(b_val[0])/b_val_max)*(wf_GxSinus_sph)*np.sqrt(3/2)
    wf_GySinus_plan_z =  np.sqrt(max(b_val[0])/b_val_max)*(wf_GySinus_sph)*np.sqrt(3/2)
    wf_GSinus_plan = np.zeros((len(wf_GxSinus_plan_y),3))
    wf_GSinus_plan[:,0] = np.zeros_like(wf_GxSinus_plan_y)
    wf_GSinus_plan[:,1] = wf_GxSinus_plan_y
    wf_GSinus_plan[:,2] = wf_GySinus_plan_z

# write .mat file
mdic = {"wf_STE": wf_GSinus_sph, "wf_LTE": wf_GSinus_lin,"wf_PTE": wf_GSinus_plan} # T/m
savemat("WF_STE_LTE_PTE.mat", mdic)

# FFT
from scipy.signal import find_peaks
from scipy.interpolate import InterpolatedUnivariateSpline

fig,axes = plt.subplots(3,1,figsize=[15,30])
#fig.suptitle(seq_name,fontsize=60)
#fig.subplots_adjust(top=0.95)
labels = ['x','y','z']
for k,key in enumerate(mdic.keys()):
    grad_wf = mdic[key]
    dt = system.grad_raster_time
    fa = 1/dt
    td = dt*grad_wf.shape[0]
    df = 1/td
    freq_axis = np.arange(-fa/2,fa/2,df)
    freq_axis_spline = np.linspace(-fa/2,fa/2,100000)
    if NOW == 0:
        offset = 13.5
    elif NOW==1:
        offset = 11.5
    else:
        offset = 0
    freq_axis = freq_axis + offset
    freq_axis_spline = freq_axis_spline + offset

    q = np.zeros_like(grad_wf)
    for i in range(grad_wf.shape[1]):
        q[:,i] = system.gamma * 2 * np.pi*np.cumsum(grad_wf[:,i])*system.grad_raster_time

    if key == "wf_LTE" and NOW==2 or NOW==0 or NOW==1:
        grad_wf = grad_wf- np.mean(grad_wf,axis=0)
        q = q - np.mean(q,axis=0)
    for i in range(grad_wf.shape[1]):
        FFT = np.abs(np.fft.fftshift(np.fft.fft(grad_wf[:,i])))
        
        if not np.all(FFT==0) and NOW !=0:
            signal = FFT
            peaks,_ = find_peaks(-signal)
            minima_freq = freq_axis[peaks]
            minima_signal = signal[peaks]
            spline = InterpolatedUnivariateSpline(minima_freq,minima_signal,k=1)
            spline_signal = spline(freq_axis)
            signal_corrected = abs(FFT - spline_signal)
        else:
            signal_corrected = FFT

        signal_norm = signal_corrected/max(signal_corrected)
        spline = InterpolatedUnivariateSpline(freq_axis,signal_norm,k=3)
        signal_norm_spline = spline(freq_axis_spline)
        axes[k].plot(freq_axis_spline,signal_norm_spline,label=labels[i],lw=10)

    axes[k].set_ylabel(r"abs(FFT)$_\mathrm{norm}$", fontsize=70)
    axes[k].tick_params(axis='both', labelsize=60)
    axes[k].grid('on')
    axes[k].set_xlim([0,200])
    axes[k].set_ylim([-0.1,1.1])
    axes[k].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=60)
plt.xlabel("f [Hz]", fontsize=70)

#%% Plotting and test report

if plot or check_TE:
    ktraj_adc, ktraj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
    if check_TE and seq_type=='diffusion':
        # sensitivity of check is set to 1us as deviation of 1us can occur due to an assymetrical pulse
        if skope:
            te_check1 = np.allclose(t_adc[sync_scans*num_samples::num_samples]-t_excitation, TE, atol=5e-6) # check echo timing, 5us deviation is okay
        else:
            te_check1 = np.allclose(t_adc[::num_samples]-t_excitation, TE, atol=5e-6)
        te_check2 = np.allclose(t_refocusing-t_excitation, TE/2, atol=1e-6)       # check refocusing timing
        if not (te_check1 and te_check2):
            raise ValueError('TE check failed. Check timing of diffusion sequence.')
        else:
            print('TE check succesful.')
    if plot:
        seq.plot(time_disp = 's')

        time_axis = np.arange(1, ktraj.shape[1] + 1) * system.grad_raster_time
        plt.figure()
        plt.plot(time_axis, ktraj.T)
        plt.plot(t_adc, ktraj_adc[0, :], '.')
        plt.figure()
        plt.plot(ktraj[0, :], ktraj[1, :], 'b')
        plt.axis('equal')
        plt.plot(ktraj_adc[0, :], ktraj_adc[1, :], 'r.')

if test_report:
    seq.test_report()

#%% save sequence and write sequence & protocol to scanner exchange folder

if seq_dest=='sim': # save to simulation folder
    if os.path.exists("/mnt/pulseq/"):
        seq.write('/mnt/pulseq/external.seq')
    if grads_off and os.path.exists("../Pulseq_sequences/"+date):
        seq.write(f"../Pulseq_sequences/{date}/{filename}_refgradsoff.seq") # save for later ECC simulation

if seq_dest=='scantest':
    seq.write(f"/mnt/mrdata/pulseq_exchange/{filename}.seq")    

elif seq_dest=='scanner':
    # create new directory if needed
    Path("../Pulseq_sequences/"+date).mkdir(parents=True, exist_ok=True) 
    
    # save seq with hash
    seqfile = f"../Pulseq_sequences/{date}/{filename}.seq"
    seq.write(seqfile)
    seq_hash = seq.get_hash()
    seqfile_hashed = os.path.splitext(seqfile)[0] + f"_{seq_hash[:5]}.seq"
    os.rename(seqfile, seqfile_hashed)

    # add hash to protocol
    signature = ismrmrd.xsd.userParameterStringType()
    signature.name = 'seq_signature'
    signature.value = seq_hash
    hdr.userParameters.userParameterString.append(signature)
    prot.write_xml_header(hdr.toXML('utf-8'))
    prot.close()
    ismrmrd_file_hash = os.path.splitext(ismrmrd_file)[0]+ f"_{seq_hash[:5]}.h5"
    if os.path.exists(ismrmrd_file_hash):
        raise ValueError("Renaming protocol file failed as file with same hash already exists")
    else:
        os.rename(ismrmrd_file, ismrmrd_file_hash)

    # prot & seq to scanner
    shutil.copyfile(seqfile_hashed, f"/mnt/mrdata/pulseq_exchange/{filename}_{seq_hash[:5]}.seq") # seq to scanner
    shutil.copyfile(ismrmrd_file_hash, f"/mnt/scratch/fire_setup/dependency/metadata/{filename}_{seq_hash[:5]}.h5")

    # write protocol parameters
    protparams = {}
    protparams['B0']         = B0
    protparams['seq_type']   = seq_type
    protparams['refscan']    = refscan
    protparams['fov']        = fov
    protparams['TR']         = TR
    protparams['TE']         = TE
    protparams['min_TE']     = min_te
    protparams['res']        = res
    protparams['slice_res']  = slice_res
    protparams['slices']     = slices
    protparams['averages']   = averages
    protparams['repetitions']= repetitions
    protparams['inner_avg']  = inner_avg
    protparams['prepscans']  = prepscans
    protparams['noisescans'] = noisescans
    protparams['dist_fac']   = dist_fac
    protparams['TA_ref']     = dur_until_ref
    protparams['TA']         = seq.duration()[0]

    protparams['rf_dur']        = rf_dur
    protparams['rf_refoc_dur']  = rf_refoc_dur
    protparams['tbp_exc']       = tbp_exc
    protparams['tbp_refoc']     = tbp_refoc
    protparams['flip_angle']    = flip_angle
    protparams['rf_spoiling']   = rf_spoiling
    protparams['sms']           = sms
    protparams['sms_factor']    = sms_factor
    protparams['sms_type']      = sms_type
    protparams['kz_steps']      = kz_steps
    protparams['wave_periods']      = wave_periods
    protparams['fat_suppr']     = fatsat
    protparams['fatsat_tbp']    = fatsat_tbp
    protparams['slr_rf']        = slr_rf    
    protparams['ext_rf']        = ext_rf  
    protparams['ext_path_exc']  = ext_path_exc  
    protparams['ext_path_ref']  = ext_path_ref  
    protparams['verse_rf']      = verse_rf    
    protparams['b1_max_fac']    = b1_max_fac         
    protparams['energy_fac']    = energy_fac
    protparams['ptx']           = ptx
    protparams['refoc_fac']     = refoc_fac

    protparams['max_slew']    = max_slew
    protparams['spiral_slew'] = spiral_slew
    protparams['max_grad']    = max_grad
    protparams['max_grad_sp'] = max_grad_sp
    protparams['Eff_Nintl']   = intl_eff # save real number of spiral intlv
    protparams['redfac']      = redfac
    protparams['max_rot']     = max_rot
    protparams['spiraltype']  = spiraltype
    protparams['spiral_os']   = spiral_os
    protparams['trans_beg']   = trans_beg
    protparams['trans_end']   = trans_end
    protparams['pre_emph']    = pre_emph
    protparams['skope']       = skope
    protparams['sync_scans']  = sync_scans
    protparams['trigger']     = trig_ctr
    protparams['trig_skip']   = trig_skip

    protparams['dwelltime']    = dwelltime
    protparams['num_samples']  = int(num_samples)
    protparams['num_segments'] = num_segments
    protparams['spiral_delay'] = spiral_delay
    protparams['os_factor'] = os_factor
    protparams['readout_dur'] = readout_dur

    if refscan:
        protparams['refscan_type'] = refscan
        protparams['flip_refscan'] = flip_refscan
        if refscan == 2:
            protparams['res_refscan'] = res_refscan
            protparams['bw_refscan'] = bw_refscan
            protparams['separate_TR'] = separate_tr

    if seq_type=='diffusion':
        protparams['b_val']             = b_val
        protparams['directions']        = directions
        protparams['diff_slewrate']     = diff_slewrate
        protparams['diff_maxgrad']      = diff_maxgrad
        protparams['diff_delay']        = te_delay.delay
        protparams['vol_TR']            = vol_TR
        protparams['NOW']               = NOW

    with open(f"../Protocols/{date}/{filename}_{seq_hash[:5]}_protparams.json", 'w') as f:
        json.dump(protparams, f)
