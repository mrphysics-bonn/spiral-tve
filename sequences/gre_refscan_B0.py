""" This function adds a reference scan to a sequence for calculation of sensitivity maps and B0 maps

    seq: Existing PyPulseq sequence
    prot: ISMRMRD metadata file
    system: MR system parameters
    params: Refscan parameters
    grads_off: Turn gradients off (only for ECC simulation)
"""
import math
import numpy as np

import ismrmrd

from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts

import pulseq_helper as ph

def gre_refscan_B0(seq, prot=None, system=Opts(), params=None, grads_off=False):

    # limit slew rate and maximum amplitude to avoid stimulation
    save_slew = system.max_slew
    save_grad = system.max_grad
    system.max_slew = 170 * system.gamma
    system.max_grad = 50e-3 * system.gamma

    # default parameters
    params_def = {"TE": [2.04e-3, 4.08e-3], "fov":210e-3, "res":2e-3, "ref_lines": 0, "flip_angle":12, "rf_dur":0.8e-3, 
                    "tbp": 2, "slices":1, "slice_res":2e-3, "dist_fac":0, "readout_bw": 1200, 
                    'separate_tr': False, 'center_out': False, 'dim3': False, 'slice_os': 0}
    for key in params:
        params_def[key] = params[key] # insert incoming parameters
    params = params_def
    
    if params['dim3']:
        slices = 1
        Nz = int(params['slices'] * (1+params['slice_os']))
        if Nz%2:
            Nz += 1 # even matrix size (oversampled slices will get cut in recon)
        slc_res = Nz * params['slice_res']
    else:
        slices = params['slices']
        Nz = 1
        slc_res = params['slice_res']

    # RF
    rf, gz, gz_reph, rf_del = make_sinc_pulse(flip_angle=params["flip_angle"] * math.pi / 180, duration=params["rf_dur"], slice_thickness=slc_res,
                                apodization=0.5, time_bw_product=params["tbp"], system=system, return_gz=True, return_delay=True)

    # Calculate readout gradient and ADC parameters
    delta_k = 1 / params["fov"]
    if params["ref_lines"] > 0:
        Ny = int(params["ref_lines"])
    else:
        Ny = int(params["fov"]/params["res"]+0.5)
    if Ny%2: 
        Ny -= 1 # even matrix size (+matrix size should not be bigger than imaging matrix)
    Nx = Ny
    samples = 2*Nx # 2x oversampling
    gx_flat_time_us = int(1e6/params["readout_bw"]) # readout_bw is in Hz/Px
    dwelltime = ph.trunc_to_raster(1e-6*gx_flat_time_us / samples, decimals=7)
    gx_flat_time = round(dwelltime*samples, 5)
    if (1e5*gx_flat_time %2 == 1):
        gx_flat_time += 10e-6 # even flat time
    diff_flat_adc = gx_flat_time - (dwelltime*samples)

    # Make readout gradients
    gx_flat_area = Nx * delta_k * (gx_flat_time / (dwelltime*samples)) # compensate for longer flat time than ADC
    gx = make_trapezoid(channel='x', flat_area=gx_flat_area, flat_time=gx_flat_time, system=system)
    gx_mid = make_trapezoid(channel='x', area=-gx.area, system=system)
    gx_pre = make_trapezoid(channel='x', area=-gx.area / 2, system=system)

    # Calculate phase encode gradients
    phase_areas = (np.arange(Ny) - Ny / 2) * delta_k
    delta_kz = 1 / slc_res
    if params['dim3']:
        phase_areas_z = (np.arange(Nz)[::-1] - Nz / 2) * delta_kz # switch orientation for recon
        slc_res = 0 # this is just for not setting any frequency offset below
    else:
        phase_areas_z = np.zeros(Nz)

    # spoilers
    gx_spoil = make_trapezoid(channel='x', area=2.5 * Nx * delta_k, rise_time=gx.rise_time, system=system)
    gx_end = ph.merge_ramps([gx, gx_spoil], system=system) # merge x-spoiler with read gradient

    # calculate TE delays
    max_gy_pre = make_trapezoid(channel='y', area=max(abs(phase_areas)), system=system)
    max_gz_pre = make_trapezoid(channel='z', area=max(abs(phase_areas_z+gz_reph.area)), system=system)
    if max(abs(phase_areas_z)) > 0:
        max_gz_rew = make_trapezoid(channel='z', area=max(abs(phase_areas_z)), system=system)
    else:
        max_gz_rew = make_trapezoid(channel='z', area=0, duration=30e-6, system=system) # dummy
    gy_pre_dur = calc_duration(max_gy_pre)
    gz_pre_dur = calc_duration(max_gz_pre)
    gz_rew_dur = calc_duration(max_gz_rew)

    # delays for rewinder - minimum delay is duration of readout gradient
    delay_end_gy = max(calc_duration(gx), calc_duration(gx_end) - gy_pre_dur)
    delay_end_gz = max(calc_duration(gx), calc_duration(gx_end) - gz_pre_dur)

    # calculate TE delays
    min_TE = np.round((rf_del.delay-rf.delay-rf.t[-1]/2 + calc_duration(gx_pre, max_gy_pre, max_gz_pre) + calc_duration(gx) / 2) / seq.grad_raster_time) * seq.grad_raster_time
    params["TE"] = sorted(params["TE"])
    n_TE = len(params["TE"])
    d_TE = np.asarray(params["TE"][1:]) - np.asarray(params["TE"][:-1])
    delay_TE = [params["TE"][0] - min_TE]
    if delay_TE[0] < 0:
        raise ValueError(f"First TE is too small by {1e3*abs(delay_TE[0])} ms.")
    for n_te, d_te in enumerate(d_TE):
        if not params['separate_tr']:
            echo_delay = d_te - calc_duration(gx) - calc_duration(gx_mid)
        else:
            echo_delay = params["TE"][n_te+1] - min_TE
        delay_TE.append(echo_delay)
        if echo_delay < 0:
            raise ValueError(f"TE {n_te+2} is too small by {1e3*abs(echo_delay)} ms. Increase readout bandwidth.")

    # delay prephaser to reduce stimu
    pre_dur = max(calc_duration(gx_pre), gz_pre_dur, gy_pre_dur)
    gx_pre.delay = delay_TE[0] + pre_dur - calc_duration(gx_pre)
    gy_pre_delay = delay_TE[0] + pre_dur - gy_pre_dur

    # ADC 
    adc = make_adc(num_samples=samples, dwell=dwelltime, delay=gx.rise_time+diff_flat_adc/2, system=system)

    # RF spoiling
    rf_spoiling_inc = 117
    rf_phase = 0
    rf_inc = 0

    # build sequence
    if params['separate_tr']:
        TR = slices * (rf_del.delay + pre_dur + delay_TE[-1] 
             + calc_duration(gx) + calc_duration(gx_spoil,max_gy_pre,max_gz_rew))
    else:
        TR = slices * (rf_del.delay + pre_dur + delay_TE[0] + (n_TE-1)*(calc_duration(gx) + calc_duration(gx_mid)) + np.sum(delay_TE[1:]) + max(gy_pre_dur+delay_end_gy, gz_rew_dur+delay_end_gz, calc_duration(gx_end)))

    print(f"Refscan volume TR: {TR*1e3:.3f} ms")

   # turn off gradients, if selected
    if grads_off:
        gz.amplitude = 0
        gx_pre.amplitude = 0
        gx_spoil.amplitude = 0
        gx.amplitude = 0
        gx_mid.amplitude = 0
        gx_end.waveform = np.zeros_like(gx_end.waveform)
        phase_areas *= 0
        phase_areas_z *= 0
        gz_reph.area = 0
    
    # center out readout
    if params['center_out']:
        phs_ix = Ny // 2
        prepscans = min(int(3/TR),25) # more dummy scans as we start in kspace center
    else:
        phs_ix = 0
        prepscans = min(int(1/TR),25)

    print(f"Refscan prepscans: {prepscans} ")

    if not params['separate_tr']:

        # imaging scans
        for i in range(-prepscans, Ny):
            ix = i if i >= 0 else None
            if ix is not None:
                if params['center_out']:
                    phs_ix += (-1)**(ix%2) * ix
                else:
                    phs_ix = ix
            if slices%2 == 1:
                slc = 0
            else:
                slc = 1

            # RF spoiling
            rf.phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi
            rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

            for s in range(slices):
                if s==int(slices/2+0.5): 
                    if slices%2 == 1:
                        slc = 1
                    else:
                        slc = 0
                rf.freq_offset = gz.amplitude * params["slice_res"] * (slc - (slices - 1) / 2) * (1+params["dist_fac"]*1e-2)

                for nz in range(Nz):
                    seq.add_block(rf, gz, rf_del)
                    gy_pre = make_trapezoid(channel='y', area=phase_areas[phs_ix], duration=gy_pre_dur, delay=gy_pre_delay, system=system)
                    gz_pre = make_trapezoid(channel='z', area=phase_areas_z[nz]+gz_reph.area, duration=gz_pre_dur, system=system)
                    delay_pre = make_delay(delay_TE[0]+pre_dur)
                    seq.add_block(gx_pre, gy_pre, gz_pre, delay_pre)
                    for k in range(n_TE-1):
                        if i < 0:
                            seq.add_block(gx)
                        else:
                            seq.add_block(gx, adc)
                        seq.add_block(gx_mid)
                        seq.add_block(make_delay(delay_TE[k+1]))
                    gy_pre.amplitude = -gy_pre.amplitude
                    gy_pre.delay = delay_end_gy
                    gz_rew = make_trapezoid(channel='z', area=-phase_areas_z[nz], duration=gz_rew_dur, delay=delay_end_gz, system=system)
                    if i < 0:
                        seq.add_block(gx_end, gy_pre, gz_rew)
                    else:
                        seq.add_block(gx_end, adc, gy_pre, gz_rew)

                    if prot is not None and i >= 0:
                        for k in range(n_TE):
                            acq = ismrmrd.Acquisition()
                            acq.idx.kspace_encode_step_1 = phs_ix
                            acq.idx.kspace_encode_step_2 = nz
                            acq.idx.slice = slc
                            acq.idx.contrast = k
                            acq.setFlag(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)
                            if i == Ny-1 and nz == Nz-1 and k == n_TE-1:
                                acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
                            prot.append_acquisition(acq)
                    
                slc += 2 # interleaved
    else:

        # imaging scans
        for i in range(-prepscans, Ny):
            ix = i if i >= 0 else None
            if ix is not None:
                if params['center_out']:
                    phs_ix += (-1)**(ix%2) * ix
                else:
                    phs_ix = ix

            for k in range(n_TE):
                if slices%2 == 1:
                    slc = 0
                else:
                    slc = 1

                # RF spoiling
                rf.phase_offset = rf_phase / 180 * np.pi
                adc.phase_offset = rf_phase / 180 * np.pi
                rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
                rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

                for s in range(slices):
                    if s==int(slices/2+0.5): 
                        if slices%2 == 1:
                            slc = 1
                        else:
                            slc = 0
                    rf.freq_offset = gz.amplitude * params["slice_res"] * (slc - (slices - 1) / 2) * (1+params["dist_fac"]*1e-2)

                    for nz in range(Nz):
                        seq.add_block(rf, gz, rf_del)
                        gy_pre = make_trapezoid(channel='y', area=phase_areas[phs_ix], duration=gy_pre_dur, delay=gy_pre_delay, system=system)
                        gz_pre = make_trapezoid(channel='z', area=phase_areas_z[nz]+gz_reph.area, duration=gz_pre_dur, system=system)
                        seq.add_block(gx_pre, gy_pre, gz_pre)
                        seq.add_block(make_delay(delay_TE[k]))
                        if i < 0:
                            seq.add_block(gx)
                        else:
                            seq.add_block(gx, adc)
                        if k<n_TE-1:
                            seq.add_block(make_delay(d=params["TE"][-1]-params["TE"][k])) # account for longer 2nd echo, spoiler always in same position
                        gy_pre.amplitude = -gy_pre.amplitude
                        gy_pre.delay = 0
                        gz_rew = make_trapezoid(channel='z', area=-phase_areas_z[nz], duration=gz_rew_dur, system=system)
                        seq.add_block(gx_spoil, gy_pre, gz_rew)

                        if prot is not None and i>=0:
                            acq = ismrmrd.Acquisition()
                            acq.idx.kspace_encode_step_1 = phs_ix
                            acq.idx.kspace_encode_step_2 = nz
                            acq.idx.slice = slc
                            acq.idx.contrast = k
                            acq.setFlag(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)
                            if i == Ny-1 and nz == Nz-1 and k == n_TE-1:
                                acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
                            prot.append_acquisition(acq)
                    
                    slc += 2
    
    delay_end_ref = make_delay(d=5) # 5s delay after reference scan to allow for relaxation
    seq.add_block(delay_end_ref)
    system.max_slew = save_slew
    system.max_grad = save_grad

    # use array to save echo times in protocol
    if prot is not None and n_TE>1:
        prot.append_array("echo_times", np.asarray(params["TE"]))
