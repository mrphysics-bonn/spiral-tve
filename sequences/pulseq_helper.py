"""
helper functions for spiral pulseq sequence    
"""   

import numpy as np
import math

dt_grad  = 10e-6     # gradient raster [s]
dt_skope = 1e-6      # skope raster [s]
dt_quot  = int(dt_grad/dt_skope)
dt_rf = 1e-6 # rf raster

#############
# FFTs
#############

def ifft(sig, dim=None):
    """ Computes the Fourier transform from k-space to image space 
    along a given or all dimensions

    :param img: image space data
    :param dim: vector of dimensions to transform
    :returns: data in k-space (along transformed dimensions)
    """
    import collections.abc
    if dim is None:
        dim = range(sig.ndim)
    elif not isinstance(dim, collections.abc.Iterable):
        dim = [dim]

    sig = np.fft.ifftshift(sig, axes=dim)
    sig = np.fft.ifftn(sig, axes=dim)
    sig = np.fft.fftshift(sig, axes=dim)

    return sig

def fft(sig, dim=None):
    """ Computes the Fourier transform from image space to k-space
    along a given or all dimensions

    :param img: image space data
    :param dim: vector of dimensions to transform
    :returns: data in k-space (along transformed dimensions)
    """
    import collections.abc
    if dim is None:
        dim = range(sig.ndim)
    elif not isinstance(dim, collections.abc.Iterable):
        dim = [dim]

    sig = np.fft.ifftshift(sig, axes=dim)
    sig = np.fft.fftn(sig, axes=dim)
    sig = np.fft.fftshift(sig, axes=dim)

    return sig

################
# Functions for gradient calculations
################

def merge_ramps(grads: list, system=None):
    """
    Merge the ramps of trapezoidal gradients

    grads: List of trapezoidal gradients in merging order. Ramps should have equal length.
    """

    delay = grads[0].delay # initial delay of first gradient
    for grad in grads:
        grad.delay = delay
        delay += grad.rise_time + grad.flat_time 
        
    return add_gradients(grads, system)

def add_gradients(grads: list, system=None):
    """
    Adds up gradient events
    """
    from pypulseq.make_arbitrary_grad import make_arbitrary_grad

    if system==None:
        raise ValueError('Provide the MR System limits.')

    # First gradient defines channel
    channel = grads[0].channel

    # read in gradient waveforms
    grad_length = []
    grad_list = []
    for grad in grads:
        w = waveform_from_seqblock(grad)
        grad_list.append(w)
        grad_length.append(len(w))
    
    # prolong waveforms to maximum length and add them up
    length=max(grad_length)
    added_grad = np.array(np.zeros(length))
    for grad in grad_list:
        grad = np.append(grad, np.zeros(length-len(grad)))
        added_grad += grad

    return make_arbitrary_grad(channel=channel, waveform=added_grad, system=system)

def merge_verse_ramps(verse_obj, crusher, system):
    """
    Merge the ramps of crushers and VERSE refocusing gradients

    verse_obj: Verse pulse object
    crusher: crusher (PyPulseq object)
    """

    vs_ramps = verse_obj.ramp_len
    crusher_ramps = round(crusher.rise_time / system.grad_raster_time)
    crusher_flat = round(crusher.flat_time / system.grad_raster_time)
    crusher_dur = crusher_ramps + crusher_flat

    g_vs = verse_obj.g_vs
    g_vs_new = g_vs[vs_ramps[0]: -vs_ramps[1]]
    ramps_new = [np.linspace(0,g_vs_new[0],crusher_ramps), np.linspace(g_vs_new[-1],0,crusher_ramps)]
    g_vs_new = np.concatenate((np.zeros(crusher_dur), ramps_new[0], g_vs_new, ramps_new[1], np.zeros(crusher_dur)))

    g_crusher = waveform_from_seqblock(crusher)
    g_vs_new[:crusher_dur+crusher_ramps] += g_crusher
    g_vs_new[-(crusher_dur+crusher_ramps):] += g_crusher

    return g_vs_new

def waveform_from_seqblock(seq_block):
    """
    extracts gradient waveform from Pypulseq sequence block
    """
    from pypulseq.Sequence.sequence import Sequence

    if seq_block.channel == 'x':
        axis = 0
    elif seq_block.channel == 'y':
        axis = 1
    elif seq_block.channel == 'z':
        axis = 2
    else:
        raise ValueError('No valid gradient waveform')
    dummy_seq = Sequence() # helper dummy sequence
    dummy_seq.add_block(seq_block)
    return dummy_seq.gradient_waveforms()[axis,:-1] # last value is a zero that does not belong to the waveform

def trap_from_area(area, system, slewrate = None, max_grad = None):
    """
    Calculate minimum time trapezoidal gradient from gradient area/moment
    In the last step the amplitude is recalculated to get the right moment and sign of the gradient

    area: Gradient area [1/m]
    system: system configuration from Pypulseq
    slewrate: (Optional) set maximum slewrate [T/m/s]
    max_grad: (Optional) set maximum gradient
    """
    if slewrate is not None:
        tmp_slew = system.max_slew
        system.max_slew = slewrate*system.gamma

    if max_grad is not None:
        tmp_grad = system.max_grad
        system.max_grad = max_grad * system.gamma

    if abs(area) < system.max_grad*round_up_to_raster(system.max_grad/system.max_slew, decimals=5):
        ftop = 0
        amp = np.sqrt(abs(area)*system.max_slew)
        ramp = round_up_to_raster(amp/system.max_slew, decimals=5) 
        amp = area/ramp

    else:
        amp = system.max_grad
        ramp = round_up_to_raster(amp/system.max_slew, decimals=5)
        ftop = round_up_to_raster(abs(area)/amp - ramp, decimals=5)
        amp = area/(ftop+ramp)

    if slewrate is not None:
        system.max_slew = tmp_slew

    if max_grad is not None:
        system.max_grad = tmp_grad

    return amp, ftop, ramp

def calc_triang_wf(amp, ftop, ramp):
    """ Calculate triangular waveform from:
    amp: amplitude [Hz]
    ftop: flat top [s]
    ramp: ramptime [s]
    """

    ramp_wf = amp * np.arange(0.5, int(ramp/10e-6+0.5)) / int(ramp/10e-6+0.5)
    wf = np.concatenate((ramp_wf, amp*np.ones(int(ftop/10e-6+0.5)), ramp_wf[::-1]))
    return wf

def rot_grad(gx, gy, phi):
    """
    rotate gradient with 2D rotation matrix
    This rotation direction matches the direction in the IDEA sequence
    """
    gx_rot = np.cos(phi)*gx+np.sin(phi)*gy
    gy_rot = -np.sin(phi)*gx+np.cos(phi)*gy    
    
    return gx_rot, gy_rot

#############
# Time raster functions
#############

def round_up_to_raster(number, decimals=0):
    """
    round number up to a specific number of decimal places.
    """
    multiplier = 10 ** decimals
    return math.ceil(number * multiplier) / multiplier

def trunc_to_raster(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

#############
# Gradient resonance check
#############

def check_resonances(grads, resonances):
    """ Checks the gradient waveform for forbidden acoustic resonances

    grads: list of gradient waveform on gradient raster (10us), e.g. [grad_x, grad_y]
    resonances: list of tuples describing the resonance bands in [Hz], e.g. [(100,200), (1000,1200)]
    """
    freq_max = []
    warning = False
    for key,grad in enumerate(grads):
        if len(grad)<20000:
            grad = np.concatenate((grad,np.zeros(20000-len(grad)))) # add zeros for higher freq resolution
        grad_ft   = fft(grad, dim=-1)
        freq      = np.arange(-1/(2*dt_grad), 1/(2*dt_grad), 1/(dt_grad*len(grad)))
        argmax    = np.argmax(abs(grad_ft[len(grad)//2:]), axis=-1)
        freq_max.append(freq[len(grad)//2 + argmax]) # peak frequencies from gradient waveform
        for res in resonances:
            if (res[0] <= freq_max[key] <= res[1]):
                warning = True

    if warning:
        raise ValueError('Frequency peak {:.0f} Hz of axis {} is in the forbidden range.'.format(freq_max[key],key))
    else:
        print('Acoustic resonance check succesful.')
    return freq_max
    