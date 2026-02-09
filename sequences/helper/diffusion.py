from helper.pulseq_helper import round_up_to_raster
import sympy as sym
import mpmath
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class diff_params():
    """
    class to set and calculate diffusion gradient parameters assuming rectangular shape
    """
    def __init__(self, b_val = None, delta = None, spacing = None):
        self.b_val = b_val
        self.delta = delta
        self.spacing = spacing

    def set_param(self, b_val = None, delta = None, spacing = None):
        if b_val:
            self.b_val=b_val # [s/mm^2]
        if delta:
            self.delta=delta # [s]
        if spacing:
            self.spacing=spacing # [s]

    def calc_params(self, system, encoding_duration=None):
        """ calculate parameter from other two given parameters
            it is assumed that the trapezoid is symmetric
            unit of system parameters is [Hz/m] - contain gamma/2*pi already
        """
        rise_time = round_up_to_raster(system.max_grad/system.max_slew, decimals=5) # maximum gradient strength and slewrate is always used
        if self.spacing is None and encoding_duration is None:
            try:
                self.spacing = (self.b_val*1e6/(2*np.pi*system.max_grad)**2-rise_time**3/30+self.delta*rise_time**2/6)/self.delta**2+self.delta/3 # [s]
                self.spacing = round(self.spacing, 5) # round to gradient raster time
                print('Spacing of diffusion gradients: '+'{:.2f}'.format(self.spacing*1e3)+' ms')
            except:
                raise ValueError('Select 2 of 3 diffusion parameters')
        elif self.delta is None and encoding_duration is None:
            try:
                # calculate roots of polynomial
                p = np.array([-((2*np.pi*system.max_grad)**2)/3, ((2*np.pi*system.max_grad)**2)*self.spacing, -1*(2*np.pi*system.max_grad)**2*rise_time**2/6, -1*self.b_val*1e6+(2*np.pi*system.max_grad)**2*rise_time**3/30])
                roots = np.roots(p)
                # check which root to choose by comparing with targeted b-value
                diff = self.b_val - 1e-6*(2*np.pi*system.max_grad)**2*(roots**2*(self.spacing-roots/3)+rise_time**3/30-roots*rise_time**2/6) 
                self.delta = roots[np.argmin(diff)]
                print('Duration of diffusion gradients: '+'{:.2f}'.format(self.delta*1e3)+' ms')
            except:
                raise ValueError('Select 2 of 3 diffusion parameters')
        elif self.b_val is None and encoding_duration is None:
            try:
                self.b_val = 1e-6*(2*np.pi*system.max_grad)**2*(self.delta**2*(self.spacing-self.delta/3)+rise_time**3/30-self.delta*rise_time**2/6)
                print('b-value: '+'{:.2f}'.format(self.b_val))
            except:
                raise ValueError('Select 2 of 3 diffusion parameters')
        elif encoding_duration is not None:
            delta, spacing = sym.symbols("delta Delta")
            mpmath.mp.dps = 15
            solution = sym.nsolve((encoding_duration-delta-spacing-rise_time,(2*np.pi*system.max_grad)**2*(delta**2*(spacing-delta/3)+rise_time**3/30-delta*rise_time**2/6)-self.b_val*1e6),(delta,spacing),(9*1e-3,70*1e-3),solver='anderson')
            self.delta = round(float(solution[0]), 5)
            self.spacing = round(float(solution[1]), 5)
        else:
            raise ValueError('Select only 2 of 3 diffusion parameters.')

def calc_bval(grad,spacing):
    """calculates the b-value of a gradient pair
    
    grad: one gradient of the equal gradient pair
    spacing: spacing of the gradients    
    """
    delta = grad.rise_time + grad.flat_time
    return 1e-6*(2*np.pi*grad.amplitude)**2*(delta**2*(spacing-delta/3)+grad.rise_time**3/30-delta*grad.rise_time**2/6)


""" Discoball
Generates a DISCS direction scheme with latitudes starting at the
northpole (DISCSo = "odd") or not at the northpole (DISCSe = 'even).

Input args: 
            N: Requested number of isotropically distributed directions
               in default mode.

            bNisNtheta: (optional) boolean for either
                        - False (default): DISCS scheme with N directions or
                        - True: DISCS scheme with n=N latitudes. The final
                        number of directions corresponds to the optimal
                        configuration with n latitudes.

Output args: s: N x 3 matrix with Cartesian direction coordinates on unit
                sphere where N is the number of directions.


Copyright (C) 2013 Ruediger Stirnberg

Original Matlab code transferred to Python by Marten Veldmann
"""

def DISCSo(N, bNisNtheta=False):

    # geometry factor
    alpha = np.deg2rad(30)
    epsilon = 1/np.cos(alpha/2) - 1

    if bNisNtheta:
        n0 = N
        k_ = np.zeros(int(np.floor(n0*0.5)))
        n_ = n0
        for i,_ in enumerate(k_):
            if i+1==n0*0.5:
                k_[i] = round(0.5*np.pi/np.arcsin(np.sin(np.pi/n0/2)/np.sin((i+1)*np.pi/n0)))
            else:
                k_[i] = round(np.pi/np.arcsin(np.sin(np.pi/n0/2)/np.sin((i+1)*np.pi/n0)))
        N = int(sum(k_))+1
    elif N==1:
        n_ = 0
    else:
        n0 = np.pi / (2* np.arccos(-np.pi/(4*(1+epsilon)*(N-1)) + np.sqrt((np.pi/(4*(1+epsilon)*(N-1)))**2+1)))
        n0 = [int(np.floor(n0)), int(np.ceil(n0))]
        if 1 in n0:
            n0.remove(1)
    
        # init. final parameters
        m_ = np.inf             # best mean squared distance error    (1 x 1)
        n_ = None               # best number of latitudinal circles  (1 x 1)
        k_ = np.array([])       # best population of lat. circles     (n_ x 1)

        for n in n0:
            # n/2 (max. lat. circle in upper hemisphere)
            nhalf = int(np.floor(n*0.5))

            # mean n.n. distance
            d0 = 2*(1+epsilon)*np.sin(np.pi/2/n)
            # reasonable population range per latitude
            k = np.stack(nhalf * [np.arange(4*n)+1])
            # latitude index matrix of same size
            i = np.stack(k.shape[1] * [np.arange(nhalf)+1]).T
            # corresponding distances
            d = 2*np.sin(i*np.pi/n)*np.sin(np.pi/k)
            if not n%2: # equator            
                d[-1,:] = 2*np.sin(i[-1,:]*np.pi/n)*np.sin(np.pi/k[-1,:]/2)

            # corresponding squared errors
            e = (d0-d)**2
            ind = np.argsort(e, axis=1)
            e = np.sort(e, axis=1)
            for i in range(nhalf):
                k[i,:] = k[i,ind[i,:]]

            # optimal parameters for given n(l):
            E = e[:,0]
            K = k[:,0]
            N0 = 1+sum(K)

            # optimal population for given n -> best population for given N:
            i2=np.array([]); e2=np.array([]); k2=np.array([])
            Ndiff = N0-N
            if Ndiff!=0:
                for i in range(nhalf):
                    if Ndiff>0:
                        # indices available for population reduction
                        tmpj=np.where(k[i,:]<K[i])[0]
                    else:
                        # indices available for population increase
                        tmpj=np.where(k[i,:]>K[i])[0]

                    i2 = np.concatenate((i2,i*np.ones(tmpj.shape)), axis=0)
                    e2 = np.concatenate((e2,e[i,tmpj]), axis=0)
                    k2 = np.concatenate((k2,k[i,tmpj]), axis=0)       

                # change by Ndiff in total on those latitudes where error is least
                ind = np.argsort(e2)
                e2 = np.sort(e2)
                k2 = k2[ind]
                i2 = i2[ind]

                # update best parameters for given n and N
                E[np.int64(i2[:int(abs(Ndiff))])] = e2[:int(abs(Ndiff))]
                K[np.int64(i2[:int(abs(Ndiff))])] = k2[:int(abs(Ndiff))]

            # mean squared error
            m = sum(E*(K-1))/N

            # update final parameters for given N
            if m<m_:
                m_ = m
                n_ = n
                k_ = K.copy()
            
    # compute coordinates
    s = np.zeros([N,3]) # coordinate array
    s[0,:] = [0,0,1]
    o = 0
    for i in range(int(np.floor(n_*0.5))):
        j = (np.arange(k_[i])+1).T

        theta = (i+1)*np.pi/n_
        if i+1<n_*0.5:
            phi = (i+1+j)*2*np.pi/k_[i]
        else: # equator
            phi = (i+1+j)*np.pi/k_[i]

        st = np.sin(theta); ct = np.cos(theta)
        sp = np.sin(phi);   cp = np.cos(phi)
        s[o+1:o+int(k_[i])+1,0] = st*cp
        s[o+1:o+int(k_[i])+1,1] = st*sp
        s[o+1:o+int(k_[i])+1,2] = ct
        o += int(k_[i])

    return s


def DISCSe(N, bNisNtheta=False):

    # geometry factor
    alpha = np.deg2rad(30)
    epsilon = 1/np.cos(alpha/2) - 1

    # ideal number of latitudinal circles over entire sphere
    if bNisNtheta:
        n0 = N
        k_ = np.zeros(int(np.floor(n0*0.5+0.5)))
        n_ = n0
        for i,_ in enumerate(k_):
            if i+0.5<n0*0.5:
                k_[i] = round(np.pi/np.arcsin(np.sin(np.pi/n0/2)/np.sin((i+0.5)*np.pi/n0)))
            else:
                k_[i] = round(0.5*np.pi/np.arcsin(np.sin(np.pi/n0/2)/np.sin((i+0.5)*np.pi/n0)))
        N = int(sum(k_))
    elif N==1:
        n_ = 1
        k_ = np.array([1])
    else:
        n0 = np.pi / (2* np.arccos(-np.pi/(4*(1+epsilon)*(N-1)) + np.sqrt((np.pi/(4*(1+epsilon)*(N-1)))**2+1)))
        n0 = [int(np.floor(n0)), int(np.ceil(n0))]
    
        # init. final parameters
        m_ = np.inf             # best mean squared distance error    (1 x 1)
        n_ = None               # best number of latitudinal circles  (1 x 1)
        k_ = np.array([])       # best population of lat. circles     (n_ x 1)

        for n in n0:
            # n/2 (max. lat. circle in upper hemisphere)
            nhalf = int(np.floor(n*0.5+0.5))

            # mean n.n. distance
            d0 = 2*(1+epsilon)*np.sin(np.pi/2/n)
            # reasonable population range per latitude
            k = np.stack(nhalf * [np.arange(4*n-1)+2])
            # latitude index matrix of same size
            i = np.stack(k.shape[1] * [np.arange(nhalf)+1]).T
            # corresponding distances
            d = 2*np.sin((i-0.5)*np.pi/n)*np.sin(np.pi/k)
            if not (n-1)%2: # equator            
                d[-1,:] = 2*np.sin(i[-1,:]*np.pi/n)*np.sin(np.pi/k[-1,:]/2)

            # corresponding squared errors
            e = (d0-d)**2
            ind = np.argsort(e, axis=1)
            e = np.sort(e, axis=1)
            for i in range(nhalf):
                k[i,:] = k[i,ind[i,:]]

            # optimal parameters for given n(l):
            E = e[:,0]
            K = k[:,0]
            N0 = sum(K)

            # optimal population for given n -> best population for given N:
            i2=np.array([]); e2=np.array([]); k2=np.array([])
            Ndiff = N0-N
            if Ndiff!=0:
                for i in range(nhalf):
                    if Ndiff>0:
                        # indices available for population reduction
                        tmpj=np.where(k[i,:]<K[i])[0]
                    else:
                        # indices available for population increase
                        tmpj=np.where(k[i,:]>K[i])[0]

                    i2 = np.concatenate((i2,i*np.ones(tmpj.shape)), axis=0)
                    e2 = np.concatenate((e2,e[i,tmpj]), axis=0)
                    k2 = np.concatenate((k2,k[i,tmpj]), axis=0)       

                # change by Ndiff in total on those latitudes where error is least
                ind = np.argsort(e2)
                e2 = np.sort(e2)
                k2 = k2[ind]
                i2 = i2[ind]

                # update best parameters for given n and N
                E[np.int64(i2[:int(abs(Ndiff))])] = e2[:int(abs(Ndiff))]
                K[np.int64(i2[:int(abs(Ndiff))])] = k2[:int(abs(Ndiff))]

            # mean squared error
            m = sum(E*(K-1))/N

            # update final parameters for given N
            if m<m_:
                m_ = m
                n_ = n
                k_ = K.copy()
            
    # compute coordinates
    s = np.zeros([N,3]) # coordinate array
    o = -1
    for i in range(int(np.floor(n_*0.5+0.5))):
        j = (np.arange(k_[i])+1).T

        theta = (i+0.5)*np.pi/n_
        if i+1<(n_+1)*0.5:
            phi = (i+0.5+j)*2*np.pi/k_[i]
        else: # equator
            phi = (i+0.5+j)*np.pi/k_[i]

        st = np.sin(theta); ct = np.cos(theta)
        sp = np.sin(phi);   cp = np.cos(phi)
        s[o+1:o+int(k_[i])+1,0] = st*cp
        s[o+1:o+int(k_[i])+1,1] = st*sp
        s[o+1:o+int(k_[i])+1,2] = ct
        o += int(k_[i])

    return s


def computeEncodingSpectrum(gradient_wf, grad_raster_time, gamma,df=None,saveplots=None):
    ''' 
    Returns the encoding spectrum b(w) (aka s(w)) of the given gradient waveform

    Input:
    - gradient_wf: effective gradient waveform [time,axis] [T/m]
    - grad_raster_time: gradient raster time [s]
    - gamma: gyromagnetic ratio [rad/(s*T)]
    - df: if set, the gradient waveform will be interpolated in such a way, that the spacing of the frequency axis is df (default:None)
    - saveplots: path to save plots; if None is given, no plots will be saved

    Returns:
    - freq_axis: frequency axis with the angular frequency w=2*pi*f [freq] [1/s]
    - b_w_cmplx: complex encoding spectrum (freq,3,3) [s²/m²]
    - b_trace_w: real trace of the encoding spectrum (freq,) [s²/m²]
    - b_fromSum: b-value (sum over all b(w)) [s/m²]
    - w_centroid: centroid ('mean') frequency of the encoding spectrum [rad/s]
    '''
    # zero-filling in time domain
    td_before_zp = grad_raster_time*gradient_wf.shape[0] # [s]
    if df:
        td=1/df
        zero_filling_factor = round(td/(grad_raster_time*gradient_wf.shape[0]))
        gradient_wf = np.pad(gradient_wf, ((0, int(zero_filling_factor*gradient_wf.shape[0])), (0, 0)), mode='constant')
    else:
        gradient_wf = np.pad(gradient_wf, ((0, int(0*gradient_wf.shape[0])), (0, 0)), mode='constant')
    # define time vector for the gradient waveform
    t_grad = (np.arange(gradient_wf.shape[0]) + 0.5) * grad_raster_time # [s]

    # compute dephasing waveforms vec(q(t))
    q_t = np.zeros_like(gradient_wf)
    for i in range(gradient_wf.shape[1]):
        q_t[:,i] = gamma * cumulative_trapezoid(gradient_wf[:,i], t_grad,initial=0) # [rad/m]
    
    # compute dephasing spectrum vec(q(w))
    dt = grad_raster_time # [s]
    freq_axis = np.fft.fftshift(np.fft.fftfreq(gradient_wf.shape[0], d=dt)) * 2*np.pi # np.linspace(-fa/2, fa/2, gradient_wf.shape[0]) * 2*np.pi # [rad/s]

    q_w = np.zeros_like(q_t,dtype=complex)
    for i in range(q_t.shape[1]):
        q_w[:,i] = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(q_t[:,i],axes=0),axis=0),axes=0)*dt
    
    # compute real and imaginary parts of the encoding spectrum tensor(b(w))
    b_w_cmplx = np.zeros((q_w.shape[0],3,3),dtype=complex)
    b_w_real = np.zeros((q_w.shape[0],3,3))
    b_w_imag = b_w_real
    for i in range(q_w.shape[0]):
        b_w_cmplx[i,:,:] = 1/(2*np.pi) * np.outer(q_w[i,:],q_w[i,:].conj())
        b_w_real[i,:,:] = np.real(1/(2*np.pi) * np.outer(q_w[i,:],q_w[i,:].conj()))
        b_w_imag[i,:,:] = np.imag(1/(2*np.pi) * np.outer(q_w[i,:],q_w[i,:].conj()))
    
    min_freq_idx=np.argmin(abs(freq_axis))
    min_freq=np.abs(freq_axis)[min_freq_idx]

    # compute spectral trace b(w)
    b_trace_w = np.zeros((b_w_cmplx.shape[0]))
    for i in range(b_w_cmplx.shape[0]):
        b_trace_w[i] = np.real(np.trace(b_w_cmplx[i,:,:])) # [s²/m²]
    
    # compute b tensor for evaluation purposes
    b_fromSum = np.trapz(b_trace_w[:], freq_axis) # [s/m²]

    # compute centroid frequency (Jiang et al., Magn. Reson., 2023)
    w_centroid = 1/b_fromSum * np.trapz(b_trace_w[:]*np.abs(freq_axis), freq_axis) # [rad/s]

    # compute qv(t)
    q_v = np.zeros_like(q_t)
    for i in range(q_t.shape[1]):
        q_v[:,i] = gamma * np.cumsum(q_t[:,i])*dt

    # do plotting of ...
    fig,axes = plt.subplots(6,1,figsize=[30,60])
    labels = ['x','y','z']
    
    # ... gradient waveform
    axes[0].plot(t_grad*1e3,gradient_wf*1e3,label=labels,lw=10)
    axes[0].set_ylabel(r"$\vec{g}(t)$[mT/m]", fontsize=70)
    axes[0].set_xlabel("t [ms]", fontsize=70)
    axes[0].tick_params(axis='both', labelsize=60)
    axes[0].grid('on')
    axes[0].set_xlim([0,td_before_zp*1e3])
    #axes[0].set_ylim([-80,80])
    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=60)

    # ... q-vector waveform
    axes[1].plot(t_grad*1e3,q_t,label=labels,lw=10)
    axes[1].set_ylabel(r"$\vec{q}(t)$[1/m]", fontsize=70)
    axes[1].set_xlabel("t [ms]", fontsize=70)
    axes[1].tick_params(axis='both', labelsize=60)
    axes[1].grid('on')
    axes[1].set_xlim([0,td_before_zp*1e3])
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=60)
    axes[1].yaxis.get_offset_text().set_size(60)

    # ... trace(b(w)) and centroid frequency
    axes[2].plot(freq_axis/(2*np.pi),b_trace_w[:]/1e6, lw=10,label=r"b($\omega$)/1e6",color='k')
    axes[2].plot([w_centroid/(2*np.pi),w_centroid/(2*np.pi)],[0,(b_trace_w[:]/1e6).max()],lw=10,label=r"$\omega_\mathrm{centr}/(2\pi)$="+str(np.round(w_centroid/(2*np.pi),0))+" 1/s",color='m')
    axes[2].set_ylabel(r"b($\omega$)/1e6", fontsize=70)
    axes[2].set_xlabel(r"$\omega/(2\pi)$ [1/s]", fontsize=70)
    axes[2].tick_params(axis='both', labelsize=60)
    axes[2].grid('on')
    axes[2].set_xlim([-100,100])
    axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=60)

    # ... real components of b(w) (imaginary components vanish in the diffusion signal attenuation beta)
    axes[3].plot(freq_axis/(2*np.pi),np.real(b_w_cmplx[:,0,0])/1e6,label='xx',lw=10)
    axes[3].plot(freq_axis/(2*np.pi),np.real(b_w_cmplx[:,1,1])/1e6,label='yy',lw=10)
    axes[3].plot(freq_axis/(2*np.pi),np.real(b_w_cmplx[:,2,2])/1e6,label='zz',lw=10)
    axes[3].plot(freq_axis/(2*np.pi),np.real(b_w_cmplx[:,0,1])/1e6,label='xy',lw=6)
    axes[3].plot(freq_axis/(2*np.pi),np.real(b_w_cmplx[:,0,2])/1e6,label='xz',lw=6)
    axes[3].plot(freq_axis/(2*np.pi),np.real(b_w_cmplx[:,1,2])/1e6,label='yz',lw=6)
    axes[3].set_ylabel(r"real $\bf{b}$($\omega$)/1e6", fontsize=70)
    axes[3].set_xlabel(r"$\omega/(2\pi)$ [1/s]", fontsize=70)
    axes[3].tick_params(axis='both', labelsize=60)
    axes[3].grid('on')
    axes[3].set_xlim([-100,100])
    axes[3].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=60)

    # ... imaginary components of b(w) (these components vanish in the diffusion signal attenuation beta)
    axes[4].plot(freq_axis/(2*np.pi),np.imag(b_w_cmplx[:,0,0])/1e6,label='xx',lw=10)
    axes[4].plot(freq_axis/(2*np.pi),np.imag(b_w_cmplx[:,1,1])/1e6,label='yy',lw=10)
    axes[4].plot(freq_axis/(2*np.pi),np.imag(b_w_cmplx[:,2,2])/1e6,label='zz',lw=10)
    axes[4].plot(freq_axis/(2*np.pi),np.imag(b_w_cmplx[:,0,1])/1e6,label='xy',lw=6)
    axes[4].plot(freq_axis/(2*np.pi),np.imag(b_w_cmplx[:,0,2])/1e6,label='xz',lw=6)
    axes[4].plot(freq_axis/(2*np.pi),np.imag(b_w_cmplx[:,1,2])/1e6,label='yz',lw=6)
    axes[4].set_ylabel(r"imag $\bf{b}$($\omega$)/1e6", fontsize=70)
    axes[4].set_xlabel(r"$\omega/(2\pi)$ [1/s]", fontsize=70)
    axes[4].tick_params(axis='both', labelsize=60)
    axes[4].grid('on')
    axes[4].set_xlim([-100,100])
    axes[4].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=60)

    axes[5].plot(t_grad*1e3,q_v,label=labels,lw=10)
    axes[5].set_ylabel(r"$\vec{qv}(t)$[1/m]", fontsize=70)
    axes[5].set_xlabel("t [ms]", fontsize=70)
    axes[5].tick_params(axis='both', labelsize=60)
    axes[5].grid('on')
    axes[5].set_xlim([0,td_before_zp*1e3])
    axes[5].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=60)
    axes[5].yaxis.get_offset_text().set_size(60)

    fig.suptitle(r'Spectral encoding of $\vec{g}(t)$', fontsize=80)
    fig.tight_layout(rect=[0, 0, 1, 0.97])  # Reserve space for the suptitle

    if not saveplots is None:
        plt.savefig(saveplots,dpi=300)
    else:
        plt.show()

    return freq_axis, b_w_cmplx, b_trace_w, b_fromSum, w_centroid

def halveProtocol(b_val,bDelta,bval_list,bDelta_list,dir_list):
    
    def getIndices(bdelta,bval):
        # Fixiere den Seed – gleiche Zufallszahlen bei jedem Lauf
        np.random.seed(42)

        # extract PTE
        mask = (bDelta_list==bdelta) & (bval_list==bval)
        dir_list_pte_nob0 = dir_list[mask]
        indices = np.where(mask)[0]

        # normalize
        num_points = dir_list_pte_nob0.shape[0]
        vecs = dir_list_pte_nob0.copy()
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        # === 2. Farthest Point Sampling (FPS) Funktion ===
        def farthest_point_sampling_on_sphere(vectors, k, start_idx=0):
            n = vectors.shape[0]
            selected_indices = []

            # Stelle sicher, dass alle Vektoren normiert sind
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

            # Starte mit festem Index
            selected_indices.append(start_idx)
            dists = np.full(n, np.inf)

            for _ in range(1, k):
                last_selected = vectors[selected_indices[-1]]
                cos_angles = vectors @ last_selected
                angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))
                dists = np.minimum(dists, angles)
                next_idx = np.argmax(dists)
                selected_indices.append(next_idx)

            return selected_indices

        # === 3. Wähle 18 gleichmäßig verteilte Punkte (mit festem Startindex) ===
        selected_indices = farthest_point_sampling_on_sphere(vecs, int(num_points/2), start_idx=0)
        selected_vectors = vecs[selected_indices]

        # === 4. 3D-Plot der Vektoren ===
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Alle Punkte (grau)
        ax.scatter(vecs[:, 0], vecs[:, 1], vecs[:, 2], color='gray', alpha=0.4, label='All directions')

        # Ausgewählte Punkte (rot)
        ax.scatter(selected_vectors[:, 0], selected_vectors[:, 1], selected_vectors[:, 2],
                color='red', s=80, label='Selected directions')

        # Linien von Ursprung zu ausgewählten Vektoren
        for vec in selected_vectors:
            ax.plot([0, vec[0]], [0, vec[1]], [0, vec[2]], color='red', alpha=0.3)

        # Einstellungen
        ax.set_title(f'Farthest Point Sampling on a sphere for bdelta={bdelta} and bval={bval}mm²/s', fontsize=14)
        ax.legend()
        ax.set_box_aspect([1, 1, 1])
        plt.show()

        # get indices for PTE:
        indices_all = sorted(indices[selected_indices])
        return indices_all

    lists_nob0_alone = []
    lists_b0_alone = []
    for enc,bDelta_elem in enumerate(bDelta):
        for b_val_elem in b_val[enc]:
            if b_val_elem == 0:
                mask = (bDelta_list==bDelta_elem) & (bval_list==0)
                indices = np.where(mask)[0]
                lists_b0_alone.append([indices[idx] for idx in range(0,len(indices),2)])
            else:
                lists_nob0_alone.append(getIndices(bDelta_elem,b_val_elem))
    
    lists_b0_alone_flat = [item for sublist in lists_b0_alone for item in sublist]
    lists_nob0_alone_flat = [item for sublist in lists_nob0_alone for item in sublist]    
    all_list = sorted(lists_b0_alone_flat+lists_nob0_alone_flat)

    final_list=np.zeros(len(bval_list))
    final_list[all_list] = 1
    
    return final_list.tolist()
