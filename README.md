# Tensor-valued encoding using spiral k-space trajectories

Sequence design is done with Pulseq [1] (with Python's toolbox PyPulseq [2]) and focusses on the q-space trajectory imaging approach [3] with protocols from Ref. [4].
Waveforms are based on the double-rotation gradient waveform (gDOR) [5] as well as on magic-angle spinning of the q-vector (qMAS) [6].

The sequences are based on the diffusion sequence by Veldmann et al. [7] (https://github.com/mrphysics-bonn/AxonDiameter). Reconstruction is done with the workflow proposed in [8] (https://github.com/mrphysics-bonn/python-ismrmrd-reco).

## Structure

- 'sequences':
	- 'write\_tve.py': sequence design according to the TVE approach, adapted from [7]
	- 'qMasOptimization.m': contains the qMAS optimization with Matlab
	- 'nlcon.m': constraints for the qMAS optimization with Matlab
	- 'gre\_refscan\_B0.py': GRE prescan for B0 mapping, adopted from [7]
	- 'helper': folder containing helper-functions for sequence design
	- 'tve.yml': YAML-file to create conda environment for sequence design
	- 'qti_waveforms':
		- 'from_optimization_to_seq': folder to store designed diffusion gradients (here: waveforms based on gDOR [5])

## Requirements

- for sequence design:
	- conda env create -f sequences/tve.yml --> conda activate tve
	- MATLAB (Optimization toolbox [9] needed) --> code was tested with MATLAB version R2018b

## Usage

- Clone the repository and install the requirements.
- To create a Pulseq file for your sequence, open the Python script 'write\_tve.py' in your preferred browser and change the sequence parameters to meet your desired sequence.
- After execution of the Python script, the following outputs will be created:
	- A sequence file (.seq) and the protocol file for reconstruction [8] (.h5) will be created in the folder, in which the Python script is located.
	- A numpy file, containing the b-tensor for each diffusion volume, will be created in the folder 'qti_optimization/btens'. The folder will be created if it does not exist.
	- If 'plotSpectrum=True' Figures showing the gradient waveforms and the q-space trajectories will be created.
	- If 'testCrushingMoments=True' Figures showing the spoiling moment for each diffusion volume before the 180° pulse are created to ensure a spoiling moment of two k-spaces.

## MATLAB optimization

The selection of the qMAS sequence ('diffSeq=0') invokes a call to a MATLAB function 'qMASOptimization.m', which uses MATLAB's local optimization tool fmincon to create the qMAS gradients for a minimal echo time. The optimization uses the script 'nlcon.m' internally to define nonlinear-constraints to not overstep gradient scanner limits.

The MATLAB function 'qMASOptimization.m is called within the Python script with the module 'subprocess'.

To test if the optimization works as intended, it it recommended to test the optimization separately within the command-window of the MATLAB-interface. An example call to the function with the inputs for the qMAS sequence from the in vivo measurements of the paper is as follows:

`[min_spacing, x_solution] = qMasOptimization(b_val_max=1400, refoc_dur=10.48, includePTE=1, maxgrad=70e-3, maxslew=190)`

The optimization should finish after approximately one minute with the following outputs: 
[0.0771, 2468315.3445479, 7604073656.6792, 0.0088600554968332]
- min\_spacing = 0.0771 [s] minimum spacing between the trapezoidal gradients of the qMAS waveform
- system\_max\_grad = 2468315.3445479; % [Hz/m] (gradient amplitudes [T/m] multiplied by gamma = 42576000 [Hz/T]) maximum Euclidean gradient amplitude of the qMAS waveform
- system\_max\_slew = 7604073656.6792; % [1/ms²] (gradient slew rate [T/m/s] multiplied by gamma = 42576000 [Hz/T]) maximum slew rate of the qMAS waveform
- delta = 0.0088600554968332; [s] duration of the trapezoidal gradient of the qMAS waveform

## References

[1] Layton KJ, Kroboth S, Jia F, et al. Pulseq: A rapid and hardware-independent pulse sequence prototyping framework: Rapid Hardware-Independent Pulse Sequence Prototyping. Magn Reson Med. 2017;77(4):1544-1552. doi:10.1002/mrm.26235

[2] Keerthi Sravan Ravi, Sairam Geethanath and John Thomas Vaughan. ‘PyPulseq: A Python Package for MRI Pulse Sequence Design’. In: Journal of Open Source Software 4.42 (2019), p. 1725

[3] Westin CF, Knutsson H, Pasternak O, et al. Q-space trajectory imaging for multidimensional diffusion MRI of the human brain. NeuroImage. 2016;135:345-362. doi:10.1016/j.neuroimage.2016.02.039

[4] Morez J, Szczepankiewicz F, Den Dekker AJ, Vanhevel F, Sijbers J, Jeurissen B. Optimal experimental design and estimation for q‐space trajectory imaging. Human Brain Mapping. 2023;44(4):1793-1809. doi:10.1002/hbm.26175

[5] Jiang H, Svenningsson L, Topgaard D. Multidimensional encoding of restricted and anisotropic diffusion by double rotation of the q vector. Magn Reson. 2023;4(1):73-85. doi:10.5194/mr-4-73-2023

[6] Eriksson S, Lasic S, Topgaard D. Isotropic diffusion weighting in PGSE NMR by magic-angle spinning of the q-vector. Journal of Magnetic Resonance. 2013;226:13-18. doi:10.1016/j.jmr.2012.10.015

[7] Veldmann M,  Edwards LJ, Pine KJ, et al. Improving MR axon radius estimation in human white matter using spiral acquisition and field monitoring. Magn Reson Med. 2024;1-15. doi: 10.1002/mrm.30180

[8] Veldmann M, Ehses P, Chow K, Nielsen J, Zaitsev M, Stöcker T. OPEN‐SOURCE MR imaging and reconstruction workflow. Magnetic Resonance in Med. 2022;88(6):2395-2407. doi:10.1002/mrm.29384

[9] MATLAB Optimization Toolbox. The MathWorks, Natick, MA, USA. 2019.
