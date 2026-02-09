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
	- MATLAB (Optimization toolbox [9] needed)

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
