# Master thesis

## Code for master thesis on tensor-valued encoding using spiral k-space trajectories

The repository includes the Python-files for the DIVIDE [1] and QTI [2] sequences using Pulseq [3] (with Python's toolbox PyPulseq [4])
and extra code for analysis that is done in the master thesis.

The sequences are based on the diffusion sequence by Veldmann et al. [5] as well as the preprocessing-pipeline (https://github.com/mrphysics-bonn/AxonDiameter). Reconstruction is done with the workflow proposed in [6].

### Structure

- 'sequences':
	- 'write\_divide.py': sequence design according to the DIVIDE approach ('divide\_helper.py' and 'diffusion.py' is needed), adapted from [5]
	- 'write\_qti.py': sequence design according to the QTI approach ('divide\_helper.py' and 'diffusion.py' is needed), adapted from [5]
	- 'NOW': folder containing the gradient waveforms of the three different sequences created with the NOW-toolbox [7] using the script 'scripted\_NOW\_Example.m', adapted from [7]
	- 'qMasOptimization.m': contains the optimization with Matlab (constraints placed in 'nlcon.m')
	- 'integral\_calculator.py': the calculations of the correction factors to consider the finite slew rate of the gradients
	- 'pulses', 'prot.py', 'pulseq\_helper.py': helper-functions for sequence design and pulse design, adopted from [5,6]
	- 'gre\_refscan\_B0.py': GRE prescan for B0 mapping, adopted from [5]
	- 'tve.yml': YAML-file to create conda environment for sequence design
- 'analysis':
	- 'process\_spiral\_Fast.sh': preprocessing of the diffusion-weighted volumes, together with the structural image using 't1_processing.py', adapted from [5]
	- 'dtd_workflow.py': step-by-step instructions on how to create diffusion-tensor-distribution-metric maps using the DIVIDE [1] and QTI [2] approach, including:
		- 'myWorkflow.m': analysis with the DIVIDE approach using the md-dmri toolbox [8]
		- 'segmentation\_pipe\_T1.sh': registration and segmentation of the created metric maps using 'map\_segmentation\_T1.sh'; the metric values for each roi (see folder 'rois') are saved into corresponding dataframes with 'into\_dataframe.py'
		- 'csv_files': folder containing the dataframes belonging to the DIVIDE and QTI comparison ('all'), to the B-tensor calculation comparison ('btens'), to the test-retest measurements ('retest') and to the DTI measurement for comparison of the macroscopic metrics ('dti1')

### Requirements

- for sequence design:
	- conda env create -f sequences/tve.yml --> conda activate tve
	- MATLAB (Optimization toolbox [9] needed)

### References

[1] Lasič, Samo, et al. ‘Microanisotropy imaging: quantification of microscopic diffusion anisotropy and orientational order parameter by diffusion MRI with magic-angle spinning of the q-vector.’ In: Frontiers in Physics 2 (2014): 11.

[2] Westin, Carl-Fredrik, et al. ‘Q-space trajectory imaging for multidimensional diffusion MRI of the human brain.’ Neuroimage 135 (2016): 345-362.

[3] Kelvin J. Layton et al. ‘Pulseq: A rapid and hardware-independent pulse sequence prototyping framework’. In: Magnetic Resonance in Medicine 77.4 (2017), pp. 1544–1552.

[4] Keerthi Sravan Ravi, Sairam Geethanath and John Thomas Vaughan. ‘PyPulseq: A Python Package for MRI Pulse Sequence Design’. In: Journal of Open Source Software 4.42 (2019), p. 1725

[5] Marten Veldmann et al. ‘Spiral readout improves in vivo MR axon radius estimation in human white matter’. In: Proceedings of the International Society for Magnetic Resonance in Medicine Annual Meeting. Vol. 31. Toronto, Canada, 2023, p. 5172.

[6] Marten Veldmann et al. ‘Open-source MR imaging and reconstruction workflow’. In: Magnetic resonance in medicine 88.6 (2022), pp. 2395–2407.

[7] Sjölund J, Szczepankiewicz F, Nilsson M, Topgaard D, Westin C-F, and Knutsson H. ‘Constrained optimization of gradient waveforms for generalized diffusion encoding’. In: Journal of Magnetic Resonance 261 (2015), 157-168.

[8] Markus Nilsson, Filip Szczepankiewicz, Björn Lampinen, André Ahlgren, João P. de Almeida Martins, Samo Lasic, Carl-Fredrik Westin, and Daniel Topgaard. An open-source framework for analysis of multidimensional diffusion MRI data implemented in MATLAB. Proc. Intl. Soc. Mag. Reson. Med. (26), Paris, France, 2018.

[9] MATLAB Optimization Toolbox. The MathWorks, Natick, MA, USA. 2019.
