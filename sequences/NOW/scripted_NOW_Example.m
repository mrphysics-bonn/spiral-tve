% Scripted example of Numerical Optimization of gradient Waveforms (NOW)
clear

% Change the parameters below to your liking. Those not
% specified are default-initialized as follows (see optimizationProblem for details):
%    Max gradient = 80 milliTesla/m
%    Max slew rate = 100 milliTesla/m/milliSecond = 100 T/m/s
%    Eta (heat dissipation parameter) = 1
%    Discretization points = 77
%    Target tensor = eye(3)
%    Initialguess = 'random'
%    zeroGradientAtIndex = [], i.e. only at start and end
%    enforceSymmetry = false;
%    redoIfFailed = true;
%    useMaxNorm = false;
%    doMaxwellComp = true;
%    MaxwellIndex = 100;
%    Motion compensation: disabled
%    Background compensation: disabled
%
% Written by Jens Sj�lund and Filip Szczepankiewicz


%%  PREP
% First, set up the optimization problem. Do this first to create a
% structure where fields are pre-specified. Note that some fields are
% read-only and that new fields cannot be created.
problem = optimizationProblem;

% Define the hardware specifications of the gradient system
problem.gMax =  67; % Maximal gradient amplitude, in [mT/m]
problem.sMax = 130; % Maximal gradient slew (per axis), in [T/(sm)]

% Request encoding and pause times based on sequence timing in [ms]
problem.durationFirstPartRequested    = 44.83;
problem.durationSecondPartRequested   = 48.56;
problem.durationZeroGradientRequested = 10.68;

% Define the b-tensor shape in arbitrary units. This example uses an
% isotropic b-tensor that results in spherical tensor encoding (STE).
problem.targetTensor = eye(3);
%problem.targetTensor = zeros(3,3);
%problem.targetTensor(1,1) = 3;

% Define the number of sample points in time. More points take longer to
% optimize but provide a smoother waveform that can have steeper slopes.
problem.N = 150;

% Set the balance between energy consumption and efficacy
problem.eta = 0.9; %In interval (0,1]

% Set the threshold for concomitant gradients (Maxwell terms). 
% Please see https://doi.org/10.1002/mrm.27828 for more information on how 
% to set this parameter.
problem.MaxwellIndex = 100; %In units of (mT/m)^2 ms
%problem.doMaxwellComp = false;

% Set the desired orders of motion compensation and corresponding
% thresholds for allowed deviations. See Szczepankiewicz et al., MRM, 2020
% for details. maxMagnitude in units s^order / m.
% problem.motionCompensation.order = [1, 2];
% problem.motionCompensation.maxMagnitude = [0, 1e-4];
problem.motionCompensation.order = [1];
problem.motionCompensation.maxMagnitude = [0];

% Toggle compensation for background gradients
%problem.doBackgroundCompensation = true;

% Make a new optimizationProblem object using the updated specifications.
% This explicit call is necessary to update all private variables.
problem = optimizationProblem(problem);


%% PRINT REQUESTED AND TRUE TIMES
% Note that due to the coarse raster, the requested and actual times may
% differ slightly.
clc
now_print_requested_and_real_times(problem);


%% RUN OPTIMIZATION
%parpool
[result, problem] = NOW_RUN(problem);
%delete(gcp('nocreate'))

%% PLOT RESULT AND SAVE RESULT
figure(1)
plot(0:problem.dt:problem.totalTimeActual,result.gwf)
xlabel('Time [ms]')
ylabel('Gradient amplitude [T/m] (not effective)')
measurementTensor = result.B
b = result.b

timeTotal = round((0:problem.dt:problem.totalTimeActual),4);

timeFirst = round((0:problem.dt:problem.durationFirstPartActual),4);
gwfFirst = result.gwf(find(timeTotal<=problem.durationFirstPartActual),1:end);

timeZero = round((0:problem.dt:problem.durationZeroGradientActual),4);
gwfZero = result.gwf(find(timeTotal>=problem.durationFirstPartActual & timeTotal<=problem.durationFirstPartActual+problem.durationZeroGradientActual),1:end);

timeSecond = round((0:problem.dt:problem.durationSecondPartActual),4);
gwfSecond = result.gwf(find(timeTotal>=problem.durationFirstPartActual+problem.durationZeroGradientActual & timeTotal<=problem.durationFirstPartActual+problem.durationZeroGradientActual+problem.durationSecondPartActual),1:end);

figure(2)
plot(timeFirst,gwfFirst)

figure(3)
plot(timeZero,gwfZero)

figure(4)
plot(timeSecond,gwfSecond)

NowResults.timeFirst = timeFirst;
NowResults.gwfFirst = gwfFirst;
NowResults.timeZero = timeZero;
NowResults.gwfZero = gwfZero;
NowResults.timeSecond = timeSecond;
NowResults.gwfSecond = gwfSecond;

save('NowResults.mat', 'NowResults');
