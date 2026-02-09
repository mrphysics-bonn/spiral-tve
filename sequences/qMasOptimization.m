function [min_spacing, x_solution] = qMasOptimization(b_val_max, refoc_dur, includePTE, maxgrad, maxslew)
    % local optimization with fmincon (using multiple starting points to make it 'global')
    % input parameters: b_val_max [s/mm²]; refoc_dur [s]; includePTE (bool)

    % constant
    gamma = 42576000; % Hz/T

    % define objective function and constraints
    objective = @(x) round((b_val_max*1e6/(2*pi*x(1))^2-round(x(1)/x(2),5)^3/30+x(3)*round(x(1)/x(2),5)^2/6)/x(3)^2+x(3)/3, 5);
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [0,0,0];
    ub = [maxgrad*gamma, maxslew*gamma, 0.1];
    nonlincon = @(x) nlcon(x, b_val_max, refoc_dur, includePTE, maxgrad, maxslew);

    % loop parameters
    delta_x0 = [0.01*1e-3 : 0.01*1e-3 : 20*1e-3]; % s
    min_spacing = 1e10; % s
    x_solution = zeros(1,3); % [Hz/m, Hz/(ms), s]

    % do global minimization
    for i = 1:size(delta_x0,2)

        x0 = [0.97*maxgrad*gamma, 0.94*maxslew*gamma, delta_x0(i)];

        try 
            x = fmincon(objective, x0, A, b, Aeq, beq, lb, ub, nonlincon, optimoptions('fmincon','Algorithm','interior-point','TolX',1e-30,'TolFun', 1e-30, 'MaxFunctionEvaluations',1e4));
            disp(['Initial Objective: ' num2str(objective(x0))])
            disp(['Final Objective: ' num2str(objective(x))])
        catch
            continue
        end

        if objective(x) < min_spacing && all(nlcon(x, b_val_max, refoc_dur, includePTE, maxgrad, maxslew)<0)
            min_spacing = objective(x);
            x_solution(1) = x(1);
            x_solution(2) = x(2);
            x_solution(3) = x(3);
        end

    end

    disp(['min_spacing: ' num2str(min_spacing)])
    disp(['x_solution: ' num2str(x_solution)])
end

