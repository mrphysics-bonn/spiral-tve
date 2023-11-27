function [c,ceq] = nlcon(x, b_val_max, refoc_dur)
    
    gamma = 42576000; % Hz/T
    ceq = [];

    diffgrad_ramp = round(x(1)/x(2), 5);
    diffgrad_flat = x(3) - diffgrad_ramp;
    diffgrad_dur = diffgrad_flat + 2*diffgrad_ramp;

    x_ramp = round((1e-3 * 70 * gamma)/(180 * gamma), 5);
    spacing = round((b_val_max*1e6/(2*pi*x(1))^2-round(x(1)/x(2),5)^3/30+x(3)*round(x(1)/x(2),5)^2/6)/x(3)^2+x(3)/3, 5);
    tp = spacing - diffgrad_dur - refoc_dur - 4*x_ramp;
    tv = (spacing+diffgrad_dur-refoc_dur)/2;
    tn = (spacing+diffgrad_dur+refoc_dur)/2;

    counter = x(3)^2*(spacing-x(3)/3)+diffgrad_ramp^3/30-x(3)*diffgrad_ramp^2/6;
    alpha_x = sqrt(counter/(x(3)^2*(spacing-diffgrad_dur+5/3*refoc_dur-4/3*x_ramp)));
    alpha_y = sqrt(counter/(x(3)^2*(tp+8*x_ramp+2*pi^2*x_ramp^2/tp+8*pi^2*x_ramp^3/(5*tp^2))));
    C = (45*spacing*diffgrad_dur^4 - 180*spacing*diffgrad_dur^3*x(3) - 60*spacing*diffgrad_dur^3*diffgrad_ramp + 270*spacing*diffgrad_dur^2*x(3)^2 + 120*spacing*diffgrad_dur^2*x(3)*diffgrad_ramp + 150*spacing*diffgrad_dur^2*diffgrad_ramp^2 ...
                - 180*spacing*diffgrad_dur*x(3)^3 - 60*spacing*diffgrad_dur*x(3)^2*diffgrad_ramp - 180*spacing*diffgrad_dur*x(3)*diffgrad_ramp^2 - 60*spacing*diffgrad_dur*diffgrad_ramp^3 + 45*spacing*x(3)^4 + 90*spacing*x(3)^2*diffgrad_ramp^2 ...
                + 45*spacing*diffgrad_ramp^4 - 12*diffgrad_dur^5 + 60*diffgrad_dur^4*x(3) - 60*diffgrad_dur^4*diffgrad_ramp - 120*diffgrad_dur^3*x(3)^2 + 120*diffgrad_dur^3*x(3)*diffgrad_ramp + 120*diffgrad_dur^3*diffgrad_ramp*tn ...
                - 120*diffgrad_dur^3*diffgrad_ramp*tv + 120*diffgrad_dur^2*x(3)^3 - 30*diffgrad_dur^2*x(3)^2*diffgrad_ramp + 90*diffgrad_dur^2*x(3)*diffgrad_ramp^2 - 240*diffgrad_dur^2*x(3)*diffgrad_ramp*tn + 240*diffgrad_dur^2*x(3)*diffgrad_ramp*tv ...
                - 40*diffgrad_dur^2*diffgrad_ramp^2*tn + 40*diffgrad_dur^2*diffgrad_ramp^2*tv + 160*diffgrad_dur^2*diffgrad_ramp^2*x_ramp - 60*diffgrad_dur*x(3)^4 - 60*diffgrad_dur*x(3)^3*diffgrad_ramp + 120*diffgrad_dur*x(3)^2*diffgrad_ramp*tn ...
                - 120*diffgrad_dur*x(3)^2*diffgrad_ramp*tv - 60*diffgrad_dur*x(3)*diffgrad_ramp^3 - 60*diffgrad_dur*diffgrad_ramp^4 + 120*diffgrad_dur*diffgrad_ramp^3*tn - 120*diffgrad_dur*diffgrad_ramp^3*tv + 12*x(3)^5 + 30*x(3)^4*diffgrad_ramp ...
                - 30*x(3)^3*diffgrad_ramp^2 + 30*x(3)^2*diffgrad_ramp^3 + 18*diffgrad_ramp^5)/(20*diffgrad_dur^2*diffgrad_ramp^2);
    alpha_z = sqrt(counter/(x(3)^2*(C/3)));
    
    Gx = sqrt(2)/3 * x(1) * x(3) * 2*pi/tp * alpha_x;
    Gy = sqrt(2/3) * x(1) * x(3) * 2*pi/tp * alpha_y;
    Gz = 4/3 * x(1) * x(3) * pi/tp * alpha_z;
           
    c(1) = Gx / (1e-3 * 68 * gamma) - 1.0;
    c(2) = Gy / (1e-3 * 68 * gamma) - 1.0;
    c(3) = Gz / (1e-3 * 68 * gamma) - 1.0;
    c(4) = x(1)*alpha_z / (1e-3 * 70 * gamma) - 1.0;
    c(5) = -1 * Gx;
    c(6) = -1 * Gy;
    c(7) = -1 * Gz;
end