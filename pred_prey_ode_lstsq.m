clear all, close all, clc
load pelts.mat

xy_data =  data(1:2, :)';
xydot_data = -1*gradient(xy_data);
%xydot_data = gradient(xy_data);

% initial guess
p      = [  1;     % b
            1;     % p
            1;     % r
            1  ]'; % d   

p = lsqcurvefit(@LotkaVolterra,p,xy_data,xydot_data)

plot(xy_data(:,1), 'b')
hold on
plot(xy_data(:,2), 'r')
hold on

xy_LV = LotkaVolterra(p,xy_data);

% n = 2;          % 2D system
% LV = @(x)[p(1)*x(1) + p(2)*x(1)*x(2); p(3)*x(1)*x(2) + p(4)*x(2)];
% tspan=1:30;   % time span
% x0 = [20; 32];        % initial conditions
% options = odeset('RelTol',1e-10,'AbsTol',1e-10*ones(1,n));
% [t,x]=ode45(@(t,x)LV(x),tspan,x0,options);  % integrate

plot(xy_LV(:,1), 'b--')
%plot(x(:,1), 'b--')
hold on
plot(xy_LV(:,2), 'r--')
%plot(x(:,2), 'r--')

legend('Prey true','Predator true', 'Prey LV', 'Predator LV')
title('Lotka-Volterra Fit: b=0.880, p=0.007, r=0.012, d=0.419, RMSE=28.7')

rmse = sqrt(mean(mean([(xy_data(:,1)-xy_LV(:,1)).^2, (xy_data(:,2)-xy_LV(:,2)).^2, ])))

%%

function xydot = LotkaVolterra(p, xy_data)
    x = xy_data(:,1);
    y = xy_data(:,2);
    
    xdot = p(1)*x - p(2)*y.*x;
    ydot = p(3)*x.*y - p(4)*y;
    xydot = [xdot, ydot];
end

