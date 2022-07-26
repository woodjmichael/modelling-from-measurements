% Copyright 2015, All Rights Reserved
% Code by Steven L. Brunton
% For Paper, "Discovering Governing Equations from Data: 
%        Sparse Identification of Nonlinear Dynamical Systems"
% by S. L. Brunton, J. L. Proctor, and J. N. Kutz

clear all, close all, clc
figpath = '../figures/';
addpath('./sparsedynamics/utils/');

%% solve LV
load pelts.mat

xy_data =  data(1:2, :)';
xydot_data = -1*gradient(xy_data);
%xydot_data = gradient(xy_data);

% initial guess
p      = [  1;     % b
            1;     % p
            1;     % r
            1  ]'; % d   

% p result = [0.8794    0.0074    0.0122    0.4186]
p = lsqcurvefit(@LotkaVolterra,p,xy_data,xydot_data);

xy_LV = LotkaVolterra(p,xy_data);

%% generate Data
polyorder = 5;  % search space up to fifth order polynomials
usesine = 0;    % no trig functions
n = 2;          % 2D system
tspan=1:30;   % time span
x0 = [20; 32];        % initial conditions
options = odeset('RelTol',1e-10,'AbsTol',1e-10*ones(1,n));
[t,xy]=ode45(@(t,xy)LV2(xy),tspan,x0,options);  % integrate

%% compute Derivative 
eps = .05;      % noise strength
xyt = xy';
for i=1:length(xy)    
    dx(i,:) = LV2(:,i)';
end
%dx = dx + eps*randn(size(dx));   % add noise

%% pool Data  (i.e., build library of nonlinear time series)
Theta = poolData(x,n,polyorder,usesine);
m = size(Theta,2);

%% compute Sparse regression: sequential least squares
lambda = 0.05;      % lambda is our sparsification knob.
Xi = sparsifyDynamics(Theta,dx,lambda,n)

%% integrate true and identified systems
[tA,xA]=ode45(@(t,x)rhs(x),tspan,x0,options);   % true model
[tB,xB]=ode45(@(t,x)sparseGalerkin(t,x,Xi,polyorder,usesine),tspan,x0,options);  % approximate

%% FIGURES!!
figure
dtA = [0; diff(tA)];
plot(xA(:,1),xA(:,2),'r','LineWidth',1.5);
hold on
dtB = [0; diff(tB)];
plot(xB(:,1),xB(:,2),'k--','LineWidth',1.2);
xlabel('x_1','FontSize',13)
ylabel('x_2','FontSize',13)
l1 = legend('True','Identified');

figure
plot(tA,xA(:,1),'r','LineWidth',1.5)
hold on
plot(tA,xA(:,2),'b-','LineWidth',1.5)
plot(tB(1:10:end),xB(1:10:end,1),'k--','LineWidth',1.2)
hold on
plot(tB(1:10:end),xB(1:10:end,2),'k--','LineWidth',1.2)
xlabel('Time')
ylabel('State, x_k')
legend('True x_1','True x_2','Identified')

%% LV
function xydot = LotkaVolterra(p, xy_data)
    x = xy_data(:,1);
    y = xy_data(:,2);
    
    xdot = p(1)*x - p(2)*y.*x;
    ydot = p(3)*x.*y - p(4)*y;
    xydot = [xdot, ydot];
end

function xydot = LV2(xy)
    p = [0.8794    0.0074    0.0122    0.4186];
    x = xy(1,:);
    y = xy(2,:);
    
    xdot = p(1)*x - p(2)*x.*y;
    ydot = p(3)*x.*y - p(4)*y;
    xydot = [xdot; ydot];
end