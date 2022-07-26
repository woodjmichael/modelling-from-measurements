clear all, close all, clc
%rng(42,'philox')
addpath('./Datasets/');
addpath('./sparsedynamics/utils/');

load pelts.mat
n = size(data,1)-1;
options = odeset('RelTol',1e-10,'AbsTol',1e-10*ones(1,n));

xy_data =  data(1:n, :)';
xydot_data = [gradient(xy_data(:,1)), gradient(xy_data(:,2))];
%xydot_data = diff(xy_data);
%xy_data = xy_data(1:end-1,:);

tspan=1:1:size(xy_data,1);   % time span
t = tspan';
x0 = xy_data(1,:)'; % initial conditions

% pool
polyorder=5;
usesine=0;
Theta = pool_data_no_constant(xy_data,n,polyorder,usesine);
m = size(Theta,2);

% sparsify
lambda = 0.001;      % lambda is our sparsification knob.
Xi = sparsifyDynamics(Theta,xydot_data,lambda,n)

% estimate
xydot_hat = Theta * Xi;
xy_hat(1,:) = xy_data(1,:);
for i=2:size(xy_data,1)
    xy_hat(i,:) = xy_hat(i-1,:) + xydot_hat(i-1,:);
end


%% plot
figure()
plot(t,xy_data(:,1), 'b')
hold on
plot(t,xy_data(:,2), 'r')
hold on
plot(t,xy_hat(:,1), 'b--')
hold on
plot(t,xy_hat(:,2), 'r--')

mae = mean(mean(abs(xy_data - xy_hat)))



function yout = pool_data_no_constant(yin,nVars,polyorder,usesine)
% Copyright 2015, All Rights Reserved
% Code by Steven L. Brunton
% For Paper, "Discovering Governing Equations from Data: 
%        Sparse Identification of Nonlinear Dynamical Systems"
% by S. L. Brunton, J. L. Proctor, and J. N. Kutz

n = size(yin,1);
% yout = zeros(n,1+nVars+(nVars*(nVars+1)/2)+(nVars*(nVars+1)*(nVars+2)/(2*3))+11);

ind = 1;
% % poly order 0
% yout(:,ind) = ones(n,1);
% ind = ind+1;

% poly order 1
for i=1:nVars
    yout(:,ind) = yin(:,i);
    ind = ind+1;
end

% x^2
if(polyorder>=2)
    % poly order 2
    for i=1:nVars
        for j=i:nVars
            yout(:,ind) = yin(:,i).*yin(:,j);
            ind = ind+1;
        end
    end
end

if(polyorder>=3)
    % poly order 3
    for i=1:nVars
        for j=i:nVars
            for k=j:nVars
                yout(:,ind) = yin(:,i).*yin(:,j).*yin(:,k);
                ind = ind+1;
            end
        end
    end
end

if(polyorder>=4)
    % poly order 4
    for i=1:nVars
        for j=i:nVars
            for k=j:nVars
                for l=k:nVars
                    yout(:,ind) = yin(:,i).*yin(:,j).*yin(:,k).*yin(:,l);
                    ind = ind+1;
                end
            end
        end
    end
end

if(polyorder>=5)
    % poly order 5
    for i=1:nVars
        for j=i:nVars
            for k=j:nVars
                for l=k:nVars
                    for m=l:nVars
                        yout(:,ind) = yin(:,i).*yin(:,j).*yin(:,k).*yin(:,l).*yin(:,m);
                        ind = ind+1;
                    end
                end
            end
        end
    end
end

if(usesine)
    for k=1:10;
        yout = [yout sin(k*yin) cos(k*yin)];
    end
end

end