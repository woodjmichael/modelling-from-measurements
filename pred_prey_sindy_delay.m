clear all, close all, clc
addpath('./Datasets/');
addpath('./sparsedynamics/utils/');
load pelts.mat
%rng(42,'philox')

% Data
data =  data(1:2,:);
xy_data =  data';

h = 2;
xy_shift = xy_data(1:end-(h-1),:);
for k = 2:h
    xy_shift(:,h*2+1) = xy_data(h:end,1);
    xy_shift(:,h*2) = xy_data(h:end,2);
end

xy_data = xy_shift;

xydot_data = gradient(xy_data')';

% xydot_data = diff(xy_data);
% xy_data = xy_data(1:end-1,:);

[n,m] = size(xy_data);    % data matrix dimensions: space, time
t = 1:n;

x0 = xy_data(1,:)'; % initial conditions

% pool
polyorder=2;
usesine=0;
Theta = pool_data_no_custom(xy_data,m,polyorder,usesine);

% sparsify
lambda = 0.005;      % lambda is our sparsification knob.
Xi = sparsifyDynamicsCustom(Theta,xydot_data,lambda)

tspan=[1:length(xy_data)];
%options = odeset('RelTol',1e-10,'AbsTol',1e-10*ones(1,n));
%[tB,xB]=ode45(@(t,xy)sparseGalerkin(t,xy_data,Xi,polyorder,usesine),tspan,x0,options);  % approximate


%%
% estimate
xydot_hat = Theta * Xi;
xy_hat(1,:) = xy_data(1,:);
for i=2:size(xy_data,1)
    xy_hat(i,:) = xy_hat(i-1,:) + xydot_hat(i-1,:);
end


%% plot
plot_true_and_dmd(  xy_data(:,1), xy_data(:,2), ...
                    xy_hat(:,1), xy_hat(:,2), ...
                    'SINDy')

% figure()
% plot(t,xy_data(:,1), 'b')
% hold on
% plot(t,xy_data(:,2), 'r')
% hold on
% plot(t,xy_hat(:,1), 'b--')
% hold on
% plot(t,xy_hat(:,2), 'r--')
% 
% mae = mean(mean(abs(xy_data - xy_hat)))



function yout = pool_data_no_custom(yin,nVars,polyorder,usesine)
    % Copyright 2015, All Rights Reserved
    % Code by Steven L. Brunton
    % For Paper, "Discovering Governing Equations from Data: 
    %        Sparse Identification of Nonlinear Dynamical Systems"
    % by S. L. Brunton, J. L. Proctor, and J. N. Kutz
    
    n = size(yin,1);
    % yout = zeros(n,1+nVars+(nVars*(nVars+1)/2)+(nVars*(nVars+1)*(nVars+2)/(2*3))+11);
    
    ind = 1;
    % poly order 0
    yout(:,ind) = ones(n,1);
    ind = ind+1;
    
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

function Xi = sparsifyDynamicsCustom(Theta,dXdt,lambda)
    % Copyright 2015, All Rights Reserved
    % Code by Steven L. Brunton
    % For Paper, "Discovering Governing Equations from Data: 
    %        Sparse Identification of Nonlinear Dynamical Systems"
    % by S. L. Brunton, J. L. Proctor, and J. N. Kutz
    
    % compute Sparse regression: sequential least squares
    Xi = Theta\dXdt;  % initial guess: Least-squares
    m = size(Xi,2); % space dimension
    
    % lambda is our sparsification knob.
    for k=1:10
        smallinds = (abs(Xi)<lambda);   % find small coefficients
        Xi(smallinds)=0;                % and threshold
        for ind = 1:m                  
            biginds = ~smallinds(:,ind);
            % Regress dynamics onto remaining terms to find sparse Xi
            Xi(biginds,ind) = Theta(:,biginds)\dXdt(:,ind); 
        end
    end
end