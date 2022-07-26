clc, clear all, close all

%% Train NN on Lorenz system for r=10,28,35
dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; r=10; r_values=[10,28,35];

% r=x(4)
Lorenz = @(t,x)([ sig * (x(2) - x(1))               ; ...
                  r * x(1)-x(1) * x(3) - x(2)       ; ...
                  x(1) * x(2) - b*x(3)                ])   ;              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

% training trajectories
input=[]; output=[];
for i=1:3
    r = r_values(i);
    for j=1:100                
        x0=[30*(rand(3,1)-0.5)];
        [t,y] = ode45(Lorenz,t,x0);
        y(:,4) = r*ones([801 1]);
        input=[input; y(1:end-1,:)];
        output=[output; y(2:end,1:3)];
    
        % plot
        plot3(y(:,1),y(:,2),y(:,3)), hold on
        plot3(x0(1),x0(2),x0(3),'ro')
    end
end
grid on, view(-23,18)
%% Train NN

net = feedforwardnet([10 10 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net.trainParam.epochs=100;
net = train(net,input.',output.');

%% NN prediction rho=10
r=10;

% training trajectories
input=[]; output=[];
for j=1:100                
    x0=[30*(rand(3,1)-0.5)];
    [t,y] = ode45(Lorenz,t,x0);
    y(:,4) = r*ones([801 1]);
    input=[input; y(1:end-1,:)];
    output=[output; y(2:end,1:3)];

    % plot
    plot3(y(:,1),y(:,2),y(:,3)), hold on
    plot3(x0(1),x0(2),x0(3),'ro')
end
grid on, view(-23,18)

figure()
x0=[20*(rand(3,1)-0.5)];
[t,y] = ode45(Lorenz,t,x0);
y(:,4) = r*ones([801 1]);
plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on

x0 = [x0;r];
ynn(1,:)=x0(1:3);
for jj=2:length(t)
    y0=net(x0);
    ynn(jj,:)=y0.'; 
    x0=y0;
    x0=[x0;r];
end
plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])
legend('ODE45', 'NN')

%% NN prediction rho=28
r=28;

% training trajectories
input=[]; output=[];
for j=1:100                
    x0=[30*(rand(3,1)-0.5)];
    [t,y] = ode45(Lorenz,t,x0);
    y(:,4) = r*ones([801 1]);
    input=[input; y(1:end-1,:)];
    output=[output; y(2:end,1:3)];

    % plot
    plot3(y(:,1),y(:,2),y(:,3)), hold on
    plot3(x0(1),x0(2),x0(3),'ro')
end
grid on, view(-23,18)

figure()
x0=[20*(rand(3,1)-0.5)];
[t,y] = ode45(Lorenz,t,x0);
y(:,4) = r*ones([801 1]);
plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on

x0 = [x0;r];
ynn(1,:)=x0(1:3);
for jj=2:length(t)
    y0=net(x0);
    ynn(jj,:)=y0.'; 
    x0=y0;
    x0=[x0;r];
end
plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])
legend('ODE45', 'NN')

%% NN prediction rho=35
r=35;

% training trajectories
input=[]; output=[];
for j=1:100                
    x0=[30*(rand(3,1)-0.5)];
    [t,y] = ode45(Lorenz,t,x0);
    y(:,4) = r*ones([801 1]);
    input=[input; y(1:end-1,:)];
    output=[output; y(2:end,1:3)];

    % plot
    plot3(y(:,1),y(:,2),y(:,3)), hold on
    plot3(x0(1),x0(2),x0(3),'ro')
end
grid on, view(-23,18)

figure()
x0=[20*(rand(3,1)-0.5)];
[t,y] = ode45(Lorenz,t,x0);
y(:,4) = r*ones([801 1]);
plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on

x0 = [x0;r];
ynn(1,:)=x0(1:3);
for jj=2:length(t)
    y0=net(x0);
    ynn(jj,:)=y0.'; 
    x0=y0;
    x0=[x0;r];
end
plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])
legend('ODE45', 'NN')

%% NN prediction rho=17
r=17;

% training trajectories
input=[]; output=[];
for j=1:100                
    x0=[30*(rand(3,1)-0.5)];
    [t,y] = ode45(Lorenz,t,x0);
    y(:,4) = r*ones([801 1]);
    input=[input; y(1:end-1,:)];
    output=[output; y(2:end,1:3)];

    % plot
    plot3(y(:,1),y(:,2),y(:,3)), hold on
    plot3(x0(1),x0(2),x0(3),'ro')
end
grid on, view(-23,18)

figure()
x0=[20*(rand(3,1)-0.5)];
[t,y] = ode45(Lorenz,t,x0);
y(:,4) = r*ones([801 1]);
plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on

x0 = [x0;r];
ynn(1,:)=x0(1:3);
for jj=2:length(t)
    y0=net(x0);
    ynn(jj,:)=y0.'; 
    x0=y0;
    x0=[x0;r];
end
plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])
legend('ODE45', 'NN')

%% NN prediction rho=40
r=40;

% training trajectories
input=[]; output=[];
for j=1:100                
    x0=[30*(rand(3,1)-0.5)];
    [t,y] = ode45(Lorenz,t,x0);
    y(:,4) = r*ones([801 1]);
    input=[input; y(1:end-1,:)];
    output=[output; y(2:end,1:3)];

    % plot
    plot3(y(:,1),y(:,2),y(:,3)), hold on
    plot3(x0(1),x0(2),x0(3),'ro')
end
grid on, view(-23,18)

figure()
x0=[20*(rand(3,1)-0.5)];
[t,y] = ode45(Lorenz,t,x0);
y(:,4) = r*ones([801 1]);
plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on

x0 = [x0;r];
ynn(1,:)=x0(1:3);
for jj=2:length(t)
    y0=net(x0);
    ynn(jj,:)=y0.'; 
    x0=y0;
    x0=[x0;r];
end
plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])
legend('ODE45', 'NN')

%% Hundred NN r=17
r=17;

% training trajectories
input=[]; output=[];
for j=1:100                
    x0=[30*(rand(3,1)-0.5)];
    [t,y] = ode45(Lorenz,t,x0);
    y(:,4) = r*ones([801 1]);

    % nn
    x0=[20*(rand(3,1)-0.5)];
    [t,y] = ode45(Lorenz,t,x0);
    y(:,4) = r*ones([801 1]);
    
    x0 = [x0;r];
    ynn(1,:)=x0(1:3);
    for jj=2:length(t)
        y0=net(x0);
        ynn(jj,:)=y0.'; 
        x0=y0;
        x0=[x0;r];
    end
end

mse = mean(mean( (y(:,1:3) - ynn(:,1:3)).^2 ));

%% Hundred NN r=40
r=40;

% training trajectories
input=[]; output=[];
for j=1:100                
    x0=[30*(rand(3,1)-0.5)];
    [t,y] = ode45(Lorenz,t,x0);
    y(:,4) = r*ones([801 1]);

    % nn
    x0=[20*(rand(3,1)-0.5)];
    [t,y] = ode45(Lorenz,t,x0);
    y(:,4) = r*ones([801 1]);
    
    x0 = [x0;r];
    ynn(1,:)=x0(1:3);
    for jj=2:length(t)
        y0=net(x0);
        ynn(jj,:)=y0.'; 
        x0=y0;
        x0=[x0;r];
    end
end

mse = mean(mean( (y(:,1:3) - ynn(:,1:3)).^2 ));

