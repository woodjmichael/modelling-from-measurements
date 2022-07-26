clear all, close all, clc
addpath('./Datasets/');
load pelts.mat
%rng(42,'philox')

% Data
X =  data(1:2,:);
[m,n] = size(X);    % data matrix dimensions: space, time
t = 1:n;

%% Delay
% rmse=22.06, h=15, r=14

% user inputs
h = 15; % lenght of each shift row
r = 14;  % truncate at r-number of modes

% some checks
h1 = h*m; % second dimension of H
h2 = n-h; % second dimension of H
r = min([r,h1,h2-1]);

% Shift matrix
H=[];
for j=1:h
    H = [   H; 
            X(:, j:end-h+j) ]; 
end 

%rows=size(XX,1);
%columns=size(XX,2);
%[U,S,V] = svd(H, 'econ'); 

% figure(1)
% plot(U(:,1:4))
% title('U Cols')
% 
% figure(3)
% plot(V(:,1:4))
% title('V Cols')
% 
% figure(3)
% plot(diag(S)/(sum(diag(S))))
% title('S')

X1 = H(:, 1:end-1);
X2 = H(:, 2:end);
[U,S,V] = svd(X1,'econ');

%  Compute DMD (Phi are eigenvectors)


U = U(:,   1:r);
S = S(1:r, 1:r);
V = V(:,   1:r);

Atilde = U'*X2*V*inv(S); % U'*X2*V/S;
[W,L] = eig(Atilde);
Phi = X2*V*inv(S)*W;

% Modes
b = Phi \ X1(:, 1); % initial conditions

l=log(diag(L)); % continuous eigenvalues

modes = [];
for t2 = 0:40                       % name t is taken
    modes = [modes, (b.*exp(l*t2))];    % time_dynamics
end    

Xhat = Phi*modes;
Xhat = real(Xhat);

% plots
plot_true_and_dmd(X(1,:), X(2,:), Xhat(1,1:n), Xhat(2,1:n), 'DMD Delay')

% first four eigven values of Atilde
figure()
plot(real(Phi(:,1)))
hold on
plot(real(Phi(:,3)))
hold on
plot(real(Phi(:,5)))
hold on
plot(real(Phi(:,7)))

% singular values
figure()
plot(diag(S)) 

% eigenvalues
figure()
scatter(real(l),imag(l))

figure()
plot(abs(real(l)))

%% Delay + bagging
% rmse=21.58, r=17, h=12, p=18

% user inputs
r = 30;  % truncate at r-number of modes (try to keep r large)
h = 12; % number of rows in H equals h*m
p = 20; % bag size
num_cycles = 100;


% Hankel "shift rows" matrix
H = [];
for i = 1:h
    H = [   H; 
            X(:,i:end-h+i-1) ];
end

% some checks
h1 = h*m; % second dimension of H
h2 = n-h; % second dimension of H
r = min([r,h1,h2-1,p-1]);
p = min(p,h2);

lambda_vec_ensembleDMD =    zeros(            r, num_cycles );
b_vec_ensembleDMD =         zeros(            r, num_cycles );
w_vec_ensembleDMD =         zeros( size(H,1), r, num_cycles );

for j = 1:num_cycles 
        % select indices
        unsorted_ind = randperm(n-h,p);

        %  sort ind so in ascending order. NOTE: evals have variable delta t
        ind = sort(unsorted_ind);

        % create dataset for this cycle by taking aforementioned indices
        xdata_cycle = H(:,ind);

        % selected index times
        t_ind = t(ind);        
        
        X1 = xdata_cycle(:, 1:end-1);
        X2 = xdata_cycle(:, 2:end);
        [U,S,V] = svd(X1,'econ');
        
        %  Compute DMD (Phi are eigenvectors)
        U = U(:,   1:r);
        S = S(1:r, 1:r);
        V = V(:,   1:r);
        
        Atilde = U'*X2*V*inv(S); % U'*X2*V/S;
        [W,L] = eig(Atilde);
        Phi = X2*V*inv(S)*W;
        
        w_cycle = Phi;

        % Modes
        b_cycle = Phi \ X1(:, 1); % initial conditions
        
        l=log(diag(L)); % continuous eigenvalues
        e1_cycle = l;

        % save 
        lambda_vec_ensembleDMD(:,j) = e1_cycle;
        b_vec_ensembleDMD(:,j)      = b_cycle;
        w_vec_ensembleDMD(:,:,j)    = w_cycle;
end

lambda_average =    mean( lambda_vec_ensembleDMD,   2);
b_average =         mean( b_vec_ensembleDMD,        2);
w_average =         mean( w_vec_ensembleDMD,        3);

t = 0:n-1;
Xhat = w_average*diag(b_average)*exp(lambda_average*t);
Xhat = real(Xhat);

rmse = sqrt(mean([(X(1,:) - Xhat(1,:) ).^2, ( X(2,:) - Xhat(2,:) ).^2]));


plot_true_and_dmd(  X(1,:), ...
                    X(2,:), ...
                    abs( Xhat(1,:) ), ...
                    abs( Xhat(2,:) ), ...
                    'DMD Delay Bagging' )
