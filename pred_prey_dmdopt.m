clear all, close all, clc
addpath('./Datasets/');
load pelts.mat
%rng(42,'philox')

% Data
X =  data(1:2,:);
[m,n] = size(X);    % data matrix dimensions: space, time
t = 1:n;

%% Basic
% rmse=31.67, r=2, imode=1

% user input
r = 2;      % reduced rank, but imode=1 always returns full matrices
imode = 1;  % operational mode of optdmd(), for 1 do full rank SVD

[w,e1,b] = optdmd(X,t,r,imode);
Xhat = w*diag(b)*exp(e1*t);

plot_true_and_dmd(  X(1,:), ...
                    X(2,:), ...
                    Xhat(1,:), ...
                    Xhat(2,:), ...
                    'OptDMD' )



%% Bagging

% user inputs
r = 2;      % reduced rank, but imode=1 always returns full matrices
imode = 1;  % operational mode of optdmd(), for 1 do full rank SVD
num_cycles = 100; % size of ensemble
p = 20; % bag size

lambda_vec_ensembleDMD =    zeros(r,num_cycles);
b_vec_ensembleDMD =         zeros(r,num_cycles);
w_vec_ensembleDMD =         zeros(m,r,num_cycles);

% get initial e1
[w,e1,b] = optdmd(X,t,r,imode);

for j = 1:num_cycles 
        % select indices
        unsorted_ind = randperm(n,p);

        %  sort ind so in ascending order. NOTE: evals have variable delta t
        ind = sort(unsorted_ind);
        %ind = unsorted_ind;

        % create dataset for this cycle by taking aforementioned indices
        xdata_cycle = X(:,ind);

        % selected index times
        t_ind = t(ind);
    
        % optdmd needs varpro_opts for some reason
        [w_cycle,e1_cycle,b_cycle] = optdmd( xdata_cycle, ...
                                             t_ind, ...
                                             r, ...
                                             imode, ...
                                             e1);

        % save 
        lambda_vec_ensembleDMD(:,j) = e1_cycle;
        b_vec_ensembleDMD(:,j)      = b_cycle;
        w_vec_ensembleDMD(:,:,j)    = w_cycle;
end

lambda_average =    mean(lambda_vec_ensembleDMD,2);
b_average =         mean(b_vec_ensembleDMD,2);
w_average =         mean(w_vec_ensembleDMD,3);

Xhat = w_average*diag(b_average)*exp(lambda_average*t);
Xhat = real(Xhat);

plot_true_and_dmd(  X(1,:), ...
                    X(2,:), ...
                    abs( Xhat(1,:) ), ...
                    abs( Xhat(2,:) ), ...
                    'OptDMD Bagging' )

%% Time Delay
% rmse=12.08 for r=12, imode=1, h=6

% user inputs
imode = 1;  % 1 = return full rank matrices, 2 = reduce to r
r = 12;     % rank of reduced matrices

% Hankel "shift rows" matrix
h = 6; % related to the length of each Hankel row length
H = [];
for i = 1:h
    H = [   H; 
            X(:,i:end-h+i-1) ];
end

% time dimension is smaller with the Hankel
[w,e1,b] = optdmd(  H, t(1:end-h), r, imode );

Xhat = w*diag(b)*exp(e1*t);
Xhat = real(Xhat);

plot_true_and_dmd(  X(1,:), ...
                    X(2,:), ...
                    abs( Xhat(1,:) ), ...
                    abs( Xhat(2,:) ), ...
                    'OptDMD Delay' )

% figure()
% scatter(real(e1), imag(e1))

%% Time Delay + Bagging
% rmse=12.08 for r=12, imode=1, h=6, p=24

% user inputs
r = 12; 
imode = 1;
h = 6; % related to the length of each Hankel row length
p = 24; % bag size
num_cycles = 100;


% Hankel "shift rows" matrix
H = [];
for i = 1:h
    H = [   H; 
            X(:,i:end-h+i-1) ];
end

% need to update r
r = min(r,size(H,1));

lambda_vec_ensembleDMD =    zeros(            r, num_cycles );
b_vec_ensembleDMD =         zeros(            r, num_cycles );
w_vec_ensembleDMD =         zeros( size(H,1), r, num_cycles );

[w,e1,b] = optdmd(  H, t(1:end-h), r, imode );

for j = 1:num_cycles 
        % select indices
        unsorted_ind = randperm(n-h,p);

        %  sort ind so in ascending order. NOTE: evals have variable delta t
        ind = sort(unsorted_ind);

        % create dataset for this cycle by taking aforementioned indices
        xdata_cycle = H(:,ind);

        % selected index times
        t_ind = t(ind);
    
        % optdmd needs varpro_opts for some reason
        [w_cycle,e1_cycle,b_cycle] = optdmd( xdata_cycle, ...
                                             t_ind, ...
                                             r, ...
                                             imode, ...
                                             e1);

        % save 
        lambda_vec_ensembleDMD(:,j) = e1_cycle;
        b_vec_ensembleDMD(:,j)      = b_cycle;
        w_vec_ensembleDMD(:,:,j)    = w_cycle;
end

lambda_average =    mean( lambda_vec_ensembleDMD,   2);
b_average =         mean( b_vec_ensembleDMD,        2);
w_average =         mean( w_vec_ensembleDMD,        3);

Xhat = w_average*diag(b_average)*exp(lambda_average*t);
Xhat = real(Xhat);

rmse = sqrt(mean([(X(1,:) - Xhat(1,:) ).^2, ( X(2,:) - Xhat(2,:) ).^2]));

plot_true_and_dmd(  X(1,:), ...
                    X(2,:), ...
                    abs( Xhat(1,:) ), ...
                    abs( Xhat(2,:) ), ...
                    'OptDMD Delay Bagging' )
