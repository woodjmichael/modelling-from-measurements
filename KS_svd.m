clear all, close all, clc
addpath('./Datasets');
load('kuramoto_sivishinky.mat')

cutoff = tt > 120;
cutoff = cutoff & tt<130;

%%
contour( x/(2*pi), tt(cutoff), uu(:,cutoff).', [-10 -5 0 5 10])
shading interp
colormap(gray)

%%
[XX, TT] = meshgrid(x,tt);

pcolor(uu')
