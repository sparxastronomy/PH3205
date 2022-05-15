% Program to plot four unequal width finite potential well 
% separated by equal widths

clc
clear all
%% Parameters for setting up the well in the interval -L < x < L.
L = 5;                   % Interval Length.
N = 1000;                % No of points.
x = linspace(-L, L, N).';% Coordinate vector.
dx = x(2) - x(1);        % Coordinate step.
a = L/30;
b = L/50;
%% Setting up the well
U = -200*(heaviside(x+3*a+1.5*b)-heaviside(x+2*a+1.5*b)+ ...
    heaviside(x+2*a+0.5*b)-heaviside(x+0.5*b)+heaviside(x-0.5*b) ...
    -heaviside(x-0.5*a-0.5*b)+heaviside(x-1.5*b-0.5*a)-heaviside(x-1.5*b-3.5*a));


%% Plotting the potential
plot(x,U,'-b');                % plot V(x) and rescaled U(x).
xlabel('x (m)');
ylabel('V(x)');
title("4 unequal width potential well separated by equal widths")
ax = gca;
ax.XLim = sqrt(2)*[-1 1];
ax.YLim = [-220 10];