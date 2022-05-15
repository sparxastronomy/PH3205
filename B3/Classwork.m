%% shooting method with rot finding approach
clc
clear
warning('off','all')
warning


% definig constants and initial conditions
L = 10;
T0 = 300;

%% getting solution for fixed L
Icguess_target = fzero(@(x) bar_res(x, L),-1);
[x,y] = ode45(@bar_temp, [0 L], [T0 Icguess_target]);
% plotting the obtained solution
figure
plot(x,y(:,1));
xlabel('x');
ylabel('T');
title('Temperature districbution in a heated rod');

%% Getting solution for varying L
L = linspace(1,20,20); % varying L
sol_ar = nan(1, length(L)); % array for storing temperature at midpoint values

for i = 1:length(L)
    EndPoint = L(i);
    % creating arrays with midpoint of the interval
    a1 = linspace(0,EndPoint/2, 50);
    a2 = linspace(EndPoint/2, EndPoint, 51);
    a2 = a2(2:length(a2));
    tspan = [a1 a2];

    % obtaining target solution for current solution
    Icguess_target = fzero(@(x) bar_res(x, EndPoint),-1);
    [x,y] = ode45(@bar_temp, tspan, [T0 Icguess_target]);

    %storing midpoint temprture
    sol_ar(i) = y(50,1);
end

% ploting midpoint values vs L
figure(2)
plot(L,sol_ar, 'b-',Marker='.', MarkerEdgeColor='red',MarkerSize=15)
xlabel("Lenght of Rod, $L$ ($m$)", Interpreter="latex",FontSize=14)
ylabel("Temperatue at midpoint, $T_{L/2}$ ($^\circ C$)", ...
    Interpreter="latex",FontSize=14)
title("Temperature distribution at midpoint", FontSize=15)

%% Defining functions

function dTdx = bar_temp(x,y)
% Returns system of 1st order ODE at current position x
% y = [current position, current dy/dx]
h_const = 0.05;
sigma = 2.7*10^(-9);
T_inf = 200;
dTdx = [y(2);-h_const*(T_inf-y(1))-sigma*(T_inf^4-y(1)^4)];
end


function r = bar_res(Icguess, L)
% Objective fuction which returs the difference between the end points of
% BVP and guess solution
T0 = 300;
TL = 400;
[x,y]= ode45(@bar_temp, [0 L], [T0 Icguess]);
r = y(end,1)-TL;
end