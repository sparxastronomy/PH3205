clc
clear all
% Potential due to two square wells of width 2w
% and a distance 2a apart.
L = 5;                   % Interval Length.
N = 1000;                % No of points.
x = linspace(-L, L, N).';% Coordinate vector.
dx = x(2) - x(1);        % Coordinate step.
a = L/20;    % width
b = L/10;    % separation
D = 200;     % depth
U = -D*(heaviside(x+ 1.5*a + b) - heaviside(x + 0.5*a +b) + heaviside(x+0.5*a) ...
    - heaviside(x-0.5*a) + heaviside(x - (0.5*a + b)) -heaviside(x-((a*3/2)+b)));

figure(1)
plot(x, U)
xlabel("x (m)")
ylabel("Potential")

%% Setting up the hamiltonian
hbar = 1;
m = 1;
e    = ones(N,1);
Lap  = spdiags([e -2*e e],[-1 0 1],N,N) / dx^2;
H    = -(1/2)*(hbar^2/m)*Lap + spdiags(U,0,N,N);

%% Getting the 3 lowest eigen values
nmodes  = 3;
[V,E]   = eigs(H,nmodes,'smallestreal');
[E,ind] = sort(diag(E)); % Convert E to vector and sort low to high.
V       = V(:,ind);

usc = D*U*max(abs(V))/max(abs(U));


figure(2)
ax = plot(x,U/500, Color='black', LineStyle='--', DisplayName='Scaled Potential');
hold on
plot(x,V(:,1), DisplayName='Energy='+string(E(1))) 
plot(x,V(:,2), DisplayName='Energy='+string(E(2)))
plot(x,V(:,3), DisplayName='Energy='+string(E(3)))
ylim([-0.5 0.25])
legend(Location="best")
xlabel("x (in m)")
ylabel("Un-normalized wave funcitons")