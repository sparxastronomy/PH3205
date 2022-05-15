%----------------------------------------------------------------
% Program : Calculate Probability Density as a function of time
% for a particle trapped in a triple-well potential.
%----------------------------------------------------------------
clc
clear
% Potential due to three square wells of width a
% and a distance 2a apart.
L = 5;                   % Interval Length.
N = 1000;                % No of points.
x = linspace(-L, L, N).';% Coordinate vector.
dx = x(2) - x(1);        % Coordinate step.
a = L/30;                % Width    
b = L/50;                % Separation
D = 200;                 % Depth
U = -D*(heaviside(x+1.5*a+b)-heaviside(x+0.5*a+b)+heaviside(x+0.5*a) ...
    -heaviside(x-0.5*a)+heaviside(x-0.5*a-b)-heaviside(x-1.5*a-b));

% Finite-difference representation of Laplacian and Hamiltonian,
% where hbar = m = 1.
hbar = 1;
m = 1;
e    = ones(N,1);
Lap  = spdiags([e -2*e e],[-1 0 1],N,N) / dx^2;
H    = -(1/2)*(hbar^2/m)*Lap + spdiags(U,0,N,N);

% Find and sort lowest nmodes eigenvectors and eigenvalues of H.
nmodes  = 2;
[V,E]   = eigs(H,nmodes,'smallestreal');
[E,ind] = sort(diag(E)); % Convert E to vector and sort low to high.
V       = V(:,ind);      % Rearrange coresponding eigenvectors
% Rescale eigenvectors so that they are always
% positive at the center of the right well.
for c = 1:nmodes
    V(:,c) = V(:,c) / sign(V((3*N/4),c));
end

%----------------------------------------------------------------
% Compute and display normalized prob. density rho(x,t).
%----------------------------------------------------------------
% Parameters for solving the problem in the interval 0 < t < TF.
TF = 4*pi*hbar/(E(2)-E(1)); % Length of time interval.
NT = 100;                   % No. of time points.
t  = linspace(0,TF,NT);     % Time vector.
% Compute probability normalization constant (at T=0).
psi_o = 0.5*V(:,1) + 0.5*V(:,2);        % Wavefunction at T=0.
sq_norm = psi_o' * psi_o * dx;          % Square norm = |<ff|ff>|^2.
Usc = 4*U*max(abs(V(:))) / max(abs(U)); % Rescale U for plotting.
% Compute and display rho(x,t) for each time t.
figure1 = figure;
% Plot lowest 2 eigenfunctions.
plot(x,V(:,1), x,V(:,2),'--g',x,U/500);
legend('\psi_{E_0}(x)','\psi_{E_1}(x)','Double well potential (rescaled)','Location','EastOutside');
xlabel('x (m)');
ylabel('unnormalized wavefunction (1/m)');
ax = gca; % Get the Current Axes object
ax.XLim = [-1 1];


%% Displying the animation
vw1 = VideoWriter('tripleWell.avi'); % Prepare the new file.
%vw1.FrameRate=30 by default
open(vw1);
for jj = 1:NT % time index parameter for stepping through loop.
    % Compute wavefunction psi(x,t) and rho(x,t) at t=t(jj).
    psi = 0.5*V(:,1)*exp(-1i*E(1)*t(jj)) ...
        + 0.5*V(:,2)*exp(-1i*E(2)*t(jj));
    rho = conj(psi) .* psi / sq_norm; % Normalized probability density.
    % Plot rho(x,jj) and rescaled potential energy Usc.
    plot(x,rho,'b-', x, Usc,'.-');
    axis([-L/8 L/8 -1 6]);
    xlabel('x (m)');
    ylabel('probability density (1/m)');
    title(['t = ' num2str(t(jj), '%05.2f') ' s']);
    writeVideo(vw1, getframe(gcf));
    % getframe returns a movie frame, a snapshot (pixmap) of the current axes or figure.
end
close(vw1);