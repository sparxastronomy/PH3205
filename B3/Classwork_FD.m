%% Fininte difference method code

clc
clear all
E= 200*10^9;
I=30000*(1/100)^4;
w=15000;
L=3;
%Boundary conditions
y1 = 0; yn = 0;
%Solution nodes
n = 50; %number of intervels
x = linspace(0, L, n);
dx = x(2)-x(1);
x_int = x(2:end-1);
%set up coefficient matrix
n_mat = n-2;
%% Method 1 of getting the matrix
A = zeros(n_mat, n_mat);
%diagonal
for ndx = 1:n_mat
    A(ndx,ndx)=-2;
end
%off-diagonal
for ndx=1:n_mat-1
    A(ndx,ndx+1) = 1;
    A(ndx+1,ndx) = 1;
end

%% Method 2 of getting the matrix
diag_vals = [1*ones(n_mat,1) -2*ones(n_mat,1) 1*ones(n_mat,1)];
B = spdiags(diag_vals, -1:1, n_mat, n_mat);
B1 = eig(B);

%% Obtaining the solution
RHS = (w*dx^2/(2*E*I))*(L*x_int-x_int.^2);
y_int = A\RHS';
%y_int = diag(B1)/RHS;
y = [y1, y_int', yn];

%% Poltting the solution
plot(linspace(0,L,length(y)), y)
xlabel('x (m)', FontSize=14)
ylabel('Deflection of Beam, y(x) (m)', FontSize=14)
