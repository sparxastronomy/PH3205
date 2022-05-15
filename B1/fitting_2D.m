x = linspace(0, 2*pi, 100);
[X,Y] = meshgrid(x,x);

a = 1;
b = 0.8;

f = a.*cos(X) + b.*sin(Y);
f_noise = 0.1.*normrnd(0,1,size(f));
f = f+f_noise;
surf(X,Y,f)
