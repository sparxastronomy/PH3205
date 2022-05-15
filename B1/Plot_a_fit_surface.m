clc;
clear;

n = 100; %number of points in one axis
x = linspace(-1,1,100);
y = linspace(-1,1,100);

% creating plotting grid
[X,Y] = meshgrid(x,y);
% degree of polynomial
a = 2;      %for x
b = 3;      %for y
Z = X.^a - Y.^b;
Z = Z +0.2*rand(100,100); %adding fluctuations to the surface

figure("Name","DataPoints")
surf(X,Y,Z,Marker=".",FaceColor='interp', DisplayName="$Z = f(x,y)$")
legend(Interpreter="latex",Location="northwest",FontSize=14)
xlabel( '$x$', 'Interpreter', 'latex', FontSize=14);
ylabel( '$y$', 'Interpreter', 'latex', FontSize=14 );
zlabel( '$f$', 'Interpreter', 'latex', FontSize=14 );
title("Surface Plot of Input Data", FontSize=18)

fprintf("Fitting the Data using a ploynomial")
[fitted, gof] = createFit(X,Y,Z,a,b)


%% Function for fitting
function [fitresult, gof] = createFit(x, y, f, a,b)
    %CREATEFIT(X,Y,F)
    %  Create a fit using polynomial model
    %
    %  Data for fit:
    %      X Input : x
    %      Y Input : y
    %      Z Output: f
    %      degree of X:a
    %      degree of Y:b
    %  Output:
    %      fitresult : a fit object representing the fit.
    %      gof : structure with goodness-of fit info.

    [xData, yData, zData] = prepareSurfaceData( x, y, f );
    % Set up fittype and options.
    ft = 'poly'+string(a)+string(b);

    % fitting the data
    [fitresult, gof] = fit( [xData, yData], zData, ft, 'Normalize', 'on' );

    % plotting the results
    figure("Name","Fitting")
    h = plot( fitresult );
    hold on
    surf(x,y,f,"LineStyle","none","FaceColor","none",Marker=".",MarkerEdgeColor="red")
    legend(h,'Fitted Surface', Interpreter="latex", Location="northwest", FontSize=14)
    title("Fitted Surface Along with Input Points", FontSize=18)
    % lables
    xlabel( '$x$', 'Interpreter', 'latex', FontSize=14);
    ylabel( '$y$', 'Interpreter', 'latex', FontSize=14 );
    zlabel( '$f$', 'Interpreter', 'latex', FontSize=14 );
end