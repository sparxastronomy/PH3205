clc;
clear;

n = 100; %number of points in one axis
x = linspace(-1,1,100);
y = linspace(-1,1,100);

% creating plotting grid
[X,Y] = meshgrid(x,y);
Z = X.^2 - Y.^2;
Z = Z +0.5*rand(100,100); %adding fluctuations to the surface

fprintf("Fitting the Data using a ploynomial")
[fitted, gof] = createFit(X,Y,Z)


%% Function for fitting
function [fitresult, gof] = createFit(x, y, f)
    %CREATEFIT(X,Y,F)
    %  Create a fit using polynomial model
    %
    %  Data for fit:
    %      X Input : x
    %      Y Input : y
    %      Z Output: f
    %  Output:
    %      fitresult : a fit object representing the fit.
    %      gof : structure with goodness-of fit info.

    [xData, yData, zData] = prepareSurfaceData( x, y, f );
    % Set up fittype and options.
    ft = 'poly22';

    % fitting the data
    [fitresult, gof] = fit( [xData, yData], zData, ft, 'Normalize', 'on' );

    % plotting the results
    figure("Name","Fitting")
    h = plot( fitresult, [xData, yData], zData );
    legend(h,'Fitted Surface','$f$ vs $x,y$', Interpreter="latex",Location='northwest', FontSize=14)
    title("Fitted Surface Along with Input Points", FontSize=18)
    % lables
    xlabel( '$x$', 'Interpreter', 'latex', FontSize=14 );
    ylabel( '$y$', 'Interpreter', 'latex', FontSize=14 );
    zlabel( '$f$', 'Interpreter', 'latex', FontSize=14 );
end