function Bayes(m1,m2,s1,s2)

x1 = -1:0.2:10;
x2 = -1:0.2:10;
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];

Y1 = mvnpdf(X,m1',s1);
Y2 = mvnpdf(X,m2',s2);

Y1_reshaped = reshape(Y1,length(x2),length(x1));
Y2_reshaped = reshape(Y2,length(x2),length(x1));

% Surf plot for Class 1
figure;
surf(x1,x2,Y1_reshaped);
colormap(jet);
shading interp;
caxis([min(Y1_reshaped(:))-0.5*range(Y1_reshaped(:)), max(Y1_reshaped(:))]);
axis([-1 10 -1 10 0 0.25]);
xlabel('x_1');
ylabel('x_2');
zlabel('P(X|\omega_1)');
title('Probability Density Function for Class 1');
colorbar;
view(-30, 30); 

% Surf plot for Class 2
figure;
surf(x1,x2,Y2_reshaped);
colormap(jet);
shading interp;
caxis([min(Y2_reshaped(:))-0.5*range(Y2_reshaped(:)), max(Y2_reshaped(:))]);
axis([-1 10 -1 10 0 0.25]);
xlabel('x_1');
ylabel('x_2');
zlabel('P(X|\omega_2)');
title('Probability Density Function for Class 2');
colorbar;
view(-30, 30); 

% Contour plots with decision boundaries
figure;
contour(X1,X2,Y1_reshaped, 20, 'b');
hold on;
contour(X1,X2,Y2_reshaped, 20, 'r');
grid on;
xlabel('x_1');
ylabel('x_2');
title('Gaussian Contours and Decision Boundaries');
axis([-1 10 -1 10]);
fprintf('Program paused. Press enter to continue.\n');
pause;

% Decision boundaries for different priors
P = [0.1, 0.25, 0.5, 0.75, 0.9];
colors = ['r', 'g', 'b', 'y', 'k'];

for i = 1:length(P)
    P1 = P(i);
    P2 = 1 - P1;

    y = (log(P1) - log(P2)) + (log(Y1_reshaped) - log(Y2_reshaped));
    y = reshape(y, length(x2), length(x1));
    
    contour(X1, X2, y, [0, 0], 'Color', colors(i), 'LineWidth', 1.5);
end

legend({'Class 1', 'Class 2', 'P1 = 0.1', 'P1 = 0.25', 'P1 = 0.5', 'P1 = 0.75', 'P1 = 0.9'}, 'Location', 'bestoutside');
hold off;
end
