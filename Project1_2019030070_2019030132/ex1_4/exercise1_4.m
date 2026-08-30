clear; close all; clc;

m1 = [2; 3;];
m2 = [4; 4;];
s1 = [2 0.5; 0.5 1;];
s2 = [1.5 -0.3; -0.3 0.8;];
s1_2 = [1.2 0.4; 0.4 1.2;];

Bayes(m1, m2, s1, s2);
fprintf('Program paused. Press enter to continue.\n');
pause;

Bayes(m1, m2, s1_2, s1_2);

