function [eigenval, eigenvec, order] = myPCA(X)
%PCA Run principal component analysis on the dataset X
%   [eigenval, eigenvec, order] = mypca(X) computes eigenvectors of the autocorrelation matrix of X
%   Returns the eigenvectors, the eigenvalues (on diagonal) and the order 
%

% Useful values
[m, n] = size(X);

% Make sure each feature from the data is zero mean
X_centered = X - mean(X);

% Compute the covariance 
covariance_matrix = (1/m) .* X_centered' * X_centered;

% Compute eigenvectors and eigenvalues
[eigenvec, eigenval] = eig(covariance_matrix);

% Convert eigenvalues matrix to a vector
eigenval = diag(eigenval);

% Sort eigenvectors and eigenvalues
[eigenval, order] = sort(eigenval, 1, 'descend');
eigenvec = eigenvec(:, order);

end
