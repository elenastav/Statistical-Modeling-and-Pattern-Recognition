function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.


% ADD YOUR CODE
[nSamples, nFeat]=size(X); 

% Initialize mu (mean) and sigma (standard deviation)
mu = zeros(1, nFeat);
sigma = zeros(1, nFeat);

for j=1:nFeat
 meanOfFeature=mean(X(:,j));
 stdOfFeature=std(X(:,j));
 % Update mu and sigma
 mu(j) = meanOfFeature;
 sigma(j) = stdOfFeature;
 X_norm(:,j)=(X(:,j)-meanOfFeature)/stdOfFeature;
end

% ============================================================

end