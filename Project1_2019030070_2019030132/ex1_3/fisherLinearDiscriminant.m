function v = fisherLinearDiscriminant(X1, X2)
    % Number of samples in each class
    m1 = size(X1, 1);
    m2 = size(X2, 1);

    % Calculate the mean vectors of each class
    mu1 = mean(X1);
    mu2 = mean(X2);

    % Calculate the scatter matrices for each class
    S1 = zeros(size(X1, 2));
    S2 = zeros(size(X2, 2));

    for i = 1:m1
        S1 = S1 + (X1(i, :) - mu1)' * (X1(i, :) - mu1);
    end

    for i = 1:m2
        S2 = S2 + (X2(i, :) - mu2)' * (X2(i, :) - mu2);
    end

    % Calculate the within-class scatter matrix
    Sw = S1 + S2;

    % Calculate the optimal direction for maximum class separation
    v = inv(Sw) * (mu2 - mu1)';

    % Normalize the direction vector to have unit norm
    v = v / norm(v);
end
