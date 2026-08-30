function A = myLDA(Samples, Labels, NewDim)
% Input:    
%   Samples: The Data Samples 
%   Labels: The labels that correspond to the Samples
%   NewDim: The New Dimension of the Feature Vector after applying LDA
    
	[NumSamples , NumFeatures] = size(Samples);
    NumLabels = length(Labels);
    if(NumSamples ~= NumLabels) then
        fprintf('\nNumber of Samples are not the same with the Number of Labels.\n\n');
        exit
    end
    Classes = unique(Labels);
    NumClasses = length(Classes); 
    Sw = zeros(NumFeatures,NumFeatures);
    Sb = zeros(NumFeatures,NumFeatures);

    %For each class i
	%Find the necessary statistics
    for i = 1 : NumClasses	
		% Get samples for class i
        class_samples = Samples(Labels == Classes(i), :);
		%Calculate the Class Prior Probability
        P(i) = size(class_samples, 1)/NumSamples;
		%Calculate the Class Mean 
        mu(i,:) = mean(class_samples , 1); 
		%Calculate the Within Class Scatter Matrix
        Sw = Sw + P(i)*(class_samples'*class_samples);
    end

    %Calculate the Global Mean
	m0= mean(mu);
    
    %Calculate the Between Class Scatter Matrix
    for i = 1 : NumClasses	
		Sb = Sb + P(i)*((mu(i,:)-m0)'*(mu(i,:)-m0));
    end
    
    %Eigen matrix EigMat=inv(Sw)*Sb
    EigMat = inv(Sw)*Sb;
    
    %Perform Eigendecomposition
    [V, D] = eig(EigMat);

    [~, ind] = sort(diag(D), 'descend');
    V = V(:, ind);
    %Select the NewDim eigenvectors corresponding to the top NewDim
    %eigenvalues (Assuming they are NewDim<=NumClasses-1)
	%% You need to return the following variable correctly.
	A=zeros(NumFeatures,NewDim);  % Return the LDA projection vectors
    A = V(:, 1:NewDim);
    
    
    
    
    
