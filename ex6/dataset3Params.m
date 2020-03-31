function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

steps = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
maxAcc = 0.0;


% ====================== YOUR CODE HERE ======================
for i = 1 : length(steps)
    for j = 1 : length(steps)
        model = svmTrain(X, y, steps(i), @(x1, x2)gaussianKernel(x1, x2, steps(j)));
        acc = (mean(svmPredict(model, Xval) == yval));
        if(acc > maxAcc)
            C = steps(i);
            sigma = steps(j);
            maxAcc = acc;
        end
    end
end
% =========================================================================

end
