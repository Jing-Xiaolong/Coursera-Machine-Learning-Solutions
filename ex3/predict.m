function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);


% ====================== YOUR CODE HERE ======================
X = [ones(m,1) X];          % m x (n + 1) -> 5000x401
z2 = X * Theta1';           % Theta1 - 25 x 401, z2 - 5000x25
a2 = [ones(m,1) 1./ (1 + exp(-z2))];        % a2 - 5000x26
z3 = a2 * Theta2';          % Theta2 - 10 x 26,  z3 - 5000x10
a2 = 1 ./ (1 + exp(-z3));
[~,p] = max(a2, [], 2);
% =========================================================================


end
