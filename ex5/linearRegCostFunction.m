function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

m = length(y);

J = 0;
grad = zeros(size(theta));


% ====================== YOUR CODE HERE ======================
h = X * theta;
J = 1 / (2*m) * sum((h - y).^2) + lambda / (2*m) * sum(theta(2:end).^2);

grad = 1 / m * X' * (h - y);
grad(2:end) = grad(2:end) + lambda / m * theta(2:end);
% =========================================================================


grad = grad(:);

end
