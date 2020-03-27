function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.


% Theta1 - 25x401,  Theta2 - 10x26
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% ====================== YOUR CODE HERE ======================

% -------------------------------------------------------------
% Part 1: Feedforward the neural network and return the cost
X = [ones(m,1) X];              % X  - 5000x401
label = zeros(m, num_labels);   % label - 5000x10
for i = 1 : m
    label(i, y(i)) = 1; 
end

z2 = X * Theta1';   % z2 - 5000x401x(25x401)' = 5000x25
a2 = sigmoid(z2);   % a2 - 5000x25
a2 = [ones(m,1) a2];% a2 - 5000x26
z3 = a2 * Theta2';  % z3 - 5000x26x(10x26)' = 5000x10
a3 = sigmoid(z3);   % z3 - 5000x10
loss = -1/m * (label .* log(a3) + (1 - label) .* log(1 - a3));
J = sum(loss(:));

Theta1Sq = Theta1(:, 2:end) .^ 2;
Theta2Sq = Theta2(:, 2:end) .^ 2;
J = J + lambda / (2*m) * (sum(Theta1Sq(:)) + sum(Theta2Sq(:)));

% -------------------------------------------------------------
% Part 2: Implement the backpropagation algorithm to compute the gradients
delta3 = a3 - label;                            % delta3 - 5000x10
Theta2_grad = 1/m * delta3' * a2;               % (5000x10)' x 5000x26 = 10x26

z2 = [ones(m,1) z2];
delta2 = delta3 * Theta2 .* sigmoidGradient(z2);% 5000x10 x 10x26 = 5000x26
delta2 = delta2(:,2:end);
Theta1_grad = 1/m * delta2' * X;                % (5000x25)' x 5000x401 = 25x401

% -------------------------------------------------------------
% Part 3: Implement regularization with the cost function and gradients.
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2(:,2:end);

% =========================================================================


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
