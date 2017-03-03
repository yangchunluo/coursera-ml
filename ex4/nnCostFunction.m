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
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Map y from label vector to binary vectors of 1's and 0's.
Y = [];
for k = 1:num_labels
    Y = [Y (y == k)];
end
% Y is m*num_labels

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

A1 = [ones(m,1) X];
Z2 = A1 * Theta1';
A2 = [ones(m, 1) sigmoid(Z2)];
Z3 = A2 * Theta2';
A3 = sigmoid(Z3);
J = sum(sum(-Y .* log(A3) - (1 - Y) .* log(1 - A3))) / m + ...
    (sum(sum(Theta1(:,2:end) .^ 2)) + sum(sum(Theta2(:,2:end) .^ 2))) * lambda / 2 / m;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

for t = 1:m
    % 1. forward propagation
    a1 = [1, X(t,:)]';              % column vector of input_layer_size + 1
    z2 = Theta1 * a1;               % column vector of hidden_layer_size
    a2 = [1; sigmoid(z2)];          % column vector of hidden_layer_size + 1
    z3 = Theta2 * a2;               % column vector output_layer_size
    a3 = sigmoid(z3);               % column vector output_layer_size

    % 2. delta for layer 3
    d3 = a3 - Y(t,:)';              % column vector output_layer_size

    % 3. delta for layer 2
    d2 = Theta2(:,2:end)' * d3 .* sigmoidGradient(z2);
    %       h*o             o*1     % column vector of hidden_layer_size

    % 4. accumulation
    Theta2_grad += d3 * a2';        % matrix of output * (hidden + 1)
    %              o*1  1*(h+1)
    Theta1_grad += d2 * a1';        % matrix of hidden * (input + 1)
    %              h*1  1*(i+1)
end

Theta1_grad /= m;
Theta1_grad += [zeros(size(Theta1, 1), 1) Theta1(:,2:end)] * lambda / m;
%              replace 1st column of Theta1 with 0s

Theta2_grad /= m;
Theta2_grad += [zeros(size(Theta2, 1), 1) Theta2(:,2:end)] * lambda / m;
%              replace 1st column of Theta2 with 0s

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
