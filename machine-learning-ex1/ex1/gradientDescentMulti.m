function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    % X is a m x (n + 1) matrix
    % theta is a (n + 1) x 1 matrix
    % error is a m x 1 vector
    error = X * theta - y;
    
    % error is a m x 1 vector
    % X is a m x (n + 1) matrix
    % (error' * X) is a 1 x (n + 1) row vector
    % (error' * X)' is a (n + 1) x 1 column vector
    delta = 1 / m * (error' * X)';
    
    % theta is a (n + 1) x 1 matrix
    % alpha is a constant
    % delta is a (n + 1) x 1 matrix
    theta = theta - (alpha * delta);










    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
