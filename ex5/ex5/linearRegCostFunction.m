function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% I'm going to do this with matrices by blanking out the first lambda
adjusted_lambda = ones(size(X, 2), 1) * lambda;
adjusted_lambda(1, 1) = 0;

J = (sum((X * theta - y).^2) + (sum(theta.^2 .* adjusted_lambda))) / (2 * m);
grad = sum(((X * theta - y) .* X)/m) + ((adjusted_lambda ./ m) .* theta)';

grad = grad(:);

end
