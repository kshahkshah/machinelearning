function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% yeah, holy shit thats long
% X * theta is the parameterized inputs, we pass to sigmoid
% then we apply both to this neutralized/counter balanced cost function for y = 0 ; y = 1
J = (1/m) * sum(((-1 .* y) .* log(sigmoid(X * theta))) - ((1 - y) .* (log(1 - sigmoid(X * theta)))));

for feature = 1:size(grad, 1)
  % X(:, feature) is just the feature in question at the moment
  grad(feature) = (1/m) * sum((sigmoid(X * theta) - y) .* X(:, feature));
end

% =============================================================

end
