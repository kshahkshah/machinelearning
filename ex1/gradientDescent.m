function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % for the sake of it multiply by X(:, 1) which is always one

    hypothesis = (theta(1) * X(:, 1)) + (theta(2) * X(:, 2));

    theta_zero = theta(1) - ((alpha/m)*((sum((hypothesis - y) .* X(:, 1)))));
    theta_one  = theta(2) - ((alpha/m)*((sum((hypothesis - y) .* X(:, 2)))));

    theta = [theta_zero; theta_one];

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    % fprintf('printing: %f \n', J_history(iter));

end

end
