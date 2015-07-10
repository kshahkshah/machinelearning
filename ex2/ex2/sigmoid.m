function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% why, the input defines the output's size, no need to do this here.
% g = zeros(size(z));

% use element wise division and power raises
g = (1 ./ (1 + (e.^(-1 * z))));


end
