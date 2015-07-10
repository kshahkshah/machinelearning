function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
best_cost = 1;

steps = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

prediction_errors_matrix = zeros(length(steps), length(steps));

for cIndex=1:length(steps)
  for sigmaIndex=1:length(steps)
    currentC     = steps(cIndex);
    currentSigma = steps(sigmaIndex);

    fprintf('Training and predicting error for C %f and sigma %f.\n', currentC, currentSigma);

    model = svmTrain(X, y, currentC, @(x1, x2) gaussianKernel(x1, x2, currentSigma));
    prediction = svmPredict(model, Xval);

    current_cost = mean(double(prediction ~= yval));

    fprintf('Accuracy is %f.\n', current_cost);

    if current_cost < best_cost

      fprintf('This is the best accuracy thus far\n');

      best_cost = current_cost;
      C = currentC;
      sigma = currentSigma;
    end

  end
end



% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
