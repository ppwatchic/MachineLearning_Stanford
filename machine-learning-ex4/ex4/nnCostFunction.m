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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Step 1 of Part 1: extend 1's column to X 
X_tmp = [ones(m,1) X];
% BTW, generate row vector yy: [1,2,3,...,num_labels]' 
yy = ones(num_labels,1);
for i = 1:num_labels
    yy(i) = i;
end
% Step 2 of Part 1: compute the hypothesis function 
% Step 2.1: first get the activation for Hidder Layer using Theta1 
z2 = X_tmp * Theta1';
a2 = sigmoid(z2);   % after this, add 1's column to a2 
a2 = [ones(m,1) a2];
% Step 3 of Part 1: similar to Step 2 
z3 = a2 * Theta2';
h_x = sigmoid(z3);    % [5000 10]
% Step 4 of Part 1: compute the cost, iterate through each output 
cost_i = 0;
yi_colvec = zeros(num_labels,1);
for i = 1:m
    yi_colvec = (yy==y(i));     % [10 1] column vector
    cost_i = cost_i -log(h_x(i,:))*yi_colvec - log(1-h_x(i,:)) *(1-yi_colvec);
end
% Step 5: compute the cost without regularization
J = cost_i / m;
% Step 6 of Part 1: Regularized cost function
Theta1_tmp = Theta1(:,(2:input_layer_size+1));      % [25 400]
Theta2_tmp = Theta2(:,(2:hidden_layer_size+1));     % [10 25]

Theta1_tmp2 = 0;
for i = 1:hidden_layer_size
    Theta1_tmp2 = Theta1_tmp2 + Theta1_tmp(i,:)*Theta1_tmp(i,:)';
end

Theta2_tmp2 = 0;
for i = 1:num_labels
    Theta2_tmp2 = Theta2_tmp2 + Theta2_tmp(i,:)*Theta2_tmp(i,:)';
end

J = J + (Theta1_tmp2+Theta2_tmp2)* lambda/(2*m);

%% Part 2: Backpropagation

for i = 1:m
    % step 1: already have: X_tmp, z2, a2, z3, a3(which is h_x) from Part 1
    % step 2: 
    Delta_3 = zeros(num_labels,1);
    yi_colvec = zeros(num_labels,1);
    for k = 1:num_labels
        yi_colvec = (yy==y(i));         % yi_colvec is [10 1] logical array 
        Delta_3(k) = h_x(i,k) - yi_colvec(k);
    end
    % step 3: for hidden layer 
    Delta_2 = zeros(hidden_layer_size+1,1);
    tmp = zeros(1,hidden_layer_size+1);
    tmp(hidden_layer_size+1) = -10000;
    tmp(1:hidden_layer_size) = z2(i,:);
    Delta_2 = (Theta2'*Delta_3).*sigmoidGradient(tmp');         % [26 1] 
    % skip first item in Delta_2 
    Delta_2 = Delta_2(2:end);                                   % [25 1]
    Delta_2 = Delta_2 + 
end





% source from:
% https://github.com/schneems/Octave/blob/master/mlclass-ex4/mlclass-ex4/nnCostFunction.m 
% I = eye(num_labels);
% Y = zeros(m, num_labels);
% for i=1:m
%   Y(i, :)= I(y(i), :);
% end
% 
% 
% 
% A1 = [ones(m, 1) X];
% Z2 = A1 * Theta1';
% A2 = [ones(size(Z2, 1), 1) sigmoid(Z2)];
% Z3 = A2*Theta2';
% H = A3 = sigmoid(Z3);
% 
% 
% penalty = (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:,2:end).^2, 2)));
% 
% J = (1/m)*sum(sum((-Y).*log(H) - (1-Y).*log(1-H), 2));
% J = J + penalty;
% 
% Sigma3 = A3 - Y;
% Sigma2 = (Sigma3*Theta2 .* sigmoidGradient([ones(size(Z2, 1), 1) Z2]))(:, 2:end);
% 
% 
% Delta_1 = Sigma2'*A1;
% Delta_2 = Sigma3'*A2;
% 
% 
% Theta1_grad = Delta_1./m + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
% Theta2_grad = Delta_2./m + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients [25 401]  [10 26]
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
