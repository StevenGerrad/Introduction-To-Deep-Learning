%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Course:  Understanding Deep Neural Networks
% Teacher: Zhang Yi
% Student:
% ID:
%
% Lab 3 - BP algorithms
%
% Task 2: implement batch BP algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clear the workspace
clear

% define the activation function
f = @(s) 1 ./ (1 + exp(-s));
% define the derivative of activation function
df = @(s) f(s) .* (1 - f(s));

% prepare the training data set
data   = [1 0 0 1
          0 1 0 1]; % samples
labels = [1 1 0 0]; % labels
m = size(data, 2);

% choose parameters, initialize the weights
alpha = 0.1;
epochs = 50000;
w1 = randn(2,3);
w2 = randn(1,3);
J = zeros(1,epochs);

% loop until weights converge
for t = 1:epochs
    % reset the total gradients
    dw1 = 0;
    dw2 = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Your code BELOW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for all samples
    for i = 1:m

% forward calculation (invoke fc)
        a1 = data(:,i);
        [a2, z2] = fc(w1, [a1;1]);
        [a3, z3] = fc(w2, [a2;1]);

% calculate cost function
        J(t) = power((a3 - labels(i)), 2)/2;

% backwork calculation (invoke bc)
        delta3 = (a3 - labels(i))*df(z3);
        delta2 = bc(w2, z2, delta3);

% cumulate the total gradients
        dw1 = dw1 + [a1;1] * delta2(1:2);
        dw2 = dw2 + [a2;1] * delta3;        % --- ATTENTION --- 

% end for all samples
    end

% update weights
    w1 = w1 - alpha * dw1';
    w2 = w2 - alpha * dw2';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Your code ABOVE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% end loop
    if mod(t,100) == 0
        fprintf('%i/%i epochs: J=%.4f\n', t, epochs, J(t));
    end
end

% display the result
for i = 1:4
    a1 = data(:,i);
    [a2, z2] = fc(w1, [a1;1]);
    [a3, z3] = fc(w2, [a2;1]);
    fprintf('Sample [%i %i] (%i) is classified as %i.\n', data(1,i), data(2,i), labels(i), a3>0.5);
end