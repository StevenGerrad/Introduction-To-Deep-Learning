clear;
clc;

%% Step 1: Data Preparation 
 
% loading dataset
load mnist_small_matlab.mat
% trainData: a matrix with size of 28x28x10000   
% trainLabels: a matrix with size of 10x10000
% testData: a matrix with size of 28x28x2000
% testLabels: a matrix with size of 10x2000


train_size = 10000; % number of training samples
% input in the 1st layer 
X_train = reshape(trainData, 784, train_size); 


test_size = 2000; % number of testing samples
% external input in the 1st layer 
X_test = reshape(testData, 784, test_size); 

%% Step 2: Design Network Architecture
% define number of layers
L = 8; 
% define number of neurons in each layer 

% layer_size = [784 % number of neurons in 1st layer
%               256 % number of neurons in 2nd layer
%               128 % number of neurons in 3rd layer
%               64  % number of neurons in 4th layer
%               10];% number of neurons in 5th layer

% layer_size = [784 % number of neurons in 1st layer
%               10];% number of neurons in 2th layer

% layer_size = [784 % number of neurons in 1st layer
%               128 % number of neurons in 2rd layer
%               10];% number of neurons in 3th layer

layer_size = [784 % number of neurons in 1st layer
              512 % number of neurons in 2rd layer
              261 % number of neurons in 3rd layer
              128 % number of neurons in 4rd layer
              65  % number of neurons in 5rd layer
              34  % number of neurons in 6rd layer
              18  % number of neurons in 7rd layer
              10];% number of neurons in 8th layer

          
%% Step 3: Initial Parameters

% initialize weights in each layer with Gaussian distribution
for l = 1:L-1
    w{l} = 0.1 * randn(layer_size(l+1,1), sum(layer_size(l,:)));
end
 
alpha = 0.05; % initialize learning rate 

%% Step 4: Define Cost Function
% cost function is defined in cost.m

%% Step 5: Define Evaluation Index
% accuracy defined in accuracy.m

%% Step 6: Train the Network
J = []; % array to store cost of each mini batch
Acc = []; % array to store accuracy of each mini batch
max_epoch = 200; % number of training epoch
mini_batch = 100; % number of sample of each mini batch

figure % plot the cost
for iter=1:max_epoch
    % randomly permute the indexes of samples in training set
idxs = randperm(train_size); 
% for each mini-batch
for k = 1:ceil(train_size/mini_batch)
    % prepare internal inputs in 1st layer denoted by a{1}
    
    start_idx = (k-1)*mini_batch+1;          % start index of kth mini-batch
    end_idx = min(k*mini_batch, train_size); % end index of kth mini-batch
	a{1} = X_train(:,idxs(start_idx:end_idx));
    % prepare labels
    y = trainLabels(:, idxs(start_idx:end_idx));
    
    % forward computation
    for l=1:L-1
        [a{l+1}, z{l+1}] = fc(w{l}, a{l});
    end

    % Compute delta of last layer
    delta{L} = (a{L} - y).* a{L} .*(1-a{L}); %delta{L}={partial J}/{partial z^L} 

    % backward computation
    for l=L-1:-1:2
        delta{l} = bc(w{l}, z{l}, delta{l+1});
    end

    % update weight 
    for l=1:L-1
        % compute the gradient
        grad_w = delta{l+1} * a{l}'; 
        w{l} = w{l} - alpha*grad_w;
    end 

    % training cost on training batch
    J = [J 1/mini_batch*sum(cost(a{L}, y))];
    Acc =[Acc accuracy(a{L}, y)]; 
    % plot training error 
    plot(J);
    %hold on;
    %plot(Acc);
    pause(0.000001);
end
fprintf('finished iter:%d, J=%.4f, Acc=%.4f\n',iter,J(end),Acc(end));
end 
% end training
% plot accuracy

% figure
% hold on;
% plot(Acc);
% ylim([0 1]);

%% Step 7: Test the Network
%test on training set
a{1} = X_train;
for l = 1:L-1
  a{l+1} = fc(w{l}, a{l});
end
train_acc = accuracy(a{L}, trainLabels);
fprintf('Accuracy on training dataset is %f%%\n', train_acc*100);

%test on testing set
a{1} = X_test;
for l = 1:L-1
   a{l+1} = fc(w{l}, a{l});
end
test_acc = accuracy(a{L}, testLabels);
fprintf('Accuracy on testing dataset is %f%%\n', test_acc*100);

%% Step 8: Store the Network Parameters
save model.mat w layer_size

