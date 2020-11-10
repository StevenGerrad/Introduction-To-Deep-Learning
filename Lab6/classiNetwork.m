
clear;
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   the neural network of classifier BELOW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 1: Data Preparation 
load labledIm.mat
% IT INCLUDE bigcat_image, lt_train_label, lt_test_data, lt_test_label 

% loading dataset
train_size = size(lt_train_data, 2); % number of training samples
% input in the 1st layer
X_train = lt_train_data;
trainLabels = lt_train_label;

test_size = size(lt_test_data, 2); % number of testing samples
% external input in the 1st layer 
X_test = lt_test_data;
testLabels = lt_test_label;

%% Step 2: Design Network Architecture
% define number of layers
L = 5; 
% define number of neurons in each layer 
layer_size = [500 0 % number of neurons in 1st layer
              0 126 % number of neurons in 2nd layer
              0 32 % number of neurons in 3rd layer
              0 8  % number of neurons in 4th layer
              0 2];% number of neurons in 5th layer

%% Step 3: Initial Parameters
% initialize weights in each layer with Gaussian distribution
w = cell(L-1, 1);
for l = 1:L-1
    w{l} = randn(layer_size(l+1,2), sum(layer_size(l,:)))* ...
            sqrt(6/(layer_size(l+1,2) + sum(layer_size(l,:))));
%     w{l} = (rand(layer_size(l+1, 2), sum(layer_size(l, :)))*2-1) * ...
%             sqrt(2/(layer_size(l+1, 2) + sum(layer_size(l,:))));
    w_mean{l} = [];
    w_std{l} = [];
end
alpha = 0.05; % initialize learning rate

%% Step 4: Define Cost Function


%% Step 5: Define Evaluation Index
% accuracy defined in accuracy.m
sigm = @(s) 1 ./ (1 + exp(-s));
dsigm = @(s) sigm(s) .* (1 - sigm(s));
relu = @(s) max(0, s);
drelu = @(s) s.*(s>0);
fs = {[], sigm, sigm, sigm, sigm, sigm, sigm, sigm};
dfs = {[], dsigm, dsigm, dsigm, dsigm, dsigm, dsigm, dsigm};

%% Step 6: Train the Network
J = []; % array to store cost of each mini batch
Acc = []; % array to store accuracy of each mini batch
max_epoch = 400; % number of training epoch
mini_batch = 40; % number of sample of each mini batch

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
            [a{l+1}, z{l+1}] = fc(w{l}, a{l}, [], fs{l+1});
        end
        % Compute delta of last layer
        delta{L} = (a{L} - y).* a{L} .*(1-a{L}); %delta{L}={partial J}/{partial z^L} 
        % backward computation
        for l=L-1:-1:2
            delta{l} = bc(w{l}, z{l}, delta{l+1}, dfs{l}, 0);
        end
        % update weight 
        for l=1:L-1
            % compute the gradient
            grad_w = delta{l+1} * a{l}';
            w{l} = w{l} - alpha*grad_w;
            w_mean{l} = [w_mean{l} mean(w{l}(:))];
            w_std{l} = [w_std{l} std(w{l}(:))];
        end
        % training cost on training batch
        J = [J 1/mini_batch*sum(1/2 * sum((a{L} - y).^2))];
        Acc =[Acc accuracy(a{L}, y)]; 
        % plot training error 
        plot(J);
        % hold on;
        % plot(Acc);
        pause(0.000001);
    end
    fprintf('finished iter:%d, J=%.4f, Acc=%.4f\n',iter,J(end),Acc(end));
end 

w_classi = w;
layer_size_classi = layer_size;
save model_classi.mat w_classi layer_size_classi alpha mini_batch

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   the neural network of classifier ABOVE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% combine the two networks
sparse = 3;
load model_auto_hope3_spa.mat layer_size w
w = w(1:sparse-1);
w_final = w;
for l = 1:size(w_classi, 1)
    w_final{l+sparse-1} = w_classi{l};
end
layer_size_final = [layer_size(1:sparse-1,:) ;layer_size_classi];

%% test the network with the testing set
fs = {[], sigm, sigm, sigm, sigm, sigm, sigm, sigm, sigm, sigm, sigm};
L = size(layer_size_final, 1);
x = cell(L, 1);
a = cell(L, 1);
z = cell(L, 1);
%test on training set
a{1} = zeros(layer_size_final(1, 2), size(bigcat_image, 2));
x{1} = bigcat_image;
for l = 1:L-1
    [a{l+1}, z{l+1}] = fc(w_final{l}, a{l}, x{l}, fs{l+1});
end
train_acc = accuracy(a{L}, trainLabels);
fprintf('Accuracy on training dataset is %f%%\n', train_acc*100);

%test on testing set
a{1} = zeros(layer_size_final(1, 2), test_size);
x{1} = X_test;
for l = 1:L-1
    [a{l+1}, z{l+1}] = fc(w_final{l}, a{l}, x{l}, fs{l+1});
end
test_acc = accuracy(a{L}, testLabels);
fprintf('Accuracy on testing dataset is %f%%\n', test_acc*100);
