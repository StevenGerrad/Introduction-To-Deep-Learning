%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Course:  Understanding Deep Neural Networks
% Teacher: Zhang Yi
% Student:
% ID:
%
% Lab 6 - Big Cat Recognition
%
% Task: In this example, the labeled training data
%       are insufficient to train a classifier by
%       using BP directly. However, a good
%       classifier can be developed by using the
%       autoencoder method.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%       说明：由于代码阶段较多，此文件仅作初步构想用
%       具体顺序为：dataPrepare->autoNetwork->labledPrepare->classiNetwork
%       a_w_show: 显示5、7层structure的a、w分布
%       J_w_show: 显示误差与w均值、标准差
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear workspace and close plot windows
clear;
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Your code BELOW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Read and prepare the data
% Resize each image to a certain size to match the input to your network.
if exist('random_data.mat','file')==0
   dataPrepare();
end

%% Train the autoencoder by using unlabeled data (the unlabeled set)
if exist('model_auto.mat','file')==0
   % 为便于调试将其改为脚本了
   autoNetwork();
end

%% Remove the layers behind sparse representation layer after training
% 将sparse之后的层统统移除

%% Form a new data set in sparse representation layer by using the labeled data set (the trianing set)
if ~exist('random_data.mat','file')==0
   labledPrepare();
end

%% Form a new training data set for supervised network (the encoded training set and its labels)
load labledIm.mat

image = im2double(image);
trainData = image;
trainLabels = image;

input_size = 80*80;   % patch
% prepare data
train_size = size(image, 2);
X_train{1} = reshape(trainData, [], train_size);
X_train{2} = zeros(0, train_size);
X_train{3} = zeros(0, train_size);      % 实际上只用三个就行了

len = size(bigcat_image, 2);
x = cell(L, 1);
a = cell(L, 1);
z = cell(L, 1);
for k = 1 : len
    % prepare internal inputs
    a{1} = zeros(layer_size(1, 2), len);
    % prepare external inputs
    for l = 1:L
        x{l} = X_train{l};
    end
    % batch forward computation
    % 相当于将labeled数据都进行压缩
    for l = 1:L-1
        [a{l+1}, z{l+1}] = fc(w{l}, a{l}, x{l}, fs{l+1});
    end
end

%% Training the network by using the new training data set

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   the neural network of classifier BELOW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 1: Data Preparation 
load labledIm.mat
% IT INCLUDE bigcat_image, lt_train_label, lt_test_data, lt_test_label 

% loading dataset
train_size = size(bigcat_image, 2); % number of training samples
% input in the 1st layer
X_train = bigcat_image;
trainLabels = lt_train_label;

test_size = size(lt_test_data, 2); % number of testing samples
% external input in the 1st layer 
X_test = lt_test_data;
testLabels = lt_test_label;

% Step 2: Design Network Architecture
% define number of layers
L = 5; 
% define number of neurons in each layer 
layer_size = [100 % number of neurons in 1st layer
              38 % number of neurons in 2nd layer
              14 % number of neurons in 3rd layer
              5  % number of neurons in 4th layer
              2];% number of neurons in 5th layer

% Step 3: Initial Parameters
% initialize weights in each layer with Gaussian distribution
for l = 1:L-1
    % w{l} = 0.1 * randn(layer_size(l+1,1), sum(layer_size(l,:)));
    w{l} = (rand(layer_size(l+1, 2), sum(layer_size(l, :)))*2-1) * ...
            sqrt(2/(layer_size(l+1, 2) + sum(layer_size(l,:))));
end
alpha = 0.05; % initialize learning rate

% Step 4: Define Cost Function
% Step 5: Define Evaluation Index
% accuracy defined in accuracy.m

% Step 6: Train the Network
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
save model_classi.mat w_classi layer_size_classi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   the neural network of classifier ABOVE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% combine the two networks
w_final = {w_auto w_classi};
layer_size_final = {layer_size_auto layer_size_classi};

%% test the network with the testing set

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Your code ABOVE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
