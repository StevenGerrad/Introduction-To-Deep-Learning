% function [] = mnistReader()

% === Introduction ===

% The MNIST dataset is a dataset of handwritten digits, comprising 60 000 
% training examples and 10 000 test examples. The dataset can be downloaded 
% from http://yann.lecun.com/exdb/mnist/.

% === Usage ===

clear;
close all;

%% read and resize the dataset
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
trainData = loadMNISTImages('dataset/train-images-idx3-ubyte');
trainLabels = loadMNISTLabels('dataset/train-labels-idx1-ubyte');
testData = loadMNISTImages('dataset/t10k-images-idx3-ubyte');
testLabels = loadMNISTLabels('dataset/t10k-labels-idx1-ubyte');

train_len = size(trainLabels, 1);
train_temp = zeros(10, train_len);
for i = 1:train_len
    train_temp(trainLabels(i)+1, i) = 1;
end
trainLabels = train_temp;

test_len = size(testLabels, 1);
test_temp = zeros(10, test_len);
for i = 1:test_len
    test_temp(testLabels(i)+1, i) = 1;
end
testLabels = test_temp;

save fash_mnist.mat trainData trainLabels testData testLabels
clear train_temp test_temp

%% make smaller dataset
ind = randperm(train_len);
trainData = trainData(:, ind(1:20000));
trainLabels = trainLabels(:, ind(1:20000));
ind = randperm(test_len);
testData = testData(:, ind(1:5000));
testLabels = testLabels(:, ind(1:5000));
save fash_mnist_25k.mat trainData trainLabels testData testLabels
% end

