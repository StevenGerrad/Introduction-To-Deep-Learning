%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Course:  Understanding Deep Neural Networks
% Teacher: Zhang Yi
% Student:
% ID:
%
% Lab 5 - Handwritten Digit to Speech Convertor
%
% Task: Design and train a neural netowrk to produce
%       standard speech according to input
%       handwritten digits in images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clear workspace and close plot windows
clear;
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Your code BELOW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% prepare the data set
load mnist_small_matlab.mat
input_size = 28 * 28;   % patch ��С
% prepare training data
train_size = size(trainLabels, 2);
X_train{1} = reshape(trainData, [], train_size);
X_train{2} = zeros(0, train_size);
X_train{3} = zeros(0, train_size);
X_train{4} = zeros(0, train_size);
X_train{5} = zeros(0, train_size);
% prepare testing data
test_size = size(testLabels, 2);
X_test{1} = reshape(trainData, [], test_size);
X_test{2} = zeros(0, test_size);
X_test{3} = zeros(0, test_size);
X_test{4} = zeros(0, test_size);
X_test{5} = zeros(0, test_size);
%% prepare standard speech audio
sample_rate = 4000;
audio = zeros(2983, 10);    % ���ȵ�֪��Ƶ�ļ�Ϊ2983-dim
for i = 1:10
    [audio(:, i), sample_rate] = audioread(fullfile('audio', sprintf('%d.wav',i-1)));
    soundsc( audio(:, 1), sample_rate );
    pause(1);
end
audio = (audio + 1)/2;

%% choose parameters
alpha = 0.1;
max_iter = 300;
mini_batch = 100;
layer_size = [input_size 512
                       0 512
                       0 1024
                       0 2048
                       0 2983];
L = size(layer_size, 1);

%% define network architecture
sigm = @(s) 1 ./ (1 + exp(-s));
dsigm = @(s) sigm(s) .* (1 - sigm(s));
fs = {[], sigm, sigm, sigm, sigm, sigm, sigm, sigm};
dfs = {[], dsigm, dsigm, dsigm, dsigm, dsigm, dsigm, dsigm};

%% initialize weights
w = cell(L-1, 1);
for l = 1:L-1
    w{l} = (rand(layer_size(l+1, 2), sum(layer_size(l, :))) * 2 - 1) * ...
            sqrt( 6/(layer_size(l+1, 2) + sum(layer_size(l,:))) );
end

%% train
J = [];
x = cell(L, 1);
a = cell(L, 1);
z = cell(L, 1);     % TODO: ��ô����(L, 1)��
delta = cell(L, 1);
for iter = 1 : max_iter
    ind = randperm(train_size);
    % for each mini-batch(��)
    for k = 1 : ceil(train_size / mini_batch)
        % prepare internal inputs
        a{1} = zeros(layer_size(1, 2), mini_batch);
        % prepare external inputs
        for l = 1:L
            x{l} = X_train{l}(:, ...
                        ind((k-1)*mini_batch+1:min(k*mini_batch, train_size)));
        end
        % prepare labels
        [~, ind_label] = max( trainLabels(:, ind( ...
                                        (k-1)*mini_batch+1 : min( k*mini_batch, train_size )...
                                        )));
        % prepare targets
        y = audio(:, ind_label);
        
        % batch forward computation
        for l = 1:L-1
            [a{l+1}, z{l+1}] = fc(w{l}, a{l}, x{l}, fs{l+1});
        end
        % cost function and error
        % Jÿ����չ������ 0.5 * 1/( ������*���ƽ���� )
        J = [J 1/2/mini_batch*sum((a{L}(:)-y(:)).^2)];
        delta{L} = (a{L} - y) .* dfs{L}(z{L});
        % batch backward computation
        for l = L-1:-1:2
            delta{l} = bc(w{l}, z{l}, delta{l+1}, dfs{l});
        end
        % update weight
        for l = 1:L-1
            gw = delta{l+1} * [x{l}; a{l}]' / mini_batch;
            w{l} = w{l} - alpha * gw;
        end
    end
    if mod(iter, 1) == 0
        fprintf('%i/%i epochs: J = %.4f\n', iter, max_iter, J(end));
    end
end

%% save model
save model.mat w layer_size J

%% display/listen to some results pairs
figure;
plot(J);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Your code ABOVE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
