% function [] = autoNetwork()
clear;
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   use the previous model of lab5 structure BELOW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load random_data.mat image
load random_data_1k.mat trainIm1k
% load random_data_1h.mat trainIm1h
image = im2double(trainIm1k);

% histogram(image,256)
% image = double(trainIm1k)/255;
% image = image*2-1;
trainData = image;
trainLabels = image;

input_size = 80 * 80;   % patch
%% prepare training data
% train_size = size(trainLabels, 2);
train_size = size(image, 2);
X_train{1} = reshape(trainData, [], train_size);
X_train{2} = zeros(0, train_size);
X_train{3} = zeros(0, train_size);
X_train{4} = zeros(0, train_size);
X_train{5} = zeros(0, train_size);
X_train{6} = zeros(0, train_size);
X_train{7} = zeros(0, train_size);

%% prepare standard internal data
% 自编码中internal data无需处理

%% choose parameters
alpha = 0.3;
max_iter = 600;
mini_batch = 40;
% layer_size = [input_size 1000
%                        0 2000
%                        0 500
%                        0 30
%                        0 500
%                        0 2000
%                        0 input_size];
layer_size = [input_size 1000
                       0 2000
                       0 500
                       0 2000
                       0 input_size];
sparse = 3;     % define the number of layer(l) that use β
% TODO: 这前面写个0是怎么意思----哦，是每一层的external input(x)
L = size(layer_size, 1);

%% define network architecture
sigm = @(s) 1 ./ (1 + exp(-s));
dsigm = @(s) sigm(s) .* (1 - sigm(s));
relu = @(s) max(0, s);
drelu = @(s) s.*(s>0);
fs = {[], sigm, sigm, sigm, sigm, sigm, sigm, sigm};
dfs = {[], dsigm, dsigm, dsigm, dsigm, dsigm, dsigm, dsigm};
% fs = {[], relu, relu, relu, relu, relu, relu, relu};
% dfs = {[], drelu, drelu, drelu, drelu, drelu, drelu, drelu};

%% initialize weights

w = cell(L-1, 1);
for l = 1:L-1
    w{l} = (randn(layer_size(l+1, 2), sum(layer_size(l, :)))) * ...
        sqrt(6/( layer_size(l+1, 2) + sum(layer_size(l,:)) ));
%     w{l} = (rand(layer_size(l+1, 2), sum(layer_size(l, :)))*2-1) * 0.01;
    % shit，这怎么抄都能抄错
    % TODO: 权值矩阵初始化规则...为什么呢？
    % 这篇博客讲了一些-- https://blog.csdn.net/cherry_yu08/article/details/79116862
    % 权重初始化令其方差为 2 / (n[l-1]+n[l])
    % http://www.deeplearning.net/tutorial/mlp.html#mlp
    w_mean{l} = [];
    w_std{l} = [];
end

%% train
J = [];
J_y = [];
J_a_l = [];
Acc = [];

x = cell(L, 1);
a = cell(L, 1);
z = cell(L, 1);
delta = cell(L, 1);
beta = 0.1;         % the parameter β used for layer(l)

% figure      % plot the J
%% loop
for iter = 1 : max_iter
    ind = randperm(train_size);
    % for each mini-batch()
    for k = 1 : ceil(train_size / mini_batch)
        % prepare internal inputs
        a{1} = zeros(layer_size(1, 2), mini_batch);
        % prepare external inputs
        for l = 1:L
            % 生成一段序列，每次长度为mini_batch
            x{l} = X_train{l}(:, ...
                        ind((k-1)*mini_batch + 1:min(k*mini_batch, train_size)));
        end
        % prepare labels
        % max 函数返回 [每一列的最大值, 每列最大值的行号]
        % [~, ind_label] = max( trainLabels(:, ind((k-1)*mini_batch + 1:min(k*mini_batch, train_size))));
        % 为什么要写一个max??? 没道理呀
        
        % prepare targets
        y = trainLabels(:, ind((k-1)*mini_batch + 1:min(k*mini_batch, train_size)));
        
        % batch forward computation
        for l = 1:L-1
            [a{l+1}, z{l+1}] = fc(w{l}, a{l}, x{l}, fs{l+1});
        end
        % cost function and error
        J_y = [J_y 1/2/mini_batch*sum((a{L}(:)-y(:)).^2)];
        J_a_l = [J_a_l 1/mini_batch*beta*sum(sum(a{sparse}(a{sparse}>0)))];      % *** ATTENTION *** Fs()
        J = [J J_y(end)+J_a_l(end)];
        Acc = [Acc accuracy(a{L}, y)];
        delta{L} = (a{L} - y) .* dfs{L}(z{L});
        
        % batch backward computation
        for l = L-1:-1:2
            if l == sparse
                delta{l} = bc(w{l}, z{l}, delta{l+1}, dfs{l}, beta);
            else
                delta{l} = bc(w{l}, z{l}, delta{l+1}, dfs{l}, 0);
            end
%             delta{l} = bc(w{l}, z{l}, delta{l+1}, dfs{l}, 0);
        end
        % update weight
        for l = 1:L-1
            gw = delta{l+1} * [x{l}; a{l}]' / mini_batch;
            w{l} = w{l} - alpha * gw;
            w_mean{l} = [w_mean{l} mean(w{l}(:))];
            w_std{l} = [w_std{l} std(w{l}(:))];
        end
%         subplot(2,1,1); plot(J_y);
%         subplot(2,1,2); plot(J_a_l);
%         pause(0.000001);
    end
    if mod(iter, 1) == 0
        fprintf('%i/%i epochs: J_y = %.4f J_a_l = %.4f J = %.4f\n', iter, max_iter, J_y(end), J_a_l(end), J(end));
        
%         fprintf('%i/%i epochs: J_y = %.4f \n', iter, max_iter, J_y(end));

%         fprintf('w{}_mean: ');
%         for l = 1:L-1
%             fprintf('%.3f ',w_mean{l}(end));
%         end
%         fprintf('w{}_std: ');
%         for l = 1:L-1
%             fprintf('%.3f ',w_std{l}(end));
%         end
%         fprintf('\n');
    end
end

% hold on;
% plot(Acc);

w_auto = w;
J_y_auto = J_y;
J_a_l_auto = J_a_l;
layer_size_auto = layer_size;
%% save model
save model_auto_addsp.mat w_auto layer_size_auto J_y_auto J_a_l_auto sparse

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   use the previous model of lab5 structure BELOW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% end

