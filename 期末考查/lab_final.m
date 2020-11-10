
% clear workspace and close plot windows
clear;
close all;

%% prepare the data set

load fash_mnist.mat
input_size=28*28;
% prepare training data
train_size=size(trainLabels,2);
X_train{1}=reshape(trainData,[],train_size);
X_train{2}=zeros(0,train_size);
X_train{3}=zeros(0,train_size);
X_train{4}=zeros(0,train_size);
X_train{5}=zeros(0,train_size);
X_train{6}=zeros(0,train_size);
X_train{7}=zeros(0,train_size);
% prepare testing data
test_size = size(testLabels, 2);
X_test{1} = reshape(testData, [], test_size);
X_test{2} = zeros(0, test_size);
X_test{3} = zeros(0, test_size);
X_test{4} = zeros(0, test_size);
X_test{5} = zeros(0, test_size);
X_test{6} = zeros(0, test_size);
X_test{7} = zeros(0, test_size);

%% prepare standard speech audio
% Y_train = trainLabels;
% Y_test = testLabels;

%% choose parameters
alpha=0.05;
max_iter=600;
mini_batch=40;
% layer_size=[input_size 56
%                      0 270
%                      0 90
%                      0 30
%                      0 10];
layer_size=[input_size 56
                     0 401
                     0 191
                     0 92
                     0 44
                     0 21
                     0 10];
L=size(layer_size,1);
 
 %% define network architecture
sigm=@(s) 1./(1+exp(-s));
dsigm=@(s) sigm(s).*(1-sigm(s));
relu = @(s) max(0, s);
drelu = @(s) s.*(s>0);
lin = @(s) s;
dlin=@(s) 1;

fs={[],sigm,sigm,sigm,sigm,sigm,sigm,sigm};
dfs={[],dsigm,dsigm,dsigm,dsigm,dsigm,dsigm};
% fs = {[], relu, relu, relu, relu, relu, relu, relu};
% dfs = {[], drelu, drelu, drelu, drelu, drelu, drelu, drelu};

%% initialize weights
w=cell(L-1,1);
for l=1:L-1
%     w{l}=(rand(layer_size(l+1,2),sum(layer_size(l,:)))*2-1) * ...
%         sqrt(6/(layer_size(l+1,2)+sum(layer_size(l,:))));
    w{l} = (randn(layer_size(l+1, 2), sum(layer_size(l, :)))) * ...
        sqrt(6/( layer_size(l+1, 2) + sum(layer_size(l,:)) ));
    w_mean{l} = [];
    w_std{l} = [];
end

%% train
J=[];
Acc = []; % array to store accuracy of each mini batch
x=cell(L,1);
a=cell(L,1);
z=cell(L,1);
delta=cell(L,1);

%% loop
for iter=1:max_iter
    ind=randperm(train_size);
    for k=1:ceil(train_size/mini_batch)
        % preapre the internal input
        a{1}=zeros(layer_size(1,2),mini_batch);
        % prepare external input
        for l=1:L
            x{l}=X_train{l}(:,ind((k-1)*mini_batch+1:min(k*mini_batch,train_size)));
        end
        % prepare labels
        % prepare targets
        y = trainLabels(:,ind((k-1)*mini_batch+1:min(k*mini_batch,train_size)));
        
        % batch forward computation
        for l=1:L-1
            [a{1+l},z{1+l}]=fc(w{l},a{l},x{l},fs{l+1});
        end
        %cost function and error
        J=[J 1/2/mini_batch*sum((a{L}(:)-y(:)).^2)];
        Acc = [Acc accuracy(a{L}, y)]; 
        mini_batch = size(a{L}, 2);
        delta{L}=(a{L}-y).*dfs{L}(z{L});
        % run bc
        
        for l=L-1:-1:2
            delta{l}=bc(w{l},z{l},delta{l+1},dfs{l});
        end
        % update weight
        for l=1:L-1
            dw=delta{l+1}*[x{l};a{l}]' /mini_batch;
            w{l}=w{l}-alpha*dw;
            w_mean{l} = [w_mean{l} mean(w{l}(:))];
            w_std{l} = [w_std{l} std(w{l}(:))];
        end
    end
    % end loop
    
    if mod(iter,1)==0
        fprintf('%i/%i epochs: J=%.4f Acc=%.4f\n', iter, max_iter, J(end), Acc(end));
    end
end

% subplot(2,1,1);
% plot(J);
% subplot(2,1,2);
% plot(Acc);

%% save model
save model_7l_easy.mat w layer_size J w_mean w_std

%% display/listen to some results pairs

% test on training set
a{1} = zeros(layer_size(1,2), train_size);
x{1} = X_train{1};
for l=1:L-1
    [a{1+l},z{1+l}]=fc(w{l},a{l},x{l},fs{l+1});
end
train_acc = accuracy(a{L}, trainLabels);
fprintf('Accuracy on training dataset is %f%%\n', train_acc*100);

%% 
% test on testing set
a{1}=zeros(layer_size(1,2),test_size);
x{1} = X_test{1};
for l=1:L-1
    [a{1+l},z{1+l}]=fc(w{l},a{l},x{l},fs{l+1});
end
test_acc = accuracy(a{L}, testLabels);
fprintf('Accuracy on testing dataset is %f%%\n', test_acc*100);


