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
input_size=28*28;
% prepare training data
train_size=size(trainLabels,2);
X_train{1}=reshape(trainData,[],train_size);
X_train{2}=zeros(0,train_size);
X_train{3}=zeros(0,train_size);
X_train{4}=zeros(0,train_size);
X_train{5}=zeros(0,train_size);
% prepare testing data
test_size = size(testLabels, 2);
X_test{1} = reshape(testData, [], test_size);
X_test{2} = zeros(0, test_size);
X_test{3} = zeros(0, test_size);
X_test{4} = zeros(0, test_size);
X_test{5} = zeros(0, test_size);

%% prepare standard speech audio
sample_rate=4000;
audio=zeros(2983,10);
for i = 1:10
    [audio(:,i),sample_rate]=audioread(fullfile('audio',sprintf('%d.wav',i-1)));
    soundsc(audio(:,i),sample_rate);
    pause(1)
end
audio=(audio+1)/2;

%% choose parameters
alpha=0.1;
max_iter=300;
mini_batch=100;
layer_size=[input_size 512
                     0 512
                     0 1024
                     0 2048
                     0 2983];
L=size(layer_size,1);
 
 %% define network architecture
sigm=@(s) 1./(1+exp(-s));
dsigm=@(s) sigm(s).*(1-sigm(s));
% lin = @(s) s;
% dlin=@(s) 1;
fs={[],sigm,sigm,sigm,sigm,sigm,sigm,sigm};
dfs={[],dsigm,dsigm,dsigm,dsigm,dsigm,dsigm};


%% initialize weights
w=cell(L-1,1);
for l=1:L-1
    w{l}=(rand(layer_size(l+1,2),sum(layer_size(l,:)))*2-1)*sqrt(6/(layer_size(l+1,2)+sum(layer_size(l,:))));
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
        [~,ind_label]=max(trainLabels(:,ind((k-1)*mini_batch+1:min(k*mini_batch,train_size))));
        % prepare targets
        y=audio(:,ind_label);
        
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
        end
    end
    % end loop
    
    if mod(iter,1)==0
        fprintf('%i/%i epochs:J=%.4f\n', iter,max_iter,J(end));
    end
end

subplot(2,1,1);
plot(J);
subplot(2,1,2);
plot(Acc);

%% save model
save model.mat w layer_size J

%% display/listen to some results pairs

%test on training set
a{1} = zeros(layer_size(1,2),train_size);
x{1} = X_train{1};
for l=1:L-1
    [a{1+l},z{1+l}]=fc(w{l},a{l},x{l},fs{l+1});
end
train_acc = accuracy(a{L}, trainLabels);
fprintf('Accuracy on training dataset is %f%%\n', train_acc*100);

%test on testing set
a{1}=zeros(layer_size(1,2),test_size);
x{1} = X_test{1};
for l=1:L-1
    [a{1+l},z{1+l}]=fc(w{l},a{l},x{l},fs{l+1});
end
test_acc = accuracy(a{L}, testLabels);
fprintf('Accuracy on testing dataset is %f%%\n', test_acc*100);

%% random test
test = randi(test_size);
sample_rate=4000;
% disp(testLabels(:, test));
acc_num = sum([0 1 2 3 4 5 6 7 8 9]*testLabels(:, test));
acc_aud = audio(:,acc_num);
% soundsc(acc_aud,sample_rate);
pause(1)
soundsc(a{L}(:,test),sample_rate);
fprintf('%d %.4f\n',acc_num,sum((a{L}(:,test)-acc_aud(:)).^2)/2);
imshow(reshape(x{1}(:,test), 28, 28));
% pause(1)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Your code ABOVE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
