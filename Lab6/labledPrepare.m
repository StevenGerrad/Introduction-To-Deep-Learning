% function [] = labledPrepare()
% 训练集与测试集以及label生成
clear;
close all;

SamplePath1 =  'lion\';  %存储图像的路径
SamplePath2 =  'tiger\';  
fileExt = '*.JPEG';  %待读取图像的后缀名
%获取所有路径
files = dir(fullfile(SamplePath1,fileExt)); 
len = size(files,1);

% aa = [];
% bb = [];
bigcat_image = [];
lt_train_label = [];
lt_test_data = [];
lt_test_label = [];

% lt_test_data = [lt_test_data im_t'];
% lt_test_label = [lt_test_label [1; 0]];
% bigcat_image = [bigcat_image im_t'];
% lt_train_label = [lt_train_label [1;0]];

%% lion data
for i=1:len
   fileName = strcat(SamplePath1,files(i).name); 
   im_t = imread(fileName);
   [~, ~, c] = size(im_t);
   % 将图像统一resize为40x40
   im_t = imresize(im_t,[80 80]);
   % 将图像转换为灰度图
   if c == 3
       im_t = rgb2gray(im_t);
   end
   % 图像重塑为一维数据
   im_t = reshape(im_t, 1, 6400);
   % make训练、测试集
   bigcat_image = [bigcat_image im_t'];
   lt_train_label = [lt_train_label [1;0]];
   if mod(i, 20) == 0
       fprintf('had read lion %d\n',i/20);
   end
end


%% tiger data
files = dir(fullfile(SamplePath2,fileExt)); 
len = size(files,1);
for i=1:len
   fileName = strcat(SamplePath2,files(i).name); 
   im_t = imread(fileName);
   [~, ~, c] = size(im_t);
   im_t = imresize(im_t,[80 80]);
   if c == 3
    im_t = rgb2gray(im_t);
   end
   im_t = reshape(im_t, 1, 6400);
   bigcat_image = [bigcat_image im_t'];
   lt_train_label = [lt_train_label [0;1]];
   if mod(i, 20) == 0
       fprintf('had read tiger %d\n',i/20);
   end
end

len = size(bigcat_image, 2);
ind = randperm(len);
lt_test_data = bigcat_image(:,ind(320+1:len));
lt_test_label = lt_train_label(:,ind(320+1:len));

bigcat_image = bigcat_image(:,ind(1:320));
lt_train_label = lt_train_label(:,ind(1:320));

%% save the lion and tiger data
% fprintf('min_aa:%d minbb_%d\n',aa, bb);
% histogram(aa);
% hold on;
% histogram(bb);
bigcat_image = im2double(bigcat_image);
lt_test_data = im2double(lt_test_data);

%% prepare the compress
load model_auto_hope3_spa.mat
input_size = 80 * 80;   % patch
% prepare data
train_size = size(bigcat_image, 2);
X_train{1} = reshape(bigcat_image, [], train_size);
X_train{2} = zeros(0, train_size);
X_train{3} = zeros(0, train_size);      % 实际上只用三个就行了

sigm = @(s) 1 ./ (1 + exp(-s));
fs = {[], sigm, sigm, sigm, sigm, sigm, sigm, sigm};

len = size(bigcat_image, 2);
L = 3;  % ** ATTENTION **
x = cell(L, 1);
a = cell(L, 1);
% z = cell(L, 1);

a{1} = zeros(layer_size(1, 2), len);
for l = 1:L
    x{l} = X_train{l};
end
for l = 1:L-1
    [a{l+1}, ~] = fc(w{l}, a{l}, x{l}, fs{l+1});
end
lt_train_data = a{L};

save labledIm.mat bigcat_image lt_train_data lt_train_label lt_test_data lt_test_label

% end

