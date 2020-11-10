function [] = dataPrepare()
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明

SamplePath1 =  'random_animal\';  %存储图像的路径
fileExt = '*.JPEG';  %待读取图像的后缀名
%获取所有路径
files = dir(fullfile(SamplePath1,fileExt)); 
len = size(files,1);

% aa = [];
% bb = [];
image = [];
%% loop
%遍历路径下每一幅图像
cnt = 0;
for i=1:len
   fileName = strcat(SamplePath1,files(i).name);
   im_t = imread(fileName);
   [a, b, c] = size(im_t);
%    aa = [aa a];
%    bb = [bb b];
   im_t = imresize(im_t,[80 80]);
   % 将图像转换为灰度图
   if c == 3
    im_t = rgb2gray(im_t);
   end
%    saveName = strcat('test\',files(i).name);
%    imwrite(im_t,saveName);
   
   im_t = reshape(im_t, 1, 6400);
   image = [image im_t'];
   if mod(i, 100) == 0
       fprintf('had read %d\n',i/100);
   end
   %norubbish_data(:,:,:,i) = image;
end
%% show the statistics
% fprintf('min_aa:%d minbb_%d\n',aa, bb);
% subplot(2,1,1);
% histogram(aa,20);
% subplot(2,1,2);
% histogram(bb,20);

%% save the data
save random_data.mat image

end

