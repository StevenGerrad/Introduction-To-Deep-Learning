function [] = dataPrepare()
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

SamplePath1 =  'random_animal\';  %�洢ͼ���·��
fileExt = '*.JPEG';  %����ȡͼ��ĺ�׺��
%��ȡ����·��
files = dir(fullfile(SamplePath1,fileExt)); 
len = size(files,1);

% aa = [];
% bb = [];
image = [];
%% loop
%����·����ÿһ��ͼ��
cnt = 0;
for i=1:len
   fileName = strcat(SamplePath1,files(i).name);
   im_t = imread(fileName);
   [a, b, c] = size(im_t);
%    aa = [aa a];
%    bb = [bb b];
   im_t = imresize(im_t,[80 80]);
   % ��ͼ��ת��Ϊ�Ҷ�ͼ
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

