%% accuracy.m 
% Define Evaluation Index
function [ acc ] = accuracy( a, y )
    mini_batch = size(a, 2);
    [~,idx_a] = max(a);
    [~,idx_y] = max(y);
    acc = sum(idx_a==idx_y) / mini_batch;
end
