% fc.m
% This is forward computation function

function [a_next, z_next] = fc(w, a)
    % define the activation function
    f = @(s) 1 ./ (1 + exp(-s)); 
  
    % forward computing 
    z_next = w * a;
    a_next = f(z_next);
end
