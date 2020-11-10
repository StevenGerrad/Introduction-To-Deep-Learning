% bc.m
% This is backward computation function

function delta = bc(w, z, delta_next)
 
    % activation function
    f = @(s) 1 ./ (1 + exp(-s)); 
    
    % derivative of activation  function
    df = @(s) f(s) .* (1 - f(s)); 
 
    % backward computing
    delta = (w' * delta_next) .* df(z);
 
end
