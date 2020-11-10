%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Course:  Understanding Deep Neural Networks
% Teacher: Zhang Yi
% Student:
% ID:
%
% Lab 6 - Big Cat Recognition
%
% Task: In this example, the labeled training data
%       are insufficient to train a classifier by
%       using BP directly. However, a good
%       classifier can be developed by using the
%       autoencoder method.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function delta = bc(w, z, delta_next, df , beta)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Your code BELOW
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % backward computing (either component or vector form)
    % f = @(s) 1 ./ (1 + exp(-s));
    % df = @(s) f(s) .* (1 - f(s));
    
    % delta = (w' * delta_next + beta) .* df(z);
    delta = df(z).*( w(:, end-size(z,1)+1:end)' * delta_next + beta);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Your code ABOVE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
