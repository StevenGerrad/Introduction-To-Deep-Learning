function a_next = fc_v(w, a)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% implement forward computation from layer l to layer l+1
% in vector form
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % define the activation function
    f = @(s) s >= 0;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Your code BELOW
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 1. add external inputs with value 1
    a = [a; 1];
    % 2. calculate net input
    a_next = w * a;
    % 3. calculate activation
    a_next = f(a_next);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Your code ABOVE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
