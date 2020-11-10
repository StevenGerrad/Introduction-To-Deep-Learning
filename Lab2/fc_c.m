function a_next = fc_c(w, a)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% implement forward computation from layer l to layer l+1
% in component form
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % define the activation function
    f = @(s) s >= 0;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Your code BELOW
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 1. add external inputs with value 1
    a = [a; 1];
    % for each neuron located in layer l+1
    for i=1:size(w,1)
        % 2. calculate net input
        a_next(i) = w(i,:)*a;
        % 3. calculate activation
        a_next(i) = f( a_next(i) );
    end
    a_next = a_next';
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Your code ABOVE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
