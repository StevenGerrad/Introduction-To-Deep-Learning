function [] = a_w_show(a,w)

if max(size(a)) == 5
    %% picture the w
    figure;
    subplot(2,2,1); 
    histogram(w{1});
    subplot(2,2,2); 
    histogram(w{2});
    subplot(2,2,3); 
    histogram(w{3});
    subplot(2,2,4); 
    histogram(w{4});

    %% picture the a
    figure;
    subplot(2,3,1);
    histogram(a{1});
    subplot(2,3,2);
    histogram(a{2});
    subplot(2,3,3);
    histogram(a{3});
    subplot(2,3,4);
    histogram(a{4});
    subplot(2,3,5);
    histogram(a{5});
end

if max(size(a)) == 7
    %% picture the w
    figure;
    subplot(3,2,1); histogram(w{1});
    subplot(3,2,2); histogram(w{2});
    subplot(3,2,3); histogram(w{3});
    subplot(3,2,4); histogram(w{4});
    subplot(3,2,5); histogram(w{5});
    subplot(3,2,6); histogram(w{6});

    %% picture the a
    figure;
    subplot(3,3,1); histogram(a{1});
    subplot(3,3,2); histogram(a{2});
    subplot(3,3,3); histogram(a{3});
    subplot(3,3,4); histogram(a{4});
    subplot(3,3,5); histogram(a{5});
    subplot(3,3,6); histogram(a{6});
    subplot(3,3,7); histogram(a{7});
end

end

