function [] = J_w_show(J_y, w_mean, w_std)
    figure;
    col = ['r', 'g', 'b', 'k', 'c', 'm', 'y'];
    %% plot J
    subplot(3,1,1);
    plot(J_y);
    ylim([0 ceil(max(J_y(:)))]);
    
    %% plot the average of W
    subplot(3,1,2);
    plot(w_mean{1},'r');
    for l = 2:size(w_mean, 2)
        hold on;plot(w_mean{l}, col(l));
    end
    %% plot the std of W
    subplot(3,1,3);
    plot(w_std{1},'r');
    for l = 2:size(w_std, 2)
        hold on;plot(w_std{l}, col(l));
    end
end

