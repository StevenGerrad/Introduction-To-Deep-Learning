function [audio_y] = get_audio(y, audio)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
    audio_y = zeros(size(audio, 2), size(y, 2));
    for i = 1:size(y, 2)
        audio_y(:,i) = audio(find(y(:, i) == 1), :);
    end
end

