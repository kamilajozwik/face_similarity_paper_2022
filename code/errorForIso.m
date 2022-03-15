function [ error ] = errorForIso( y, f)

error = sum((y-f).^2) / sum((y - mean(y)).^2);

end

