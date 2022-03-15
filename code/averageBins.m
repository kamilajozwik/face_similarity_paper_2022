function [ matrix_avg ] = averageBins( M, n )
% M = rand([216 31286]);
% n = 9   % want 9-row average.

% reshape
tmp = reshape(M, [n numel(M)/n]);
% mean column-wise (and only 9 rows per col)
tmp = mean(tmp);
% reshape back
matrix_avg = reshape(tmp, [ size(M,1)/n size(M,2) ]);
end

