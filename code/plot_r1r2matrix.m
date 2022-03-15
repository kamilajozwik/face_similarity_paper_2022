function [ ] = plot_r1r2matrix( all_r1, all_r2, all_theta, all_similarities, theta )

r_values = unique(all_r1);
bin = all_theta == theta;
r1 = all_r1(bin);
r2 = all_r2(bin);
sim = all_similarities(bin);
S = [];
for i1 = 1:length(r_values)
    for i2 = 1:length(r_values)
        indexes1 = find(r1 == r_values(i1));
        indexes2 = find(r2 == r_values(i2));
        indexes = intersect(indexes1, indexes2);
        S(i1, i2) = mean(sim(indexes));
    end
end
% make S symmetric
S_transposed = S';
S(isnan(S)) = S_transposed(isnan(S));
% plot S
surf(r_values, r_values, S, 'EdgeColor', 'None', 'facecolor', 'interp');
view(2);

end

