function [ f ] = k_nearest_theta_fit( data )
    [r1, r2, theta, ~, sim, ~] = extract_results(data);
    k = 10;
    
    function y = model(input)
        y = zeros(1, size(input, 2));
        for i = 1:size(input, 1)
            x = input(i, :);
            distances = sqrt((r1 - x(1)).^2 + (r2 - x(2)).^2 + (theta - x(3)).^2);
            [~, indexes] = sort(distances);
            neighbours = indexes(1:k);
            y(i) = mean(sim(neighbours));
        end
    end

    f = @model;

end

