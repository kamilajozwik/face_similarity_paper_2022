function [ f ] = k_nearest_distance_fit( data )
    [~, ~, ~, eu_d, sim, ~] = extract_results(data);
    k = 10;

    function y = model(input)
        y = zeros(1, length(input));
        for i = 1:length(input)
            x = input(i);
            [~, indexes] = sort(abs(eu_d - x));
            neighbours = indexes(1:k);
            y(i) = mean(sim(neighbours));
        end
    end

    f = @model;

end

