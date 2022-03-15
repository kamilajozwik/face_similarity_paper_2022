function [ f ] = linear_distance_fit( data )
    [~, ~, ~, eu_d, sim, ~] = extract_results(data);
    
    param = polyfit(eu_d, sim, 1);
    f = @(x) param(1)*x + param(2);
end

