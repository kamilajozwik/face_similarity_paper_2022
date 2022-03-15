function [ res ] = extract_eu_distance( data )
    [~, ~, ~, eu_d, ~, ~] = extract_results(data);
    res = eu_d';
end

