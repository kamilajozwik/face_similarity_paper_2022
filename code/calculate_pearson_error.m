function [ pearson_error ] = calculate_pearson_error( similarities )

pearson_error(1) = pearson(similarities(1,:), similarities(2,:));
pearson_error(2) = pearson(similarities(1,:), similarities(3,:));
pearson_error(3) = pearson(similarities(2,:), similarities(3,:));

pearson_error(4) = errorForIso(similarities(1,:), similarities(2,:));
pearson_error(5) = errorForIso(similarities(1,:), similarities(3,:));
pearson_error(6) = errorForIso(similarities(2,:), similarities(3,:));

end

