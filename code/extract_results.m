function [ all_r1, all_r2, all_theta, all_eu_distances, all_similarities, seperators, all_pair_ids ] = extract_results( resultsStruct )

all_eu_distances = [];
all_similarities = [];

all_r1 = [];
all_r2 = [];
all_theta = [];
all_pair_ids = [];
%% perform initial analysis (n.b. coordinates are in [x, y] format, i.e. column 2 has the vertical data 


seperators = [];
 for trial = 1:size(resultsStruct, 2)
    for i = 2:9 % i.e. leave out the identity separator for now (at row 1)
        % compute euclidean distance
        geodat=resultsStruct(trial).geometricInfo(i).polarRelations;
        r1=geodat(1); r2=geodat(2); theta_deg = geodat(3); 
        eu_d = sqrt(r1^2 + r2^2 - 2*r1*r2*cosd(theta_deg));
        y_coord = resultsStruct(trial).facePositions(i,2); 
%         plot(eu_d, y_coord, 'r*');
        all_r1(end+1) = r1;
        all_r2(end+1) = r2;
        all_theta(end+1) = theta_deg;
        
        all_eu_distances(end+1) = eu_d;
        all_similarities(end+1) = 1-y_coord;
        all_pair_ids(end+1) = resultsStruct(trial).PairOrderAndID(i-1);
        
        %plot3(resultsStruct(trial).geometricInfo(i).polarRelations(1), resultsStruct(trial).geometricInfo(i).polarRelations(2), resultsStruct(trial).geometricInfo(i).polarRelations(3),'MarkerFaceColor', cmap(round(resultsStruct(2).facePositions(i,2)*64), :))
%         hold on
    end
    seperator = resultsStruct(trial).facePositions(1,2);
    seperators(end+1) = seperator;
 end

% sort from smallest eu_distance to biggest
all_parameters = [all_r1' all_r2' all_theta' all_eu_distances' all_similarities'];
[~, order] = sort(all_parameters(:, 4));
all_parameters_sorted = all_parameters(order, :);

% all_r1 = all_parameters_sorted(:,1)';
% all_r2 = all_parameters_sorted(:,2)';
% all_theta = all_parameters_sorted(:,3)';
% all_eu_distances = all_parameters_sorted(:,4)';
% all_similarities = all_parameters_sorted(:,5)';

 
end

