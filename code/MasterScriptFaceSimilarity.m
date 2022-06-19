
%% PRELIM: load data

load('data/all_data_combined/all_similarities.mat');

load('data/all_data_combined/all_subjects.mat');
load('data/all_data_combined/all_pair_ids.mat');
load('data/all_data_combined/all_sessions.mat');
load('data/all_data_combined/all_separators.mat');
load('data/all_data_combined/all_eu_distances.mat');
load('data/all_data_combined/all_r1.mat');
load('data/all_data_combined/all_r2.mat');
load('data/all_data_combined/all_separators.mat');
load('data/all_data_combined/all_theta.mat');
load('data/all_data_combined/all_pair_ids.mat');


%% PRELIM: get similarities for face pairs
for face_pair = 1:232
    indeces_of_face_pair = all_pair_ids == face_pair;
    
    similarity_across_two_sessions = all_similarities(indeces_of_face_pair);
    similarity_of_face_pair(:,face_pair) = similarity_across_two_sessions;
    mean_similarity_of_face_pair(face_pair) = mean(similarity_across_two_sessions);
    std_similarity_of_face_pair(face_pair) = std(similarity_across_two_sessions);
end

similarity_of_face_pair_session_one = similarity_of_face_pair((1:2:end),:);
similarity_of_face_pair_session_two = similarity_of_face_pair((2:2:end),:);
concatenated_two_sessions_subj_repeated = cat(3, similarity_of_face_pair_session_one, similarity_of_face_pair_session_two);
similarity_of_face_pair_two_sessions = mean(concatenated_two_sessions_subj_repeated,3);
similarity_of_face_pair_two_sessions_mean = mean(similarity_of_face_pair_two_sessions,1);


save_mat_kmj('analysis/concatenated_two_sessions_subj_repeated.mat', 'concatenated_two_sessions_subj_repeated');
save_mat_kmj('analysis/mean_similarity_of_face_pair.mat', 'mean_similarity_of_face_pair');

[~, order_similarity] = sort(mean_similarity_of_face_pair);

for subj = 1:26
    [~, order_similarity_per_subj(subj,:)] = sort(similarity_of_face_pair_two_sessions(subj,:));
end
[ordered_mean_similarity_of_face_pair, order_similarity] = sort(mean_similarity_of_face_pair);

%% PRELIM: get Euclidean distances for face pairs
for face_pair = 1:232
    indeces_of_face_pair = all_pair_ids == face_pair;
    
    eu_distances_across_two_sessions = all_eu_distances(indeces_of_face_pair);
    eu_distances_of_face_pair(:,face_pair) = eu_distances_across_two_sessions;
    mean_eu_distances_of_face_pair(face_pair) = mean(eu_distances_across_two_sessions);
    std_eu_distances_of_face_pair(face_pair) = std(eu_distances_across_two_sessions);
end

eu_distances_of_face_pair_session_one = eu_distances_of_face_pair((1:2:end),:);
eu_distances_of_face_pair_session_two = eu_distances_of_face_pair((2:2:end),:);
concatenated_two_sessions_subj_repeated_eu_distances = cat(3, eu_distances_of_face_pair_session_one, eu_distances_of_face_pair_session_two);
eu_distances_of_face_pair_two_sessions = mean(concatenated_two_sessions_subj_repeated_eu_distances,3);
eu_distances_of_face_pair_two_sessions_mean = squeeze(mean(eu_distances_of_face_pair_two_sessions,1));

save_mat_kmj('analysis/concatenated_two_sessions_subj_repeated_eu_distances.mat', 'concatenated_two_sessions_subj_repeated_eu_distances');

%% PRELIM: get theta for face pairs
for face_pair = 1:232
    indeces_of_face_pair = all_pair_ids == face_pair;
    
    theta_across_two_sessions = all_theta(indeces_of_face_pair);
    theta_of_face_pair(:,face_pair) = theta_across_two_sessions;
    mean_theta_of_face_pair(face_pair) = mean(theta_across_two_sessions);
    std_theta_of_face_pair(face_pair) = std(theta_across_two_sessions);
end

%% PRELIM: get r1 for face pairs
for face_pair = 1:232
    indeces_of_face_pair = all_pair_ids == face_pair;
    
    r1_across_two_sessions = all_r1(indeces_of_face_pair);
    r1_of_face_pair(:,face_pair) = r1_across_two_sessions;
    mean_r1_of_face_pair(face_pair) = mean(r1_across_two_sessions);
    std_r1_of_face_pair(face_pair) = std(r1_across_two_sessions);
end

%% PRELIM: get r2 for face pairs
for face_pair = 1:232
    indeces_of_face_pair = all_pair_ids == face_pair;
    
    r2_across_two_sessions = all_r2(indeces_of_face_pair);
    r2_of_face_pair(:,face_pair) = r2_across_two_sessions;
    mean_r2_of_face_pair(face_pair) = mean(r2_across_two_sessions);
    std_r2_of_face_pair(face_pair) = std(r2_across_two_sessions);
end

%% PRELIM: linear and sigmoid BFS distance
linear_bfs_distance = linear_distance_fit(all_sessions);
sigmoid_bfs_distance = sigmoid_distance_fit(all_sessions);

%% PRELIM: define mean identity separator and find after how many face pairs it is places sampling every 20 face pairs

mean_identity_separator = mean(all_separators);

stderror_identity_separator = std(all_separators)/sqrt(26);

index_of_identity_separator = find(ordered_mean_similarity_of_face_pair<mean_identity_separator);

index_of_identity_separator_max = max(index_of_identity_separator);

%% PRELIM: confidence intervals identity separator

SEM = std(all_separators)/sqrt(length(all_separators));% Standard Error
ts = tinv([0.05  0.95],length(all_separators)-1);      % T-Score
CI = mean(all_separators) + ts*SEM;                     % 

%% PRELIM: restructure data into pairs per subject
number_of_subjects = 26;

pairs_thetas = [];
pairs_eu_distances = [];
pairs_labels = [];
pairs_r1 = [];
pairs_r2 = [];

pair_ids = [];

for s = 1:number_of_subjects
    
    subject = all_subjects(s);
    
    pair_number = 0;
    
    for session = 1:2
        for trial = 1:29
            identity_line_position = subject.sessions(session, trial).facePositions(1,2);
            for pair = 2:9
                pair_number = pair_number + 1;
                pair_position = subject.sessions(session, trial).facePositions(pair,2);
                pair_id = subject.sessions(session, trial).PairOrderAndID(pair-1);
                geodat = subject.sessions(session, trial).geometricInfo(pair).polarRelations;
                r1=geodat(1); r2=geodat(2); theta_deg = geodat(3); 
                 if r1==0 || r2==0
                    theta_deg = nan;
                    r1 = nan;
                    r2 = nan;
                    eu_d = nan;
                end
                eu_d = sqrt(r1^2 + r2^2 - 2*r1*r2*cosd(theta_deg));
                pair_ids(s, pair_number) = pair_id;
                pairs_r1(s, pair_number) = r1;
                pairs_r2(s, pair_number) = r2;
                pairs_thetas(s, pair_number) = theta_deg;
                pairs_eu_distances(s, pair_number) = eu_d;
                if pair_position <= identity_line_position
                    pairs_labels(s, pair_number) = 1;
                else
                    pairs_labels(s, pair_number) = 2;
                end
            end
        end
    end
end

%% PRELIM: get pairs below and above identity line

thetas_below_identity_line = pairs_thetas(pairs_labels==1);
thetas_above_identity_line = pairs_thetas(pairs_labels==2);

eu_distances_below_identity_line = pairs_eu_distances(pairs_labels==1);
eu_distances_above_identity_line = pairs_eu_distances(pairs_labels==2);

r1_below_identity_line = pairs_r1(pairs_labels==1);
r1_above_identity_line = pairs_r1(pairs_labels==2);
r2_below_identity_line = pairs_r2(pairs_labels==1);
r2_above_identity_line = pairs_r2(pairs_labels==2);
abs_diff_r1_r2_below_identity_line = abs(r1_below_identity_line - r2_below_identity_line);
abs_diff_r1_r2_above_identity_line = abs(r1_above_identity_line - r2_above_identity_line);

%% PRELIM: compute dprimes and AUC

do_plot_individual_figs = 0;

dprimes_eu_d= [];
dprimes_theta = [];
dprimes_abs_r1_r2 = [];
AUC_eu_d = [];
AUC_theta = [];
AUC_abs_r1_r2 = [];

for s = 1:number_of_subjects
    [dprimes_eu_d(s), AUC_eu_d(s)] = dprime_and_roc(pairs_eu_distances(s,:)', pairs_labels(s,:)', do_plot_individual_figs);
    if do_plot_individual_figs; save_figure_kmj(['analysis/AUC/roc_curves/eu_distance/subj' num2str(s,'%02d') '.png']); end
    [dprimes_theta(s), AUC_theta(s)] = dprime_and_roc(pairs_thetas(s,:)', pairs_labels(s,:)', do_plot_individual_figs);
    if do_plot_individual_figs; save_figure_kmj(['analysis/AUC/roc_curves/theta/subj' num2str(s,'%02d') '.png']); end
    [dprimes_abs_r1_r2(s), AUC_abs_r1_r2(s)] = dprime_and_roc(abs(pairs_r1(s,:)-pairs_r2(s,:))', pairs_labels(s,:)', do_plot_individual_figs);
    if do_plot_individual_figs; save_figure_kmj(['analysis/AUC/roc_curves/absolute_difference/subj' num2str(s,'%02d') '.png']); end
end




%% Compute mean correlation between subjects across two sessions
similarity_of_face_pair_corr_pairwise_across_subjects = corr(similarity_of_face_pair');
mean_similarity_of_face_pair_corr = mean(similarity_of_face_pair_corr_pairwise_across_subjects(:));

%% Compute mean correlation between sessions
% subject data is interleaved, which can be deducted from reading the
% results, so we have subj1 sess1, subj 1 sess 2, subj2 sess 1 and subj2
% sess 2, that is why we extract session 1 and 2 taking every second
% entry like below
sess_1 = similarity_of_face_pair(1:2:end,:);
sess_2 = similarity_of_face_pair(2:2:end,:);
correlation_between_session_1_and_session_2_across_subjects = corr2(sess_1, sess_2);

%% arrange face pairs in montage showing every 20 face pairs for similarity judgements
number_of_face_pairs = 232;
file_folder = 'data/face_pair_images/';
figure;
filenames_similarity = {};
for i = 1: length(order_similarity)
    filenames_similarity{i} = [file_folder,'/pair_' num2str(order_similarity(i),'%03.f') '.tif'];
end

montage(filenames_similarity(1:20:number_of_face_pairs), 'Size', [12,1]);

save_figure_kmj('analysis/face_ranking/face_similarity_ranking_every_20');

%% save kate's stimuli as pngs
file_folder_kate = 'data/from_kate/facepair_info_144px_main_expt_KSscript/';
number_of_face_pairs = 232;
filenames_similarity_kate = {};
for i = 1: number_of_face_pairs
    load([file_folder_kate,'/pair' num2str(i,'%06.f') '.mat']);
%     pair{i} = imshow([img1, img2]);
    imwrite([img1, img2], ['data/from_kate/facepair_info_144px_main_expt_pairs_png/pair_' num2str(i,'%03.f'), '.png']);
end

%% arrange face pairs in montage showing every 20 face pairs for similarity judgements using kate's stimuli
file_folder_kate = 'data/from_kate/facepair_info_144px_main_expt_pairs_png/';
number_of_face_pairs = 232;
filenames_similarity_kate = {};
for i = 1: length(order_similarity)
    filenames_similarity_kate{i} = [file_folder_kate,'/pair_' num2str(order_similarity(i),'%03.f') '.png'];
end

figure;
montage(filenames_similarity_kate(fliplr(1:20:number_of_face_pairs)), 'Size', [12,1]);

save_figure_kmj('analysis/face_ranking/face_similarity_ranking_every_20_kate_face_pairs');

%% arrange face pairs in one big montage
number_of_face_pairs = 232;
filenames_similarity_kate = {};
for i = 1: length(order_similarity)
    filenames_similarity_kate{i} = [file_folder_kate,'/pair_' num2str(order_similarity(i),'%03.f') '.png'];
end

figure;
montage(filenames_similarity_kate(fliplr(1:(number_of_face_pairs))), 'Size', [20, 12]);

save_figure_kmj('analysis/montage_similarity_kate_face_pairs');
%% selecting only face pairs that were classified as te same similarity
figure;
all_separators_mean = mean(all_separators)
number_of_face_pairs_same_identity = sum(mean_similarity_of_face_pair<all_separators_mean)
montage(filenames_similarity_kate(1:(number_of_face_pairs_same_identity)), 'Size', [10,13]);

save_figure_kmj('analysis/montage_similarity_only_same_identity_pairs');
%% arrange face pairs in montage showing every 20 face pairs for VGG face WEIGHTED

VGG_face_distances = csvread('data/from_kate/weighted_models/01_VGG-Face_best.csv');


[~, order_VGG_face] = sort(VGG_face_distances);

number_of_face_pairs = 232;
file_folder_kate = 'data/from_kate/facepair_info_144px_main_expt_pairs_png/';
filenames_VGG_face_kate = {};
for i = 1: length(order_VGG_face)
    filenames_VGG_face_kate{i} = [file_folder_kate,'/pair_' num2str(order_VGG_face(i),'%03.f') '.png'];
end

figure;
montage(filenames_VGG_face_kate(fliplr(1:20:number_of_face_pairs)), 'Size', [12,1]);

save_figure_kmj('analysis/face_ranking/VGG_face_ranking_every_20_weighted_kate_face_pairs');

%% arrange face pairs in montage showing every 20 face pairs for VGG face WEIGHTED

VGG_face_distances = csvread('data/from_kate/weighted_models/01_VGG-Face_best.csv');


[~, order_VGG_face] = sort(VGG_face_distances);

number_of_face_pairs = 232;
file_folder = 'data/face_pair_images/';
filenames_VGG_face = {};
for i = 1: length(order_VGG_face)
    filenames_VGG_face{i} = [file_folder,'/pair_' num2str(order_VGG_face(i),'%03.f') '.tif'];
end

figure;
montage(filenames_VGG_face(1:20:number_of_face_pairs), 'Size', [12,1]);

save_figure_kmj('analysis/face_ranking/VGG_face_ranking_every_20_weighted');


%% correlation between order of similarity and VGGface
% corr2 is Pearson's correlation
order_correlation_similarity_VGG_face = corr2(order_similarity, order_VGG_face);

%% correlation between subjects

order_correlation_similarity_subjects = corr(order_similarity_per_subj');

order_correlation_similarity_subjects(logical(diag(diag(order_correlation_similarity_subjects)))) = 0; % putting zeros on diagonal as otherwise squreform won't work

mean_correlation_similarity_subjects = mean(squareform(order_correlation_similarity_subjects));


%% correlation between subjects with the mean subject (upperbound of the noise ceiling, more comparable to the case with VGG16)

mean_similarity_judgements = mean(order_similarity_per_subj, 1);

single_subj_corr_with_mean_across_subj = [];

for subj = 1:26
    single_subj_corr_with_mean_across_subj(subj) = corr2(order_similarity_per_subj(subj, :), mean_similarity_judgements);
end

mean_single_subj_corr_with_mean_across_subj = mean(single_subj_corr_with_mean_across_subj);

save_mat_kmj('analysis/correlations/pair_ranking_correlations_mean_subject_other_subjects', 'mean_single_subj_corr_with_mean_across_subj');


%% correlation between subjects with the VGG face (upperbound of the noise ceiling, more comparable to the case with VGG16)

single_subj_corr_with_VGG_face = [];

for subj = 1:26
    single_subj_corr_with_VGG_face(subj) = corr2(order_similarity_per_subj(subj, :), order_VGG_face);
end

mean_single_subj_corr_with_VGG_face = mean(single_subj_corr_with_VGG_face);

save_mat_kmj('analysis/correlations/pair_ranking_correlations_mean_subject_VGG_face', 'mean_single_subj_corr_with_VGG_face');

%% plot identity separator on data
close all;
figure('Name', 'Face similarity as a function of Basel Face Space');
sigmoid_bfs_distance = linear_distance_fit(all_sessions);
plot(all_eu_distances, all_similarities, 'ro','MarkerSize',2);
hold on;
mean_identity_separator_repeated = repelem(mean_identity_separator, length(all_eu_distances))
line([0 80], [mean_identity_separator mean_identity_separator]); 
hold on;
line([0 80], [mean_identity_separator+stderror_identity_separator mean_identity_separator+stderror_identity_separator]);
hold on;
line([0 80], [mean_identity_separator-stderror_identity_separator mean_identity_separator-stderror_identity_separator]);
xlabel('BFM Euclidean distance');
ylabel('Facial similarity judgements');

set(gca,'box', 'off');
xlim([-0.2 80]);

ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20;

save_figure_kmj('analysis/identity_separator_over_data', figure(1));

%% get thetas, r1, r2 for faces below average idnetity line
[indeces_all_similarities_below_identity_line]=find(all_similarities<mean_identity_separator)

all_similarities_below_identity_line = all_similarities(indeces_all_similarities_below_identity_line);

all_theta_below_identity_line = all_theta(indeces_all_similarities_below_identity_line);

all_r1_below_identity_line = all_r1(indeces_all_similarities_below_identity_line);

all_r2_below_identity_line = all_r2(indeces_all_similarities_below_identity_line);

%% get MEAN Eu dist, thetas, r1, r2 for faces below average idnetity line
[indeces_mean_similarities_below_identity_line]=find(mean_similarity_of_face_pair<mean_identity_separator)

[indeces_mean_similarities_above_identity_line]=find(mean_similarity_of_face_pair>mean_identity_separator)

mean_similarities_below_identity_line = mean_similarity_of_face_pair(indeces_mean_similarities_below_identity_line);

mean_theta_below_identity_line = mean_theta_of_face_pair(indeces_mean_similarities_below_identity_line);

mean_r1_below_identity_line = mean_r1_of_face_pair(indeces_mean_similarities_below_identity_line);

mean_r2_below_identity_line = mean_r2_of_face_pair(indeces_mean_similarities_below_identity_line);

mean_eu_dist_below_identity_line = mean_eu_distances_of_face_pair(indeces_mean_similarities_below_identity_line);

mean_similarities_below_identity_line = mean_similarity_of_face_pair(indeces_mean_similarities_below_identity_line);

mean_similarities_above_identity_line = mean_similarity_of_face_pair(indeces_mean_similarities_above_identity_line);

mean_theta_above_identity_line = mean_theta_of_face_pair(indeces_mean_similarities_above_identity_line);

mean_r1_above_identity_line = mean_r1_of_face_pair(indeces_mean_similarities_above_identity_line);

mean_r2_above_identity_line = mean_r2_of_face_pair(indeces_mean_similarities_above_identity_line);

mean_eu_dist_above_identity_line = mean_eu_distances_of_face_pair(indeces_mean_similarities_above_identity_line);

mean_mean_r1_r2_above_identity_line = (mean_r1_above_identity_line+mean_r2_above_identity_line)/2;

mean_mean_r1_r2_below_identity_line = (mean_r1_below_identity_line+mean_r2_below_identity_line)/2;

abs_difference_r1_r2_above_identity_line = abs(mean_r1_above_identity_line-mean_r2_above_identity_line);

abs_difference_r1_r2_below_identity_line = abs(mean_r1_below_identity_line-mean_r2_below_identity_line);


%% find indeces of face pairs where r1 and r2 were the largest
[indeces_mean_r1_below_identity_line_value_40]=find(mean_r1_below_identity_line==40);

[indeces_mean_r2_below_identity_line_value_40]=find(mean_r2_below_identity_line==40);

intersect_r1_r2_value_40_below_idnetity_line = intersect(indeces_mean_r1_below_identity_line_value_40, indeces_mean_r2_below_identity_line_value_40);

save_mat_kmj('analysis/intersect_r1_r2_value_40_below_idnetity_line.mat', 'intersect_r1_r2_value_40_below_idnetity_line')

%% logistic regression and dprime (based on mean data)

% create input to logistic regression
X = mean_theta_of_face_pair';

Y = nan(232,1);
Y(indeces_mean_similarities_below_identity_line)=1;
Y(indeces_mean_similarities_above_identity_line)=2;

[dprime, AUC] = dprime_and_roc(X, Y)



%% isotropicity across two sessions, same stimulus set

similarities = [similarity_of_face_pair_session_one, similarity_of_face_pair_session_two];

for repeat = 1:1
    spreads = [];
    random_sample_spreads = [];
    sampled_pair_ids = [];
    for s = 1:number_of_subjects
        geometries = [pairs_thetas(1,:); pairs_r1(1,:); pairs_r2(1,:)];
        nan_indexes = all(isnan(geometries));
        geometries(:, nan_indexes) = -1; % changing NaNs to -1 to make the rest of the code here easier (because NaN != NaN)
    %     geometries(:, nan_indexes) = [];

        unique_geometries = unique(geometries', 'rows')';

        for geometry = unique_geometries
            pair_indexes = find(all(geometries == geometry));
            if ~isequal(geometry, [-1 -1 -1]')
    %             pair_indexes
                % Apart from pairs of identical face, each geometry should be present just twice (once per session)
                assert(length(pair_indexes) == 2);
                assert(pair_indexes(1) <= 232);
                assert(pair_indexes(2) > 232);
                spreads(end+1) = abs(similarities(s, pair_indexes(2)) - similarities(s, pair_indexes(1)));

                non_nan_pairs = find(~nan_indexes);
                sampled_pairs = non_nan_pairs(randperm(length(non_nan_pairs),2));
                random_sample_spreads(end+1) = abs(similarities(s, sampled_pairs(2)) - similarities(s, sampled_pairs(1)));
                
                sampled_pair_ids(end+1,:) = pair_ids(s, sampled_pairs);
            end
        end
    end

    figure; plot([sort(spreads); sort(random_sample_spreads)]', '.'); legend({'spreads', 'random sample spreads'});
    save_figure_kmj('analysis/isotropicity/spreads.png');

    [p, h, stats]  = signrank(spreads, random_sample_spreads)
    
    save_mat_kmj('analysis/isotropicity/isotropicity.mat', 'sampled_pair_ids', 'p', 'h', 'stats');
end


%% count what is the percentage of faces classified as the same identity per face pair

% sort
n_pairs = length(order_similarity);
reordered_pairs_labels = [];
for s = 1:number_of_subjects
    for session = 1:2
        if session == 1
            sess_indices = 1:232;
        else
            sess_indices = 233:464;
        end
        for p = 1:n_pairs
            pair_id = order_similarity(p);
            index_of_pair_in_sess_1 = find(pair_ids(s,sess_indices) == pair_id);
            pair_number = (session-1)*n_pairs + p;
            reordered_pairs_labels(s, pair_number) = pairs_labels(s, index_of_pair_in_sess_1);
        end
    end
end

frequency_pairs_below_identity_line  = sum(reordered_pairs_labels == 1);

frequency_pairs_below_identity_line_sess_1 = frequency_pairs_below_identity_line(1:232);
frequency_pairs_below_identity_line_sess_2 = frequency_pairs_below_identity_line(233:end);
frequency_pairs_below_identity_line_both_sess = (frequency_pairs_below_identity_line_sess_1 + frequency_pairs_below_identity_line_sess_2)/2;

frequency_pairs_below_identity_line_both_sess_norm = frequency_pairs_below_identity_line_both_sess/26;

frequency_every_20  = frequency_pairs_below_identity_line_both_sess_norm(1:20:end);

%% plot colorbar of frequency every 20 pair
figure;
colormap(gray(256));
imagesc(frequency_every_20');
colorbar;
caxis([0 1]);
colorbar('southoutside');
save_figure_kmj('analysis/frequency_same_identity/frequency_of_same_identity_every_20_face_pair_colorbar');

%% plot dprimes

mean_dprimes_eu_d = nanmean(dprimes_eu_d);
mean_dprimes_theta = nanmean(dprimes_theta);
mean_dprimes_abs_r1_r2 = nanmean(dprimes_abs_r1_r2);

stderr_dprimes_eu_d = (nanstd(dprimes_eu_d))/number_of_subjects;
stderr_dprimes_theta = (nanstd(dprimes_theta))/number_of_subjects;
stderr_dprimes_abs_r1_r2 = (nanstd(dprimes_abs_r1_r2))/number_of_subjects;

close all;
figure;

mean_dprime = [mean_dprimes_eu_d, mean_dprimes_theta, mean_dprimes_abs_r1_r2];

stderr_dprime = [stderr_dprimes_eu_d, stderr_dprimes_theta, stderr_dprimes_abs_r1_r2];

bar(mean_dprime, 'FaceColor', [0.5 0.5 0.5]);
hold on;

% sigstar(significance_matrix_unique_variance_fdr_corrected_indeces,max_y_position,[], 1);

hold on;

number_of_models = 3;

e = errorbar(1:number_of_models, mean_dprime, stderr_dprime, '.', 'CapSize',0, 'LineWidth', 6, 'CapSize' , 6);
e.Color = 'black';
e.Marker = 'none';

model_names  = {'Euclidean distance','Theta','|r_{1} - r_{2}|'};
set(gca,'FontName','Arial','TickLength',[0 0],'XTick',...
    1:number_of_models,'XTickLabel',... 
    model_names, 'XTickLabelRotation',45); 
    ylabel('d prime ', 'FontName','Arial');

set(gca,'box', 'off');

% ylim([0 0.025]);

ax = gca; ax.XAxis.FontSize = 25; ax.YAxis.FontSize = 25;

save_figure_kmj('analysis/dprimes/dprimes', figure(1));

%% plot AUC

mean_AUC_eu_d = nanmean(AUC_eu_d);
mean_AUCs_theta = nanmean(AUC_theta);
mean_AUC_abs_r1_r2 = nanmean(AUC_abs_r1_r2);

stderr_AUC_eu_d = (nanstd(AUC_eu_d))/number_of_subjects;
stderr_AUC_theta = (nanstd(AUC_theta))/number_of_subjects;
stderr_AUC_abs_r1_r2 = (nanstd(AUC_abs_r1_r2))/number_of_subjects;

close all;
figure;

mean_AUC = [mean_AUC_eu_d, mean_AUCs_theta, mean_AUC_abs_r1_r2];

stderr_AUC = [stderr_AUC_eu_d, stderr_AUC_theta, stderr_AUC_abs_r1_r2];

bar(mean_AUC, 'FaceColor', [0.5 0.5 0.5]);
hold on;

% sigstar(significance_matrix_unique_variance_fdr_corrected_indeces,max_y_position,[], 1);

hold on;

number_of_models = 3;

e = errorbar(1:number_of_models, mean_AUC, stderr_AUC, '.', 'CapSize',0, 'LineWidth', 6, 'CapSize' , 6);
e.Color = 'black';
e.Marker = 'none';

model_names  = {'Euclidean distance','Theta','|r_{1} - r_{2}|'};
set(gca,'FontName','Arial','TickLength',[0 0],'XTick',...
    1:number_of_models,'XTickLabel',... 
    model_names, 'XTickLabelRotation',45); 
    ylabel('Area under ROC Curve ', 'FontName','Arial');

set(gca,'box', 'off');

% ylim([0 0.025]);

ax = gca; ax.XAxis.FontSize = 25; ax.YAxis.FontSize = 25;

save_figure_kmj('analysis/AUC/AUC', figure(1));


%% plot histogram for MEAN theta below and above identity line
close all;
figure;
colour_above_identity_line = [180/255,180/255,180/255];
colour_below_identity_line = [100/255,100/255,100/255];

bins = 6;
histogram(thetas_below_identity_line , bins, 'FaceColor',colour_above_identity_line , 'FaceAlpha', 0.5, 'EdgeAlpha', 1);
hold on;
histogram(thetas_above_identity_line, bins, 'FaceColor',colour_below_identity_line , 'FaceAlpha', 0.5, 'EdgeAlpha', 1);

l = legend( 'different identity', 'same identity');
l.FontSize = 50;
l.FontName = 'Arial';
legend('boxoff');

ax = gca; ax.XAxis.FontSize = 40; ax.YAxis.FontSize = 40; ax.YAxis.Exponent = 0;

ytickformat('%g')

xticks(0:30:180);

set(gca,'box', 'off');

ylabel('Frequency', 'FontName','Arial', 'Fontsize', 50);
% ylabel('Frequency of facial dissimilarity judgements', 'FontName','Arial', 'Fontsize', 40);
xlabel('Theta', 'FontName','Arial', 'Fontsize', 50);

ylim([0, 4000]);

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 10], 'PaperUnits', 'Inches', 'PaperSize', [20, 10])

save_figure_kmj('analysis/histograms/hist_same_differnt_identity_theta', figure(1));

%% plot histogram for MEAN absolute difference between r1 and r2 below and above identity line
close all;
figure;

colour_above_identity_line = [180/255,180/255,180/255];
colour_below_identity_line = [100/255,100/255,100/255];
bins = 6;
histogram(abs_diff_r1_r2_below_identity_line, bins, 'FaceColor',colour_above_identity_line, 'FaceAlpha', 0.5, 'EdgeAlpha', 1);
hold on;
histogram(abs_diff_r1_r2_above_identity_line, bins, 'FaceColor',colour_below_identity_line, 'FaceAlpha', 0.5, 'EdgeAlpha', 1);

l = legend( 'different identity', 'same identity');
l.FontSize = 50;
l.FontName = 'Arial';
legend('boxoff');

% ylim([0, 40]);

ax = gca; ax.XAxis.FontSize = 40; ax.YAxis.FontSize = 40; ax.YAxis.Exponent = 0;

ytickformat('%g')

xticks(0:10:40);

set(gca,'box', 'off');

ylabel('Frequency', 'FontName','Arial', 'Fontsize', 50);
% ylabel('Frequency of facial dissimilarity judgements', 'FontName','Arial', 'Fontsize', 40);
xlabel('|r_{1} - r_{2}|', 'FontName','Arial', 'Fontsize', 50);

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 10], 'PaperUnits', 'Inches', 'PaperSize', [20, 10])

save_figure_kmj('analysis/histograms/hist_same_differnt_identity_abs_difference_r1_r2', figure(1));

%% plot histogram for MEAN eu dist below and above identity line
close all;
figure;
bins = 6;
colour_above_identity_line = [180/255,180/255,180/255];
colour_below_identity_line = [100/255,100/255,100/255];

histogram(eu_distances_below_identity_line, bins,'BinWidth',15, 'FaceColor',colour_above_identity_line, 'FaceAlpha', 0.5, 'EdgeAlpha', 1);
hold on;
histogram(eu_distances_above_identity_line, bins, 'BinWidth',15, 'FaceColor',colour_below_identity_line, 'FaceAlpha', 0.5, 'EdgeAlpha', 1);

l = legend( 'different identity', 'same identity');
l.FontSize = 50;
l.FontName = 'Arial';
legend('boxoff');

% ylim([0, 40])

ax = gca; ax.XAxis.FontSize = 40; ax.YAxis.FontSize = 40; ax.YAxis.Exponent = 0;

ytickformat('%g')

xticks(0:20:80);

set(gca,'box', 'off');

ylabel('Frequency', 'FontName','Arial', 'Fontsize', 50);
% ylabel('Frequency of facial dissimilarity judgements', 'FontName','Arial', 'Fontsize', 40);
xlabel('Euclidean distance', 'FontName','Arial', 'Fontsize', 50);

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 10], 'PaperUnits', 'Inches', 'PaperSize', [20, 10])

save_figure_kmj('analysis/histograms/hist_same_differnt_identity_eu_dist', figure(1));

%% plot distributions of theta
close all;
figure; 
colour = [100/255,100/255,100/255];
indeces_r1_r2_zero = find(all_r1 ==0 | all_r2 ==0);
all_theta(indeces_r1_r2_zero) = nan;
histogram(all_theta,'FaceColor',colour, 'FaceAlpha', 0.5, 'EdgeAlpha', 1)


% ylim([0, 40])

ax = gca; ax.XAxis.FontSize = 40; ax.YAxis.FontSize = 40; ax.YAxis.Exponent = 0;

ytickformat('%g')

xticks(0:30:180);

set(gca,'box', 'off');

ylabel('Frequency', 'FontName','Arial', 'Fontsize', 50);
% ylabel('Frequency of facial dissimilarity judgements', 'FontName','Arial', 'Fontsize', 40);
xlabel('Theta', 'FontName','Arial', 'Fontsize', 50);

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 10], 'PaperUnits', 'Inches', 'PaperSize', [20, 10])

save_figure_kmj('analysis/histograms/hist_theta_distribution', figure(1));

%% plot distributions of r1
close all;
figure; 
colour = [100/255,100/255,100/255];
histogram(all_r1, 'FaceColor',colour, 'FaceAlpha', 0.5, 'EdgeAlpha', 1)

% ylim([0, 40])

ax = gca; ax.XAxis.FontSize = 40; ax.YAxis.FontSize = 40; ax.YAxis.Exponent = 0;

ytickformat('%g')

xticks(0:10:40);

set(gca,'box', 'off');

ylabel('Frequency', 'FontName','Arial', 'Fontsize', 50);
% ylabel('Frequency of facial dissimilarity judgements', 'FontName','Arial', 'Fontsize', 40);
xlabel('r_{1}', 'FontName','Arial', 'Fontsize', 50);

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 10], 'PaperUnits', 'Inches', 'PaperSize', [20, 10])

save_figure_kmj('analysis/histograms/hist_r1_distribution', figure(1));

%% plot distributions of r2
close all;
figure; 
colour = [100/255,100/255,100/255];
histogram(all_r2, 'FaceColor',colour, 'FaceAlpha', 0.5, 'EdgeAlpha', 1)

% ylim([0, 40])

ax = gca; ax.XAxis.FontSize = 40; ax.YAxis.FontSize = 40; ax.YAxis.Exponent = 0;

ytickformat('%g')

xticks(0:10:40);

set(gca,'box', 'off');

ylabel('Frequency', 'FontName','Arial', 'Fontsize', 50);
% ylabel('Frequency of facial dissimilarity judgements', 'FontName','Arial', 'Fontsize', 40);
xlabel('r_{2}', 'FontName','Arial', 'Fontsize', 50);

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 10], 'PaperUnits', 'Inches', 'PaperSize', [20, 10])

save_figure_kmj('analysis/histograms/hist_r2_distribution', figure(1));

%% ttest if distributions are statistically differnt from each other: above and below identity line theta, r1, r2, r1-r2
significant_difference_eu_dist_below_above_identity_line = ttest2(mean_eu_dist_below_identity_line, mean_eu_dist_above_identity_line);

significant_difference_theta_below_above_identity_line = ttest2(mean_theta_below_identity_line, mean_theta_above_identity_line);

significant_difference_r1_below_above_identity_line = ttest2(mean_r1_below_identity_line, mean_r1_above_identity_line);

significant_difference_r2_below_above_identity_line = ttest2(mean_r2_below_identity_line, mean_r2_above_identity_line);

significant_difference_mean_r1_r2_below_above_identity_line = ttest2(mean_mean_r1_r2_below_identity_line, mean_mean_r1_r2_above_identity_line);

sign_diff_abs_differnece_r1_r2_below_above_identity_line = ttest2(abs_difference_r1_r2_below_identity_line, abs_difference_r1_r2_above_identity_line);

%% (1) plot the similarity judgments
%% as a function of Euclidean distance
%% group

figure('Name', 'Face similarity as a function of Basel Face Space');
plot(all_eu_distances, all_similarities, 'ro','MarkerSize',2);
title('Face similarity as a function of Basel Face Space');
xlabel('Euclidean distance');
ylabel('Similarity judgements');

%% each subject

figure('Name', 'Face similarity as a function of Basel Face Space for each subject');

hold on;

for i = 1:length(all_subjects)
    subject = all_subjects(i);
    [r1, r2, theta, eu_d, sim, sep] = extract_results(subject.sessions_combined);
    subplot(5, 6, i);
    plot(eu_d, sim, 'r.', 'MarkerSize',2);
    xlim([0 80]);
    set(gca,'XTickLabel', []);
    set(gca,'YTickLabel', []);
    title(subject.name);
end
mtit(gcf, 'Face similarity as a function of Basel Face Space')

%% as a function of R1, R2, and theta (one 3D scatterplot with colour coding of perceptual similarity)
%% group
figure('Name', 'Face similarity as a function of R1, R2 and theta');
hold on;
scatter3(all_r1,all_r2,all_theta, 50, all_similarities,'.');
% plot other half of space (r1 and r2 have symmetric roles)
scatter3(all_r2,all_r1,all_theta, 50, all_similarities,'.');
xlabel('R1');
ylabel('R2');
zlabel('Theta');
title('Face similarity as a function of R1, R2 and theta')
colorbar;
view([-1 -1.4 1]);

%% each subject
figure('Name', 'Face similarity as a function of R1, R2 and theta for each subject');
for i = 1:length(all_subjects)
    subject = all_subjects(i);
    [r1, r2, theta, eu_d, sim, sep] = extract_results(subject.sessions_combined);
    subplot(6, 5, i);
    hold on;
    scatter3(all_r1,all_r2,all_theta, 10, all_similarities,'.');
    scatter3(all_r2,all_r1,all_theta, 10, all_similarities,'.');
    xlabel(subject.name);
    set(gca,'XTickLabel', []);
    set(gca,'YTickLabel', []);
    set(gca,'ZTickLabel', []);
    view([-1 -1.4 1]);
end
mtit(gcf,'Face similarity as a function of R1, R2 and theta')


%% each subject
figure ('Name','R1-by-R2 matrix for each theta bin for each subject');
hold on;
i = 0;
for s = 1:length(all_subjects)
    subject = all_subjects(s);
    [r1, r2, theta, eu_d, sim, sep] = extract_results(subject.sessions_combined);
    j = 0;
    for t = unique(theta)
        i = i + 1;
        j = j + 1;
        subplot(length(unique(theta)), length(all_subjects), s + length(all_subjects) * (j-1));
        plot_r1r2matrix(r1, r2, theta, sim, t);
        set(gca,'XTickLabel', []);
        set(gca,'YTickLabel', []);
        if s == 1
            ylabel(t);
        end
        if j == 1
            title(subject.name);
        end
    end
end

p = mtit(gcf,'R1-by-R2 matrix for each theta bin');
set(p.th,'Position',[0.5 1.05]);

%% %%%%%%%%%%%%%% MODELS %%%%%%%%%%%%%%%%%%%%%%
%% sigmoid BFS distance
[sigmoid_bfs_distance, sigmoid_bfs_distance_inv] = sigmoid_distance_fit(all_sessions);
 
%% testing how well the inverse worked
similarity_fake_data = [0.1:0.01:0.8]
eu_dist_fits = sigmoid_bfs_distance_inv(similarity_fake_data)
plot(eu_dist_fits, similarity_fake_data)

%% compute confidecne intervals of identity line
stderror_identity_separator = std(all_separators)/sqrt(26);
ts = tinv([0.025  0.975],length(all_separators)-1); % T-Score
CI = mean(all_separators) + ts*stderror_identity_separator; % Confidence Intervals

%% compute the facial similarity value of Eucliean distance that Kate got from her simulation
kate_simulation_eu_dist = 28.19;
kate_simulation_sim = sigmoid_bfs_distance(kate_simulation_eu_dist);

%% check whether Kate simultation number falls withing Euclidean distance of confidecne intervals of face identity judgements using the inverse of sigmoidal function
kate_simulation_eu_dist = 28.19;

mean_eu_dist = sigmoid_bfs_distance_inv(mean(all_separators));

CI_eu = sigmoid_bfs_distance_inv(CI);

%% plot linear and sigmoid BFS distance on the data
close all;
figure('Name', 'Face similarity as a function of Basel Face Space');
plot(all_eu_distances, all_similarities, 'ro','MarkerSize',2);
hold on;
plot_1d(linear_bfs_distance, '', 'b');
hold on;
plot_1d(sigmoid_bfs_distance, '', 'k');
xlabel('BFM Euclidean distance');
ylabel('Facial dissimilarity judgements');

set(gca,'box', 'off');
xlim([-0.2 80]);
xticks(0:10:80)

ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20;

save_figure_kmj('analysis/linear_and_sigmoidal_model_fit_over_data', figure(1));


%% plot linear and sigmoid BFS distance on the MEAN data
close all;
figure('Name', 'Face similarity as a function of Basel Face Space');
plot(eu_distances_of_face_pair_two_sessions_mean, similarity_of_face_pair_two_sessions_mean, 'r.','MarkerSize',15);
hold on;
plot_1d(linear_bfs_distance, '', 'b');
hold on;
plot_1d(sigmoid_bfs_distance, '', 'k');
xlabel('BFM Euclidean distance');
ylabel('Facial dissimilarity judgements');

set(gca,'box', 'off');
xlim([-0.2 80]);
xticks(0:10:80)

ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20;

save_figure_kmj('analysis/linear_and_sigmoidal_model_fit_over_mean_data', figure(1));

%% plot sigmoid BFS distance on the data with identity line with standard error 

close all;
figure;
mean_identity_separator_repeated = repelem(mean_identity_separator, length(all_eu_distances))
% line([0 80], [mean_identity_separator mean_identity_separator]); 
% hold on;
% line([0 80], [mean_identity_separator+stderror_identity_separator mean_identity_separator+stderror_identity_separator]);
% hold on;
% line([0 80], [mean_identity_separator-stderror_identity_separator mean_identity_separator-stderror_identity_separator]);
hold on;
sigmoid_bfs_distance = sigmoid_distance_fit(all_sessions);
plot(all_eu_distances, all_similarities, 'ro','MarkerSize',2);
hold on;

dark_green_colour = [0/255, 176/255, 80/255];
plot([0:length(mean_identity_separator_repeated)-1],mean_identity_separator_repeated, 'color',dark_green_colour, 'LineWidth',4);
plot([0:length(mean_identity_separator_repeated)-1],mean_identity_separator_repeated+stderror_identity_separator, 'color',dark_green_colour, 'LineWidth',1);
plot([0:length(mean_identity_separator_repeated)-1],mean_identity_separator_repeated-stderror_identity_separator, 'color',dark_green_colour, 'LineWidth',1);

plot_1d(sigmoid_bfs_distance, '', 'k');

xlabel('BFM Euclidean distance');
ylabel('Facial dissimilarity judgements');

set(gca,'box', 'off');
xlim([-0.2 80]);
xticks(0:10:80)

ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20;

save_figure_kmj('analysis/identity_separator_with_sigmoidal_model_fit_over_data', figure(1));

%% plot linear and sigmoid BFS distance on the MEAN data 

close all;
figure;

sigmoid_bfs_distance = sigmoid_distance_fit(all_sessions);
plot_1d(linear_bfs_distance, '',[224/255,224/255,224/255]);
hold on;
plot(eu_distances_of_face_pair_two_sessions_mean, similarity_of_face_pair_two_sessions_mean,'.','MarkerSize',15, 'MarkerEdgeColor',[180/255,180/255,180/255]);
hold on;
% plot_1d(sigmoid_bfs_distance, '', 'k');
plot_1d(sigmoid_bfs_distance, '', [120/255,120/255,120/255]);

xlabel('BFM Euclidean distance');
ylabel('Face dissimilarity judgements');

set(gca,'box', 'off');
xlim([-0.2 80]);
xticks(0:10:80)

ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20;

save_figure_kmj('analysis/linear_and_sigmoidal_model_fit_over_mean_data', figure(1));

%% plot sigmoid BFS distance with identity line with standard error 

close all;
figure;
mean_identity_separator_repeated = repelem(mean_identity_separator, length(all_eu_distances))
% line([0 80], [mean_identity_separator mean_identity_separator]); 
% hold on;
% line([0 80], [mean_identity_separator+stderror_identity_separator mean_identity_separator+stderror_identity_separator]);
% hold on;
% line([0 80], [mean_identity_separator-stderror_identity_separator mean_identity_separator-stderror_identity_separator]);
hold on;
sigmoid_bfs_distance = sigmoid_distance_fit(all_sessions);

dark_green_colour = [200/255,200/255,200/255];
plot([0:length(mean_identity_separator_repeated)-1],mean_identity_separator_repeated, 'color',dark_green_colour, 'LineWidth',4);
plot([0:length(mean_identity_separator_repeated)-1],mean_identity_separator_repeated+stderror_identity_separator, 'color',dark_green_colour, 'LineWidth',1);
plot([0:length(mean_identity_separator_repeated)-1],mean_identity_separator_repeated-stderror_identity_separator, 'color',dark_green_colour, 'LineWidth',1);

% plot_1d(sigmoid_bfs_distance, '', 'k');
plot_1d(sigmoid_bfs_distance, '', [120/255,120/255,120/255]);

xlabel('BFM Euclidean distance');
ylabel('Face dissimilarity judgements');

set(gca,'box', 'off');
xlim([-0.2 80]);
xticks(0:10:80)

ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20;

save_figure_kmj('analysis/identity_separator_with_sigmoidal_model_fit', figure(1));


%% plot linear and sigmoid BFS distance with identity line with standard error 

close all;
figure;
mean_identity_separator_repeated = repelem(mean_identity_separator, length(all_eu_distances))
% line([0 80], [mean_identity_separator mean_identity_separator]); 
% hold on;
% line([0 80], [mean_identity_separator+stderror_identity_separator mean_identity_separator+stderror_identity_separator]);
% hold on;
% line([0 80], [mean_identity_separator-stderror_identity_separator mean_identity_separator-stderror_identity_separator]);
hold on;
sigmoid_bfs_distance = sigmoid_distance_fit(all_sessions);
plot_1d(linear_bfs_distance, '',[224/255,224/255,224/255]);
plot(eu_distances_of_face_pair_two_sessions_mean, similarity_of_face_pair_two_sessions_mean,'.','MarkerSize',15, 'MarkerEdgeColor',[180/255,180/255,180/255]);

dark_green_colour = [0/255, 176/255, 80/255];
plot([0:length(mean_identity_separator_repeated)-1],mean_identity_separator_repeated, 'color',dark_green_colour, 'LineWidth',4);
plot([0:length(mean_identity_separator_repeated)-1],mean_identity_separator_repeated+stderror_identity_separator, 'color',dark_green_colour, 'LineWidth',1);
plot([0:length(mean_identity_separator_repeated)-1],mean_identity_separator_repeated-stderror_identity_separator, 'color',dark_green_colour, 'LineWidth',1);

% plot_1d(sigmoid_bfs_distance, '', 'k');
plot_1d(sigmoid_bfs_distance, '', [120/255,120/255,120/255]);

xlabel('BFM Euclidean distance');
ylabel('Facial dissimilarity judgements');

set(gca,'box', 'off');
xlim([-0.2 80]);
xticks(0:10:80)

ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20;

save_figure_kmj('analysis/identity_separator_with_linear_and_sigmoidal_model_fit_over_mean_data', figure(1));

%% gaussian-tuned exemplar code 
%% simpler closed-form models 

%% (2c) ...data-driven models
%% mean of k nearest neighbours in R1,R2,theta-space in the other session of the same subject (same stimuli, same subject)
% k = 10;
% k_nearest_bfs_distance = k_nearest_eu_distance(k, all_eu_distances, all_similarities);
% 
% plot_1d(k_nearest_bfs_distance, 'k nearest Euclidean distance');

%% mean of k nearest neighbours in R1,R2,theta-space in the other session of the another subject(same stimuli, different subject)

%% mean of k nearest neighbours in R1,R2,theta-space in the other half of the stimuli of the same subject (different stimuli, same subject)




%% (4) inferentially compare the models
% for each subject
% fit each model to half the data
% measure the predictive accuracy of the model as the coefficient of determination of the fitted model prediction of the other half of the data
% do this again after reversing training and test set and average the predictive accuracies
% do this in two generalisation variants
%   generalising across sessions (same stimuli, same subjects)
%   generalising across stimuli (fit with a random half of the stimuli using both sessions, test with the other half of the stimuli using both sessions)

models = [
    struct('name', 'Sigmoid BFS distance',  'fit', @sigmoid_distance_fit,   'extract', @extract_eu_distance)
    struct('name', 'Linear BFS distance',   'fit', @linear_distance_fit,    'extract', @extract_eu_distance)
];

across_sessions = zeros(length(all_subjects), length(models));
across_stimuli = zeros(length(all_subjects), length(models));

% Seeding the random number generator, so that random split of data is
% the same at every run
seed = RandStream('mt19937ar','Seed',0);

do_plot_individual_figs = 0;
if do_plot_individual_figs
    figure;
end
    
for s = 1:length(all_subjects)
    subject = all_subjects(s);
    
    % generalising across sessions (same stimuli, same subjects) 
    for m = 1:length(models)
        subplot(1,length(models), m);
        model = models(m);
        
        half1 = subject.sessions(1,:);
        half2 = subject.sessions(2,:);
        
        accuracy1 = prediction_accuracy(half1, half2, model, do_plot_individual_figs);
        accuracy2 = prediction_accuracy(half2, half1, model);
        across_sessions(s, m) = mean([accuracy1, accuracy2]);
    end
    
    % generalising across stimuli (fit with a random half of the stimuli using both sessions, test with the other half of the stimuli using both sessions)
    for m = 1:length(models)
        model = models(m);
        
        n = length(subject.sessions_combined);
        perm = randperm(seed, n);
        half1 = subject.sessions_combined(perm(1:n/2));
        half2 = subject.sessions_combined(perm(n/2+1:n));
        
        accuracy1 = prediction_accuracy(half1, half2, model);
        accuracy2 = prediction_accuracy(half2, half1, model);
        across_stimuli(s, m) = mean([accuracy1, accuracy2]);
    end
    
    % for each pair of models, store the difference in predictive accuracies in a matrix (models by models) for each of the generalisation variants
    all_subjects(s).mm_matrix_sessions = zeros(length(models));
    all_subjects(s).mm_matrix_stimuli = zeros(length(models));
    for m1 = 1:length(models)
        for m2 = 1:length(models)
            all_subjects(s).mm_matrix_sessions(m1, m2) = abs(across_sessions(s, m1) - across_sessions(s, m2));
            all_subjects(s).mm_matrix_stimuli(m1, m2) = abs(across_stimuli(s, m1) - across_stimuli(s, m2));
        end
    end
    
end

mean_across_sessions_sigmoid = mean(across_sessions(:,1))
mean_across_sessions_linear = mean(across_sessions(:,2))

mean_across_stimuli_sigmoid = mean(across_stimuli(:,1))
mean_across_stimuli_linear = mean(across_stimuli(:,2))

stderr_across_sessions_sigmoid = (std(across_sessions(:,1)))/number_of_subjects
stderr_across_sessions_linear = (std(across_sessions(:,2)))/number_of_subjects

stderr_across_stimuli_sigmoid = (std(across_stimuli(:,1)))/number_of_subjects
stderr_across_stimuli_linear = (std(across_stimuli(:,2)))/number_of_subjects

csvwrite('results/mean_across_sessions_sigmoid.csv', mean_across_sessions_sigmoid);
csvwrite('results/mean_across_sessions_linear.csv', mean_across_sessions_linear);

csvwrite('results/mean_across_stimuli_sigmoid.csv', mean_across_stimuli_sigmoid);
csvwrite('results/mean_across_stimuli_linear.csv', mean_across_stimuli_linear);

csvwrite('results/stderr_across_sessions_sigmoid.csv', stderr_across_sessions_sigmoid);
csvwrite('results/stderr_across_sessions_linear.csv', stderr_across_sessions_linear);

csvwrite('results/stderr_across_stimuli_sigmoid.csv', stderr_across_stimuli_sigmoid);
csvwrite('results/stderr_across_stimuli_linear.csv', stderr_across_stimuli_linear);

signrank(across_stimuli(:,1), across_stimuli(:,2))

%% Write results to TSV files

model_names = cell(length(models));
for m = 1:length(models)
    model_names{m} = models(m).name;
end
subject_names = cell(length(all_subjects));
for s = 1:length(all_subjects)
    subject_names{s} = all_subjects(s).name;
end

write_as_tsv('results/prediction_accuracy_across_sessions.tsv', across_sessions, model_names, subject_names)
write_as_tsv('results/prediction_accuracy_across_stimuli.tsv', across_stimuli, model_names, subject_names)


%% not sure what was been done here

[~, indexes] = sort(across_sessions, 2);
best_model = cell(length(all_subjects), length(models));
for s = 1:length(all_subjects)
    subject = all_subjects(s); 
    % generalising across sessions (same stimuli, same subjects)
    for m = 1:length(models)
        s
        m
        indexes(s, m)
        model_names{indexes(s, m)}
        best_model(s, m)
        best_model
        best_model{s, m} = model_names{indexes(s, m)};
%         model = models(m);
    end
end
write_as_tsv('results/best_model_across_sessions.tsv', best_model, model_names, subject_names)


%% for each pair of models, use a signed-rank test to test if the median of the differences in predictive accuracies across subjects is greater than 0

signranks = zeros(length(models));
differences = zeros(length(models));
ratios = ones(length(models));
ratios1 = ones(length(models));
for m1 = 1:length(models)
    for m2 = 1:length(models)
        if m1 ~= m2
            signranks(m1, m2) = signrank(across_sessions(:, m1), across_sessions(:, m2));
            differences(m1, m2) = mean(across_sessions(:, m1) - across_sessions(:, m2));
            ratios(m1, m2) = mean(across_sessions(:, m1) ./ across_sessions(:, m2));
            ratios1(m1, m2) = mean(across_sessions(:, m1) ./ (across_sessions(:, m1) + across_sessions(:, m2)));
        end
    end
end

write_as_tsv('results/signranks.tsv', signranks, model_names, model_names)
write_as_tsv('results/differences.tsv', differences, model_names, model_names)
write_as_tsv('results/ratios.tsv', ratios, model_names, model_names)
write_as_tsv('results/ratios1.tsv', ratios1, model_names, model_names)

% are subjects significantly distinct?



%% Isotropicity/uniformity same stimuli test
% from Kate: The simplest test of this "uniformity" would probably be to 
% calculate whether there's a significant correlation between Theta and 
% Dissimilarity within each of these bins (I'd guess probably not, even with 
% these fairly wide bins - and you could also try with bins half the size). 
% I think it would be fine to include a plot like this in the Supplementary 
% Materials (or main ms maybe), and report that there are no significant 
% correlations between Theta and Dissimilarity within bins of face pairs with 
% similar Euclidean distance (if that turns out to be true), and that 
% therefore we find no evidence that the space is not uniform.

% figure bins_eu_dist_theta_sim
figure;
% n_bins = 5;
n_bins = 10;
cmap = distinguishable_colors(n_bins);
[bins, edges] = discretize(eu_distances_of_face_pair(1,:), n_bins);
colors = cmap(bins, :);
% scatter(mean_similarity_of_face_pair, theta_of_face_pair(1,:), 100, colors, 'filled');
g = {edges(bins), edges(bins+1)};
h = gscatter(mean_similarity_of_face_pair, theta_of_face_pair(1,:), g);
htitle = get(findobj(gcf, 'Type', 'Legend'),'Title');
set(htitle,'String','Eu distance');
xlabel('Dissimilarity');
ylabel('Theta');

save_figure_kmj('analysis/uniformity_test/bins_eu_dist_theta_sim')

%% figure correlation_theta_dissim_bins

correlations = nan(number_of_subjects*2, n_bins);
clear p h stats
for bin = 1:10
    pair_indexes = bins == bin;
    correlations(:,bin) = corr(theta_of_face_pair(1,pair_indexes)', similarity_of_face_pair(:,pair_indexes)');
    stderr_correlations(:,bin) = std(correlations(:,bin))/(number_of_subjects*2);
    mean_correlations(:,bin) = mean(correlations(:,bin));
    [p(bin),h(bin), stats(bin)] = signrank(correlations(:,bin));
end

figure;

max_y_position = max(correlations+stderr_correlations);


close all;
figure;

bar(mean_correlations, 'FaceColor', [0.5 0.5 0.5]);
hold on;

e = errorbar(1:bin, mean_correlations, stderr_correlations, '.', 'CapSize',0, 'LineWidth', 1);
e.Color = 'black';
e.Marker = 'none';

model_names = {'0,8','8,16','16,24','24,32','32,40','40,48','48,56','56,64','64,72', '72,80'};
model_names_with_significance = {};
for model_number = 1:length(model_names)
    if p(model_number)<0.05
        model_names_with_significance{model_number} = [model_names{model_number}, ' *'];
    else
        model_names_with_significance{model_number} = [model_names{model_number}, '  '];
    end
end


set(gca,'FontName','Arial','TickLength',[0 0],'XTick',...
    1:length(model_names),'XTickLabel',... 
    model_names_with_significance, 'XTickLabelRotation',45); 
    ylabel('R between theta and dissimilarity', 'FontName','Arial');
    xlabel('Eu dist bins', 'FontName','Arial');

set(gca,'box', 'off');

% yticks([0, 0.005, 0.010,0.015]);
% 
% ylim([0 0.2]);
% xlim([-0.01 0]);

ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20;
ax.YAxis.Exponent = 0
% xtickformat('%.0f')

save_figure_kmj('analysis/uniformity_test/correlation_theta_dissim_bins')
     


