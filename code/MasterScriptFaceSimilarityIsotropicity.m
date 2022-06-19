
%% PRELIM: load data iso

load('data/all_data_combined/all_similarities_iso.mat');

load('data/all_data_combined/all_subjects_iso.mat');
load('data/all_data_combined/all_sessions_iso.mat');
load('data/all_data_combined/all_separators_iso.mat');
load('data/all_data_combined/all_eu_distances_iso.mat');
load('data/all_data_combined/all_r1_iso.mat');
load('data/all_data_combined/all_r2_iso.mat');
load('data/all_data_combined/all_separators_iso.mat');
load('data/all_data_combined/all_theta_iso.mat');
load('data/all_data_combined/all_pair_ids_iso.mat');

%% PRELIM: load data non_iso

load('data/all_data_combined/all_subjects.mat');
load('data/all_data_combined/all_pair_ids.mat');
load('data/all_data_combined/all_similarities.mat');
load('data/all_data_combined/all_eu_distances.mat');
load('data/all_data_combined/all_r1.mat');
load('data/all_data_combined/all_r2.mat');
load('data/all_data_combined/all_separators.mat');
load('data/all_data_combined/all_theta.mat');

all_subjects_no_iso = all_subjects;
all_r1_no_iso = all_r1;
all_r2_no_iso = all_r2;
all_theta_no_iso = all_theta;
all_eu_no_iso = all_eu_distances;
all_similarities_no_iso = all_similarities;
all_separators_no_iso = all_separators;

%% PRELIM: get similarities for face pairs

for face_pair = 1:232
    indeces_of_face_pair_iso = all_pair_ids_iso == face_pair;
    
    similarity_iso = all_similarities_iso(indeces_of_face_pair_iso);
    similarity_of_face_pair_iso(:,face_pair) = similarity_iso;
    mean_similarity_of_face_pair_iso(face_pair) = mean(similarity_iso);
    std_similarity_of_face_pair_iso(face_pair) = std(similarity_iso);
end

save_mat_kmj('analysis/similarity_of_face_pair_iso.mat', 'similarity_of_face_pair_iso');

% [~, order_similarity] = sort(similarity_of_face_pair_iso);

for subj = 1:15
    [~, order_similarity_per_subj(subj,:)] = sort(similarity_of_face_pair_iso(subj,:));
end
[ordered_mean_similarity_of_face_pair, order_similarity] = sort(mean_similarity_of_face_pair_iso);

%% PRELIM: get eu distances for face pairs

for face_pair = 1:232
    indeces_of_face_pair_iso = all_pair_ids_iso == face_pair;
    
    eu_distances_iso = all_eu_distances_iso(indeces_of_face_pair_iso);
    eu_distances_of_face_pair_iso(:,face_pair) = eu_distances_iso;
    mean_eu_distances_of_face_pair_iso(face_pair) = mean(eu_distances_iso);
end

save_mat_kmj('analysis/mean_eu_distances_of_face_pair_iso.mat', 'mean_eu_distances_of_face_pair_iso');

%% PRELIM: get theta for face pairs
for face_pair = 1:232
    indeces_of_face_pair_iso = all_pair_ids_iso == face_pair;
    
    theta_iso = all_theta_iso(indeces_of_face_pair_iso);
    theta_of_face_pair_iso(:,face_pair) = theta_iso;
    mean_theta_of_face_pair_iso(face_pair) = mean(theta_iso);
   
end

%% PRELIM: get r1 for face pairs
for face_pair = 1:232
    indeces_of_face_pair_iso = all_pair_ids_iso == face_pair;
    
    r1_iso = all_r1_iso(indeces_of_face_pair_iso);
    r1_of_face_pair_iso(:,face_pair) = r1_iso;
    mean_r1_of_face_pair_iso(face_pair) = mean(r1_iso);
    
end

%% PRELIM: get r2 for face pairs
for face_pair = 1:232
    indeces_of_face_pair_iso = all_pair_ids_iso == face_pair;
    
    r2_iso = all_r2_iso(indeces_of_face_pair_iso);
    r2_of_face_pair_iso(:,face_pair) = r2_iso;
    mean_r2_of_face_pair_iso(face_pair) = mean(r2_iso);
    
end


%% PRELIM: Compute mean correlation between subjects across participants
similarity_of_face_pair_corr_pairwise_across_subjects_iso = corr(similarity_of_face_pair_iso');
mean_similarity_of_face_pair_corr_iso = mean(similarity_of_face_pair_corr_pairwise_across_subjects_iso(:));


%% PRELIM: get necessary data
file_folder_kate = 'data/from_kate/facepair_info_144px_isotropicity_expt_pairs_png';
number_of_face_pairs = 232;
filenames_similarity = {};
for i = 1: length(order_similarity)
    filenames_similarity{i} = [file_folder_kate,'/pair_' num2str(order_similarity(i),'%03.f') '.png'];
end


%% PRELIM: define mean identity separator and find after how many face pairs it is places sampling every 20 face pairs

mean_identity_separator_iso = mean(all_separators_iso);

stderror_identity_separator_iso = std(all_separators_iso)/sqrt(15);

index_of_identity_separator = find(ordered_mean_similarity_of_face_pair<mean_identity_separator_iso);

index_of_identity_separator_max = max(index_of_identity_separator);


%% PRELIM: get thetas, r1, r2 for faces below average idnetity line
[indeces_all_similarities_iso_below_identity_line]=find(all_similarities_iso<mean_identity_separator_iso);

all_similarities_iso_below_identity_line = all_similarities_iso(indeces_all_similarities_iso_below_identity_line);

all_theta_iso_below_identity_line = all_theta_iso(indeces_all_similarities_iso_below_identity_line);

all_r1_iso_below_identity_line = all_r1_iso(indeces_all_similarities_iso_below_identity_line);

all_r2_iso_below_identity_line = all_r2_iso(indeces_all_similarities_iso_below_identity_line);

%% PRELIM: get MEAN Eu dist, thetas, r1, r2 for faces below average idnetity line

[indeces_mean_similarities_above_identity_line_iso]=find(mean_similarity_of_face_pair_iso>mean_identity_separator_iso);

[indeces_mean_similarities_below_identity_line_iso]=find(mean_similarity_of_face_pair_iso<mean_identity_separator_iso);

mean_theta_below_identity_line_iso = mean_theta_of_face_pair_iso(indeces_mean_similarities_below_identity_line_iso);

mean_r1_below_identity_line_iso = mean_r1_of_face_pair_iso(indeces_mean_similarities_below_identity_line_iso);

mean_r2_below_identity_line_iso = mean_r2_of_face_pair_iso(indeces_mean_similarities_below_identity_line_iso);
 
mean_eu_dist_below_identity_line_iso = mean_eu_distances_of_face_pair_iso(indeces_mean_similarities_below_identity_line_iso);

mean_similarities_below_identity_line_iso = mean_similarity_of_face_pair_iso(indeces_mean_similarities_below_identity_line_iso);

mean_similarities_above_identity_line_iso = mean_similarity_of_face_pair_iso(indeces_mean_similarities_above_identity_line_iso);

mean_theta_above_identity_line_iso = mean_theta_of_face_pair_iso(indeces_mean_similarities_above_identity_line_iso);

mean_r1_above_identity_line_iso = mean_r1_of_face_pair_iso(indeces_mean_similarities_above_identity_line_iso);

mean_r2_above_identity_line_iso = mean_r2_of_face_pair_iso(indeces_mean_similarities_above_identity_line_iso);

mean_eu_dist_above_identity_line_iso = mean_eu_distances_of_face_pair_iso(indeces_mean_similarities_above_identity_line_iso);

mean_mean_r1_r2_above_identity_line_iso = (mean_r1_above_identity_line_iso+mean_r2_above_identity_line_iso)/2;

mean_mean_r1_r2_below_identity_line_iso = (mean_r1_below_identity_line_iso+mean_r2_below_identity_line_iso)/2;

abs_difference_r1_r2_above_identity_line_iso = abs(mean_r1_above_identity_line_iso-mean_r2_above_identity_line_iso);

abs_difference_r1_r2_below_identity_line_iso = abs(mean_r1_below_identity_line_iso-mean_r2_below_identity_line_iso);


%% PRELIM: logistic regression and dprime (based on individual subjects data)

number_of_subjects_iso = 15;

dprimes_eu_d_iso= [];
dprimes_theta_iso = [];
dprimes_abs_r1_r2_iso = [];
AUC_eu_d_iso = [];
AUC_theta_iso = [];
AUC_abs_r1_r2_iso = [];


pairs_thetas_iso = [];
pairs_eu_distances_iso = [];
pairs_labels_iso = [];
pairs_r1_iso = [];
pairs_r2_iso = [];

pair_ids_iso = [];

for s = 1:number_of_subjects_iso
    
    subject = all_subjects_iso(s);
    
    pair_number = 0;
    
    for session = 1:1
        for trial = 1:29
            identity_line_position = subject.sessions(session, trial).facePositions(1,2);
            for pair = 2:9
                pair_number = pair_number + 1;
                pair_position = subject.sessions(session, trial).facePositions(pair,2);
                pair_id = subject.sessions(session, trial).PairOrderAndID(pair-1);
                geodat = subject.sessions(session, trial).geometricInfo(pair).polarRelations;
                r1_iso=geodat(1); r2_iso=geodat(2); theta_deg_iso = geodat(3); 
                if r1_iso==0 || r2_iso==0
                    theta_deg_iso = nan;
                    r1_iso = nan;
                    r2_iso = nan;
                    eu_d_iso = nan;
                end
                eu_d_iso = sqrt(r1_iso^2 + r2_iso^2 - 2*r1_iso*r2_iso*cosd(theta_deg_iso));
                pair_ids_iso(s, pair_number) = pair_id;
                pairs_r1_iso(s, pair_number) = r1_iso;
                pairs_r2_iso(s, pair_number) = r2_iso;
                pairs_thetas_iso(s, pair_number) = theta_deg_iso;
                pairs_eu_distances_iso(s, pair_number) = eu_d_iso;
                if pair_position <= identity_line_position
                    pairs_labels_iso(s, pair_number) = 1;
                else
                    pairs_labels_iso(s, pair_number) = 2;
                end
            end
        end
    end
    
    [dprimes_eu_d_iso(s), AUC_eu_d_iso(s)] = dprime_and_roc(pairs_eu_distances_iso(s,:)', pairs_labels_iso(s,:)');
    [dprimes_theta_iso(s), AUC_theta_iso(s)] = dprime_and_roc(pairs_thetas_iso(s,:)', pairs_labels_iso(s,:)');
    [dprimes_abs_r1_r2_iso(s), AUC_abs_r1_r2_iso(s)] = dprime_and_roc(abs(pairs_r1_iso(s,:)-pairs_r2_iso(s,:))', pairs_labels_iso(s,:)');
end

indeces_theta_180_iso = find(all_theta_iso ==180 & all_r1_iso ==0 | all_r2_iso ==0);
%% PRELIM: count what is the percentage of faces classified as the same identity per face pair

% sort
n_pairs = length(order_similarity);
reordered_pairs_labels_iso = [];
for s = 1:number_of_subjects_iso
    for p = 1:n_pairs
        pair_id = order_similarity(p);
        index_of_pair_in_sess_1 = find(pair_ids_iso(s,:) == pair_id);
        pair_number = (session-1)*n_pairs + p;
        reordered_pairs_labels_iso(s, pair_number) = pairs_labels_iso(s, index_of_pair_in_sess_1);
    end
end

frequency_pairs_below_identity_line  = sum(reordered_pairs_labels_iso == 1);

frequency_pairs_below_identity_line_both_sess_norm = frequency_pairs_below_identity_line  /15;

frequency_every_20  = frequency_pairs_below_identity_line_both_sess_norm(1:20:end);

%% PRELIM: linear and sigmoid BFS distance
linear_bfs_distance = linear_distance_fit(all_sessions_iso);
sigmoid_bfs_distance = sigmoid_distance_fit(all_sessions_iso);

%% PRELIM: get pairs below and above identity line

thetas_below_identity_line_iso = pairs_thetas_iso(pairs_labels_iso==1);
thetas_above_identity_line_iso = pairs_thetas_iso(pairs_labels_iso==2);

eu_distances_below_identity_line_iso = pairs_eu_distances_iso(pairs_labels_iso==1);
eu_distances_above_identity_line_iso = pairs_eu_distances_iso(pairs_labels_iso==2);

r1_below_identity_line_iso = pairs_r1_iso(pairs_labels_iso==1);
r1_above_identity_line_iso = pairs_r1_iso(pairs_labels_iso==2);
r2_below_identity_line_iso = pairs_r2_iso(pairs_labels_iso==1);
r2_above_identity_line_iso = pairs_r2_iso(pairs_labels_iso==2);
abs_diff_r1_r2_below_identity_line_iso = abs(r1_below_identity_line_iso - r2_below_identity_line_iso);
abs_diff_r1_r2_above_identity_line_iso = abs(r1_above_identity_line_iso - r2_above_identity_line_iso);



%% arrange face pairs in montage showing every 20 face pairs for similarity judgements
figure;
montage(filenames_similarity(1:20:number_of_face_pairs), 'Size', [12,1]);

save_figure_kmj('analysis/face_ranking/face_similarity_ranking_every_20_iso');

%% save kate's stimuli as pngs
file_folder_kate = 'data/from_kate/facepair_info_144px_isotropicity_expt/';
number_of_face_pairs = 232;
filenames_similarity_kate = {};
for i = 1: number_of_face_pairs
    load([file_folder_kate,'/pair' num2str(i,'%06.f') '.mat']);
%     pair{i} = imshow([img1, img2]);
    imwrite([img1, img2], ['data/from_kate/facepair_info_144px_isotropicity_expt_pairs_png/pair_' num2str(i,'%03.f'), '.png']);
end

%% arrange face pairs in montage showing every 20 face pairs for similarity judgements using kate's stimuli
file_folder_kate = 'data/from_kate/facepair_info_144px_isotropicity_expt_pairs_png/';
number_of_face_pairs = 232;
filenames_similarity_kate = {};
for i = 1: length(order_similarity)
    filenames_similarity_kate{i} = [file_folder_kate,'/pair_' num2str(order_similarity(i),'%03.f') '.png'];
end

figure;
montage(filenames_similarity_kate(fliplr(1:20:number_of_face_pairs)), 'Size', [12,1]);

save_figure_kmj('analysis/face_ranking/face_similarity_ranking_every_20_iso_kate_face_pairs');

%% arrange face pairs in one big montage
number_of_face_pairs = 232;
filenames_similarity_kate = {};
for i = 1: length(order_similarity)
    filenames_similarity_kate{i} = [file_folder_kate,'/pair_' num2str(order_similarity(i),'%03.f') '.png'];
end

figure;
montage(filenames_similarity_kate(fliplr(1:(number_of_face_pairs))), 'Size', [20, 12]);

save_figure_kmj('analysis/montage_similarity_iso_kate_face_pairs');

%% selecting only face pairs that were classified as te same similarity
figure;
all_separators_mean_iso = mean(all_separators_iso)
number_of_face_pairs_same_identity = sum(mean_similarity_of_face_pair_iso<all_separators_mean_iso)
montage(filenames_similarity_kate(1:(number_of_face_pairs_same_identity)), 'Size', [11,13]);

save_figure_kmj('analysis/montage_similarity_iso_only_same_identity_pairs');


%% arrange face pairs in montage showing every 20 face pairs for VGG face WEIGHTED

VGG_face_distances = csvread('data/model_predictions_before_revision/isotropicity_expt/01_VGG-Face_best.csv');


[~, order_VGG_face] = sort(VGG_face_distances);

number_of_face_pairs = 232;
file_folder_kate = 'data/from_kate/facepair_info_144px_isotropicity_expt_pairs_png/';
filenames_VGG_face_kate = {};
for i = 1: length(order_VGG_face)
    filenames_VGG_face_kate{i} = [file_folder_kate,'/pair_' num2str(order_VGG_face(i),'%03.f') '.png'];
end

figure;
montage(filenames_VGG_face_kate(fliplr(1:20:number_of_face_pairs)), 'Size', [12,1]);

save_figure_kmj('analysis/face_ranking/VGG_face_ranking_every_20_weighted_iso_kate_face_pairs');


%% arrange face pairs in montage showing every 20 face pairs for VGG face WEIGHTED

VGG_face_distances = csvread('data/model_predictions_before_revision/isotropicity_expt/01_VGG-Face_best.csv');


[~, order_VGG_face] = sort(VGG_face_distances);

number_of_face_pairs = 232;
file_folder = 'data/face_pair_images_iso/';
filenames_VGG_face = {};
for i = 1: length(order_VGG_face)
    filenames_VGG_face{i} = [file_folder,'/pair_' num2str(order_VGG_face(i),'%03.f') '.tif'];
end

figure;
montage(filenames_VGG_face(fliplr(1:20:number_of_face_pairs)), 'Size', [12,1]);

save_figure_kmj('analysis/face_ranking/VGG_face_ranking_every_20_weighted_iso');


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

for subj = 1:15
    single_subj_corr_with_mean_across_subj(subj) = corr2(order_similarity_per_subj(subj, :), mean_similarity_judgements);
end

mean_single_subj_corr_with_mean_across_subj = mean(single_subj_corr_with_mean_across_subj);

save_mat_kmj('analysis/correlations/pair_ranking_correlations_mean_subject_other_subjects_iso', 'mean_single_subj_corr_with_mean_across_subj');

%% correlation between subjects with the VGG face (upperbound of the noise ceiling, more comparable to the case with VGG16)

single_subj_corr_with_VGG_face = [];

for subj = 1:15
    single_subj_corr_with_VGG_face(subj) = corr2(order_similarity_per_subj(subj, :), order_VGG_face);
end

mean_single_subj_corr_with_VGG_face = mean(single_subj_corr_with_VGG_face);

save_mat_kmj('analysis/correlations/pair_ranking_correlations_mean_subject_VGG_face_iso', 'mean_single_subj_corr_with_VGG_face');

%% logistic regression and dprime (based on mean data)

% create input to logistic regression
X_iso = mean_theta_of_face_pair_iso';

Y_iso = nan(232,1);
Y_iso(indeces_mean_similarities_below_identity_line_iso)=1;
Y_iso(indeces_mean_similarities_above_identity_line_iso)=2;

[dprime_iso, AU_isoC] = dprime_and_roc(X_iso, Y_iso)

%% plot colorbar of frequency every 20 pair
figure;
colormap(gray(256));
imagesc(frequency_every_20');
colorbar;
caxis([0 1]);
colorbar('southoutside');
save_figure_kmj('analysis/frequency_same_identity/frequency_of_same_identity_iso_every_20_face_pair_colorbar');

%% plot dprimes

mean_dprimes_eu_d_iso = nanmean(dprimes_eu_d_iso);
mean_dprimes_theta_iso = nanmean(dprimes_theta_iso);
mean_dprimes_abs_r1_r2_iso = nanmean(dprimes_abs_r1_r2_iso);

stderr_dprimes_eu_d_iso = (nanstd(dprimes_eu_d_iso))/number_of_subjects_iso;
stderr_dprimes_theta_iso = (nanstd(dprimes_theta_iso))/number_of_subjects_iso;
stderr_dprimes_abs_r1_r2_iso = (nanstd(dprimes_abs_r1_r2_iso))/number_of_subjects_iso;

close all;
figure;

mean_dprime = [mean_dprimes_eu_d_iso, mean_dprimes_theta_iso, mean_dprimes_abs_r1_r2_iso];

stderr_dprime = [stderr_dprimes_eu_d_iso, stderr_dprimes_theta_iso, stderr_dprimes_abs_r1_r2_iso];

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

save_figure_kmj('analysis/dprimes/dprimes_iso', figure(1));

%% plot AUC

mean_AUC_eu_d_iso = nanmean(AUC_eu_d_iso);
mean_AUCs_theta_iso = nanmean(AUC_theta_iso);
mean_AUC_abs_r1_r2_iso = nanmean(AUC_abs_r1_r2_iso);

stderr_AUC_eu_d_iso = (nanstd(AUC_eu_d_iso))/number_of_subjects_iso;
stderr_AUC_theta_iso = (nanstd(AUC_theta_iso))/number_of_subjects_iso;
stderr_AUC_abs_r1_r2_iso = (nanstd(AUC_abs_r1_r2_iso))/number_of_subjects_iso;

close all;
figure;

mean_AUC = [mean_AUC_eu_d_iso, mean_AUCs_theta_iso, mean_AUC_abs_r1_r2_iso];

stderr_AUC = [stderr_AUC_eu_d_iso, stderr_AUC_theta_iso, stderr_AUC_abs_r1_r2_iso];

bar(mean_AUC, 'FaceColor', [0.5 0.5 0.5]);
hold on;

% sigstar(significance_matrix_unique_variance_fdr_corrected_indeces,max_y_position,[], 1);

hold on;

number_of_models = 3;

e = errorbar(1:number_of_models, mean_AUC, stderr_AUC, '.', 'CapSize',0, 'LineWidth', 6, 'CapSize' , 6);
e.Color = 'black';
e.Marker = 'none';

ax = gca; ax.XAxis.FontSize = 25; ax.YAxis.FontSize = 25;

model_names  = {'Euclidean distance','Theta','|r_{1} - r_{2}|'};
set(gca,'FontName','Arial','TickLength',[0 0],'XTick',...
    1:number_of_models,'XTickLabel',... 
    model_names, 'XTickLabelRotation',45); 
    ylabel('Area under ROC Curve ', 'FontName','Arial');

set(gca,'box', 'off');

% ylim([0 0.025]);



save_figure_kmj('analysis/AUC/AUC_iso', figure(1));

%% plot histogram for MEAN theta below and above identity line
close all;
figure;
colour_above_identity_line = [180/255,180/255,180/255];
colour_below_identity_line = [100/255,100/255,100/255];

bins = 6;
histogram(thetas_below_identity_line_iso, bins, 'FaceColor',colour_above_identity_line, 'FaceAlpha', 0.5, 'EdgeAlpha', 1);
hold on;
histogram(thetas_above_identity_line_iso, bins, 'FaceColor',colour_below_identity_line, 'FaceAlpha', 0.5, 'EdgeAlpha', 1);

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

ylim([0, 1200]);

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 10], 'PaperUnits', 'Inches', 'PaperSize', [20, 10])

save_figure_kmj('analysis/histograms/hist_same_differnt_identity_theta_iso', figure(1));

%% plot histogram for MEAN absolute difference between r1 and r2 below and above identity line
close all;
figure;

colour_above_identity_line = [180/255,180/255,180/255];
colour_below_identity_line = [100/255,100/255,100/255];
bins = 6;
histogram(abs_diff_r1_r2_below_identity_line_iso, bins, 'FaceColor',colour_above_identity_line, 'FaceAlpha', 0.5, 'EdgeAlpha', 1);
hold on;
histogram(abs_diff_r1_r2_above_identity_line_iso, bins, 'FaceColor',colour_below_identity_line, 'FaceAlpha', 0.5, 'EdgeAlpha', 1);

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

save_figure_kmj('analysis/histograms/hist_same_differnt_identity_abs_difference_r1_r2_iso', figure(1));

%% plot histogram for MEAN eu dist below and above identity line
close all;
figure;
bins = 6;
colour_above_identity_line = [180/255,180/255,180/255];
colour_below_identity_line = [100/255,100/255,100/255];

histogram(eu_distances_below_identity_line_iso, bins, 'BinWidth',15, 'FaceColor',colour_above_identity_line, 'FaceAlpha', 0.5, 'EdgeAlpha', 1);
hold on;
histogram(eu_distances_above_identity_line_iso, bins,'BinWidth',15,  'FaceColor',colour_below_identity_line, 'FaceAlpha', 0.5, 'EdgeAlpha', 1);

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

save_figure_kmj('analysis/histograms/hist_same_differnt_identity_eu_dist_iso', figure(1));

%% plot distributions of theta
close all;
figure; 
colour = [100/255,100/255,100/255];
indeces_r1_r2_zero_iso = find(all_r1_iso ==0 | all_r2_iso ==0);
all_theta_iso(indeces_r1_r2_zero_iso) = nan;
histogram(all_theta_iso,'FaceColor',colour, 'FaceAlpha', 0.5, 'EdgeAlpha', 1)


% ylim([0, 40])

ax = gca; ax.XAxis.FontSize = 40; ax.YAxis.FontSize = 40; ax.YAxis.Exponent = 0;

ytickformat('%g')

xticks(0:30:180);

set(gca,'box', 'off');

ylabel('Frequency', 'FontName','Arial', 'Fontsize', 50);
% ylabel('Frequency of facial dissimilarity judgements', 'FontName','Arial', 'Fontsize', 40);
xlabel('Theta', 'FontName','Arial', 'Fontsize', 50);

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 10], 'PaperUnits', 'Inches', 'PaperSize', [20, 10])

save_figure_kmj('analysis/histograms/hist_theta_distribution_iso', figure(1));

%% plot distributions of r1
close all;
figure; 
colour = [100/255,100/255,100/255];
histogram(all_r1_iso, 'FaceColor',colour, 'FaceAlpha', 0.5, 'EdgeAlpha', 1)

% ylim([0, 40])

ax = gca; ax.XAxis.FontSize = 40; ax.YAxis.FontSize = 40; ax.YAxis.Exponent = 0;

ytickformat('%g')

xticks(0:10:40);

set(gca,'box', 'off');

ylabel('Frequency', 'FontName','Arial', 'Fontsize', 50);
% ylabel('Frequency of facial dissimilarity judgements', 'FontName','Arial', 'Fontsize', 40);
xlabel('r_{1}', 'FontName','Arial', 'Fontsize', 50);

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 10], 'PaperUnits', 'Inches', 'PaperSize', [20, 10])

save_figure_kmj('analysis/histograms/hist_r1_distribution_iso', figure(1));

%% plot distributions of r2
close all;
figure; 
colour = [100/255,100/255,100/255];
bar(all_r2_iso, 'FaceColor',colour, 'FaceAlpha', 0.5, 'EdgeAlpha', 1)

% ylim([0, 40])

ax = gca; ax.XAxis.FontSize = 40; ax.YAxis.FontSize = 40; ax.YAxis.Exponent = 0;

ytickformat('%g')

xticks(0:10:40);

set(gca,'box', 'off');

ylabel('Frequency', 'FontName','Arial', 'Fontsize', 50);
% ylabel('Frequency of facial dissimilarity judgements', 'FontName','Arial', 'Fontsize', 40);
xlabel('r_{2}', 'FontName','Arial', 'Fontsize', 50);

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 10], 'PaperUnits', 'Inches', 'PaperSize', [20, 10])

save_figure_kmj('analysis/histograms/hist_r2_distribution_iso', figure(1));

%% confidence intervals identity separator

SEM = std(all_separators_iso)/sqrt(length(all_separators_iso));% Standard Error
ts = tinv([0.05  0.95],length(all_separators_iso)-1);      % T-Score
CI = mean(all_separators_iso) + ts*SEM     

%% %%%%%%%%%%%%%% MODELS %%%%%%%%%%%%%%%%%%%%%%
%% sigmoid BFS distance and inverse
[sigmoid_bfs_distance, sigmoid_bfs_distance_inv] = sigmoid_distance_fit(all_sessions_iso);

%% test whether identity separator has a Euclidean distance that is significantly different from Kate's simulation
% Comparison whether the simulation that Kate run falls within the 95% confidence intervals, basically compute confidence intervals on identity line across two sessions in iso and main experiment and then see whether this mean value that Kate calculated (29 or something like that) falls within this values range for both non-iso and iso experiments
load('data/all_data_combined/all_separators.mat');

% compute confidence intervals
% non-iso experiment
stderror_identity_separator = std(all_separators)/sqrt(26);
ts = tinv([0.025  0.975],length(all_separators)-1); % T-Score
CI = mean(all_separators) + ts*stderror_identity_separator; % Confidence Intervals

%% compute confidecne intervals of identity line iso
stderror_identity_separator_iso = std(all_separators_iso)/sqrt(15);
ts_iso = tinv([0.025  0.975],length(all_separators_iso)-1); % T-Score
CI_iso = mean(all_separators_iso) + ts*stderror_identity_separator_iso; % Confidence Intervals


%% compute the facial similarity value of Eucliean distance that Kate got from her simulation
kate_simulation_eu_dist = 28.19;
kate_simulation_sim = sigmoid_bfs_distance(kate_simulation_eu_dist);

%% check whether Kate simultation number falls withing Euclidean distance of confidecne intervals of face identity judgements using the inverse of sigmoidal function iso
kate_simulation_eu_dist = 28.19;

mean_eu_dist_iso = sigmoid_bfs_distance_inv(mean(all_separators_iso));

CI_eu_dis_iso = sigmoid_bfs_distance_inv(CI_iso);

%% test whether identity separator in iso and non-iso experiments are significantly different from each other
load('data/all_data_combined/all_separators.mat');
mean_identity_separator = mean(all_separators);
identity_separator_mean = mean_identity_separator;
identity_separator_all_subj = all_separators;
mean_identity_separator_iso = mean(all_separators_iso);
identity_separator_mean_iso = mean_identity_separator_iso;
identity_separator_all_subj_iso = all_separators_iso;
[h,p,ci,stats] = ttest2(identity_separator_all_subj,identity_separator_all_subj_iso)


%% plot linear and sigmoid BFS distance on the data
close all;
figure('Name', 'Face similarity as a function of Basel Face Space');
% sigmoid_bfs_distance = sigmoid_distance_fit(all_sessions);
plot(all_eu_distances_iso, all_similarities_iso, 'ro','MarkerSize',2);
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

save_figure_kmj('analysis/linear_and_sigmoidal_model_fit_over_data_iso', figure(1));

%% plot linear and sigmoid BFS distance on the MEAN data
close all;
figure('Name', 'Face similarity as a function of Basel Face Space');
plot(mean_eu_distances_of_face_pair_iso, mean_similarity_of_face_pair_iso, 'r.','MarkerSize',15);
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

save_figure_kmj('analysis/linear_and_sigmoidal_model_fit_over_mean_data_iso', figure(1));

%% plot sigmoid BFS distance on the data with identity line

close all;
figure;
mean_identity_separator_iso_repeated = repelem(mean_identity_separator_iso, length(all_eu_distances_iso));
% line([0 80], [mean_identity_separator_iso mean_identity_separator_iso]); 
% hold on;
% line([0 80], [mean_identity_separator_iso+stderror_identity_separator_iso mean_identity_separator_iso+stderror_identity_separator_iso]);
% hold on;
% line([0 80], [mean_identity_separator_iso-stderror_identity_separator_iso mean_identity_separator_iso-stderror_identity_separator_iso]);
hold on;
sigmoid_bfs_distance = sigmoid_distance_fit(all_sessions_iso);
plot(all_eu_distances_iso, all_similarities_iso, 'ro','MarkerSize',2);
hold on;

dark_green_colour = [0/255, 176/255, 80/255];
plot([0:length(mean_identity_separator_iso_repeated)-1],mean_identity_separator_iso_repeated, 'color',dark_green_colour, 'LineWidth',4);
plot([0:length(mean_identity_separator_iso_repeated)-1],mean_identity_separator_iso_repeated+stderror_identity_separator_iso, 'color',dark_green_colour, 'LineWidth',1);
plot([0:length(mean_identity_separator_iso_repeated)-1],mean_identity_separator_iso_repeated-stderror_identity_separator_iso, 'color',dark_green_colour, 'LineWidth',1);

plot_1d(sigmoid_bfs_distance, '', 'k');

xlabel('BFM Euclidean distance');
ylabel('Facial dissimilarity judgements');

set(gca,'box', 'off');
xlim([-0.2 80]);
xticks(0:10:80)

ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20;

save_figure_kmj('analysis/identity_separator_with_sigmoidal_model_fit_over_data_iso', figure(1));

%% plot linear and sigmoid BFS distance on the MEAN data with identity line

close all;
figure;
mean_identity_separator_iso_repeated = repelem(mean_identity_separator_iso, length(all_eu_distances_iso))
% line([0 80], [mean_identity_separator_iso mean_identity_separator_iso]); 
% hold on;
% line([0 80], [mean_identity_separator_iso+stderror_identity_separator_iso mean_identity_separator_iso+stderror_identity_separator_iso]);
% hold on;
% line([0 80], [mean_identity_separator_iso-stderror_identity_separator_iso mean_identity_separator_iso-stderror_identity_separator_iso]);
hold on;
sigmoid_bfs_distance = sigmoid_distance_fit(all_sessions_iso);
plot_1d(linear_bfs_distance, '',[224/255,224/255,224/255]);
plot(mean_eu_distances_of_face_pair_iso, mean_similarity_of_face_pair_iso, '.','MarkerSize',15,'MarkerEdgeColor',[180/255,180/255,180/255]);
hold on;

dark_green_colour = [0/255, 176/255, 80/255];
plot([0:length(mean_identity_separator_iso_repeated)-1],mean_identity_separator_iso_repeated, 'color',dark_green_colour, 'LineWidth',4);
plot([0:length(mean_identity_separator_iso_repeated)-1],mean_identity_separator_iso_repeated+stderror_identity_separator_iso, 'color',dark_green_colour, 'LineWidth',1);
plot([0:length(mean_identity_separator_iso_repeated)-1],mean_identity_separator_iso_repeated-stderror_identity_separator_iso, 'color',dark_green_colour, 'LineWidth',1);

% plot_1d(sigmoid_bfs_distance, '', 'k');
plot_1d(sigmoid_bfs_distance, '', [120/255,120/255,120/255]);

xlabel('BFM Euclidean distance');
ylabel('Face dissimilarity judgements');

set(gca,'box', 'off');
xlim([-0.2 80]);
xticks(0:10:80)

ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20;

save_figure_kmj('analysis/identity_separator_with_linear_and_sigmoidal_model_fit_over_mean_data_iso', figure(1));

%% plot linear and sigmoid BFS distance on the MEAN data 

close all;
figure;

sigmoid_bfs_distance = sigmoid_distance_fit(all_sessions_iso);
plot_1d(linear_bfs_distance, '',[224/255,224/255,224/255]);
hold on;
plot(mean_eu_distances_of_face_pair_iso, mean_similarity_of_face_pair_iso,'.','MarkerSize',15, 'MarkerEdgeColor',[180/255,180/255,180/255]);
hold on;
% plot_1d(sigmoid_bfs_distance, '', 'k');
plot_1d(sigmoid_bfs_distance, '', [120/255,120/255,120/255]);

xlabel('BFM Euclidean distance');
ylabel('Face dissimilarity judgements');

set(gca,'box', 'off');
xlim([-0.2 80]);
xticks(0:10:80)

ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20;

save_figure_kmj('analysis/linear_and_sigmoidal_model_fit_over_mean_data_iso', figure(1));

%% plot sigmoid BFS distance with identity line with standard error 

close all;
figure;
mean_identity_separator_repeated_iso = repelem(mean_identity_separator_iso, length(all_eu_distances_iso))
% line([0 80], [mean_identity_separator mean_identity_separator]); 
% hold on;
% line([0 80], [mean_identity_separator+stderror_identity_separator mean_identity_separator+stderror_identity_separator]);
% hold on;
% line([0 80], [mean_identity_separator-stderror_identity_separator mean_identity_separator-stderror_identity_separator]);
hold on;
sigmoid_bfs_distance = sigmoid_distance_fit(all_sessions_iso);

dark_green_colour = [200/255,200/255,200/255];
plot([0:length(mean_identity_separator_repeated_iso)-1],mean_identity_separator_repeated_iso, 'color',dark_green_colour, 'LineWidth',4);
plot([0:length(mean_identity_separator_repeated_iso)-1],mean_identity_separator_repeated_iso+stderror_identity_separator_iso, 'color',dark_green_colour, 'LineWidth',1);
plot([0:length(mean_identity_separator_repeated_iso)-1],mean_identity_separator_repeated_iso-stderror_identity_separator_iso, 'color',dark_green_colour, 'LineWidth',1);

% plot_1d(sigmoid_bfs_distance, '', 'k');
plot_1d(sigmoid_bfs_distance, '', [120/255,120/255,120/255]);

xlabel('BFM Euclidean distance');
ylabel('Face dissimilarity judgements');

set(gca,'box', 'off');
xlim([-0.2 80]);
xticks(0:10:80)

ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20;

save_figure_kmj('analysis/identity_separator_with_sigmoidal_model_fit_iso', figure(1));



%% indeces of subjects that participated in iso exp
% manual intersect of subject names in:
all_subjects.name;
all_subjects_iso.name;
indeces_subj_in_so_exp = [1 3 4 6 8 11 13 14 18 19 22 23 24 25 26];

%% get similarities for face pairs
similarity_of_face_pair = nan(52,232);
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

%% testing correlations between sessions with stimulus set A and stimulus set B for subset of subjects that participated in both sessions

session_1_set_a = similarity_of_face_pair_session_one(indeces_subj_in_so_exp,:);

session_2_set_a = similarity_of_face_pair_session_two(indeces_subj_in_so_exp,:);

session_set_b = similarity_of_face_pair_iso;

session_1_set_a_all_subj = similarity_of_face_pair_session_one;

session_2_set_a_all_subj = similarity_of_face_pair_session_two;

corr_session_1_set_a_session_2_set_a_all_subj = corr2(session_1_set_a, session_2_set_a);

corr_session_1_set_a_session_2_set_a = corr2(session_1_set_a_all_subj, session_2_set_a_all_subj);

corr_session_1_set_a_session_set_b = corr2(session_1_set_a, session_set_b);

corr_session_2_set_a_session_set_b = corr2(session_2_set_a, session_set_b);

num_subj_set_a_set_b = 15;

save_mat_kmj('analysis/correlations/correlations_across_sessions', 'corr_session_1_set_a_session_2_set_a', 'corr_session_1_set_a_session_set_b', 'corr_session_2_set_a_session_set_b', 'corr_session_1_set_a_session_2_set_a_all_subj');

%% subjectwise_corr

%% subjectwise_corr_session_1_set_a_session_2_set_a_all_subj
subjectwise_corr_session_1_set_a_session_2_set_a_all_subj = [];
num_subj_set_a_set_b_all_subj = 26
for i = 1:num_subj_set_a_set_b_all_subj
    subjectwise_corr_session_1_set_a_session_2_set_a_all_subj(i) = corr(session_1_set_a_all_subj(i,:)', session_2_set_a_all_subj(i,:)');
end

std_subjectwise_corr_session_1_set_a_session_2_set_a_all_subj = std(subjectwise_corr_session_1_set_a_session_2_set_a_all_subj)/sqrt(num_subj_set_a_set_b_all_subj);
mean_subjectwise_corr_session_1_set_a_session_2_set_a_all_subj = mean(subjectwise_corr_session_1_set_a_session_2_set_a_all_subj);


%% subjectwise_corr_session_1_set_a_session_2_set_a
subjectwise_corr_session_1_set_a_session_2_set_a = [];
for i = 1:15
    subjectwise_corr_session_1_set_a_session_2_set_a(i) = corr(session_1_set_a(i,:)', session_2_set_a(i,:)');
end

std_subjectwise_corr_session_1_set_a_session_2_set_a = std(subjectwise_corr_session_1_set_a_session_2_set_a)/sqrt(num_subj_set_a_set_b);
mean_subjectwise_corr_session_1_set_a_session_2_set_a = mean(subjectwise_corr_session_1_set_a_session_2_set_a);

%% subjectwise_corr_session_1_set_a_session_set_b
subjectwise_corr_session_1_set_a_session_set_b = [];
for i = 1:15
    subjectwise_corr_session_1_set_a_session_set_b(i) = corr(session_1_set_a(i,:)', session_set_b(i,:)');
end

std_subjectwise_corr_session_1_set_a_session_set_b = std(subjectwise_corr_session_1_set_a_session_set_b)/sqrt(num_subj_set_a_set_b);
mean_subjectwise_corr_session_1_set_a_session_set_b = mean(subjectwise_corr_session_1_set_a_session_set_b);

%% subjectwise_corr_session_2_set_a_session_set_b
subjectwise_corr_session_2_set_a_session_set_b = [];
for i = 1:15
    subjectwise_corr_session_2_set_a_session_set_b(i) = corr(session_2_set_a(i,:)', session_set_b(i,:)');
end

std_subjectwise_corr_session_2_set_a_session_set_b = std(subjectwise_corr_session_2_set_a_session_set_b)/sqrt(num_subj_set_a_set_b);
mean_subjectwise_corr_session_2_set_a_session_set_b = mean(subjectwise_corr_session_2_set_a_session_set_b);

%% all mean correlations and stds

mean_correlations_between_sessions = [mean_subjectwise_corr_session_1_set_a_session_2_set_a, mean_subjectwise_corr_session_1_set_a_session_set_b, mean_subjectwise_corr_session_2_set_a_session_set_b];
std_correlations_between_sessions = [std_subjectwise_corr_session_1_set_a_session_2_set_a, std_subjectwise_corr_session_1_set_a_session_set_b, std_subjectwise_corr_session_2_set_a_session_set_b];

%% ttest to test whetehr there are significant differences between correlations

[H_session_1_session_3_ttest,p_session_1_session_3_ttest,~] = ttest(subjectwise_corr_session_1_set_a_session_2_set_a, subjectwise_corr_session_1_set_a_session_set_b, 'tail',  'right');

[H_session_2_session_3_ttest,p_session_1_session_3_ttest,~] = ttest(subjectwise_corr_session_1_set_a_session_2_set_a, subjectwise_corr_session_2_set_a_session_set_b, 'tail', 'right');

%% signrank to test whetehr there are significant differences between correlations

[p_H_session_1_session_3_signrank,H_session_1_session_3_signrank,~] = signrank(subjectwise_corr_session_1_set_a_session_2_set_a, subjectwise_corr_session_1_set_a_session_set_b, 'tail',  'right');

[p_session_2_session_3_signrank,H_session_2_session_3_signrank,~] = signrank(subjectwise_corr_session_1_set_a_session_2_set_a, subjectwise_corr_session_2_set_a_session_set_b, 'tail',  'right');

%% checking whetehr the correlations between stimulsu a sessions are higher for all subjects as compared to correlations with stimulsu set b session
session_1_stim_a_session_2_stim_b_difference = subjectwise_corr_session_1_set_a_session_2_set_a-subjectwise_corr_session_1_set_a_session_set_b;
session_1_stim_a_session_3_stim_b_difference = subjectwise_corr_session_1_set_a_session_2_set_a-subjectwise_corr_session_2_set_a_session_set_b;

%% plot results

close all;
figure;

bar(mean_correlations_between_sessions,0.5, 'FaceColor', [0.5 0.5 0.5]);
hold on;

e = errorbar(1:length(mean_correlations_between_sessions), mean_correlations_between_sessions, std_correlations_between_sessions, '.', 'CapSize',0, 'LineWidth', 2);
e.Color = 'black';
e.Marker = 'none';

set(gca,'FontName','Arial','TickLength',[0 0],'XTick',...
    1:length(mean_correlations_between_sessions),'XTickLabel',... 
    {'set A sess 1 sess 2', 'set A sess 1 set B', 'set A sess 2 set B'}, 'XTickLabelRotation',45); 

ylabel({'Judgements replicability'; 'Average across subjects'; 'Pearson correlation'}, 'FontName','Arial');

set(gca,'box', 'off');
% 
% ylim([0 0.06]);

ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20;

save_figure_kmj('analysis/correlations_between_sessions', figure(1));


%% analysis below only on iso data
%% plot iso data

%% functions fits iso data


%% combine no_iso and iso
all_subjects = struct;

for s = 1:length(all_subjects_iso)
    name_iso = all_subjects_iso(s).name;
    name_no_iso = strrep(name_iso, 'iso', '');
    index = getnameidx({all_subjects_no_iso.name}, name_no_iso);

    session1 = all_subjects_no_iso(index).sessions(1, :);
    session2 = all_subjects_no_iso(index).sessions(2, :);
    session3 = all_subjects_iso(s).sessions;

    all_subjects(s).sessions = [session1; session2; session3];
    all_subjects(s).sessions_combined = [session1, session2, session3];
    all_subjects(s).name = name_no_iso;
    
    [all_subjects(s).r1(1, :), all_subjects(s).r2(1, :), all_subjects(s).theta(1,:), all_subjects(s).eu_distances(1,:), all_subjects(s).similarities(1,:), all_subjects(s).separators(1,:)] = extract_results(session1);
    [all_subjects(s).r1(2, :), all_subjects(s).r2(2, :), all_subjects(s).theta(2,:), all_subjects(s).eu_distances(2,:), all_subjects(s).similarities(2,:), all_subjects(s).separators(2,:)] = extract_results(session2);
    [all_subjects(s).r1(3, :), all_subjects(s).r2(3, :), all_subjects(s).theta(3,:), all_subjects(s).eu_distances(3,:), all_subjects(s).similarities(3,:), all_subjects(s).separators(3,:)] = extract_results(session3);
end



%% (1) plot the similarity judgments
%% as a function of Euclidean distance
%% group
%% each subject

figure('Name', 'Face similarity as a function of Basel Face Space for each subject');

hold on;

for i = 1:length(all_subjects_no_iso)
    subject = all_subjects_no_iso(i);
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
scatter3(all_r1_no_iso,all_r2_no_iso,all_theta_no_iso, 50, all_similarities_no_iso,'.');
% plot other half of space (r1 and r2 have symmetric roles)
scatter3(all_r2_no_iso,all_r1_no_iso,all_theta_no_iso, 50, all_similarities_no_iso,'.');
xlabel('R1');
ylabel('R2');
zlabel('Theta');
title('Face similarity as a function of R1, R2 and theta')
colorbar;
view([-1 -1.4 1]);

%% each subject
figure('Name', 'Face similarity as a function of R1, R2 and theta for each subject');
for i = 1:length(all_subjects_no_iso)
    subject = all_subjects_no_iso(i);
    [r1, r2, theta, eu_d, sim, sep] = extract_results(subject.sessions_combined);
    subplot(6, 5, i);
    hold on;
    scatter3(all_r1_no_iso,all_r2_no_iso,all_theta_no_iso, 10, all_similarities_no_iso,'.');
    scatter3(all_r2_no_iso,all_r1_no_iso,all_theta_no_iso, 10, all_similarities_no_iso,'.');
    xlabel(subject.name);
    set(gca,'XTickLabel', []);
    set(gca,'YTickLabel', []);
    set(gca,'ZTickLabel', []);
    view([-1 -1.4 1]);
end
mtit(gcf,'Face similarity as a function of R1, R2 and theta')

%% for each theta bin, plot an R1-by-R2 matrix colour-coding perceptual distance (interpolated from the data)  
%% group
figure ('Name','R1-by-R2 matrix for each theta bin');
i = 0;
for theta = unique(all_theta_no_iso)
    i = i + 1;
    subplot(3, 3, i);
    plot_r1r2matrix(all_r1_no_iso, all_r2_no_iso, all_theta_no_iso, all_similarities_no_iso, theta);
    xlabel('R1');
    ylabel('R2');
    title(['theta = ',num2str(theta)]);
end
p = mtit(gcf,'R1-by-R2 matrix for each theta bin');
set(p.th,'Position',[0.5 1.03]);

%% each subject
figure ('Name','R1-by-R2 matrix for each theta bin for each subject');
hold on;
i = 0;
for s = 1:length(all_subjects_no_iso)
    subject = all_subjects_no_iso(s);
    [r1, r2, theta, eu_d, sim, sep] = extract_results(subject.sessions_combined);
    j = 0;
    for t = unique(theta)
        i = i + 1;
        j = j + 1;
        subplot(length(unique(theta)), length(all_subjects_no_iso), s + length(all_subjects_no_iso) * (j-1));
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



%% save all figures

for fig = findall(0,'Type','figure')'
   name = get(fig, 'Name');
   save_figure_kmj(['figures/Isotropicity/' name '.tiff'], fig)
   print(fig, '-dpsc2','-r300','-append','figures/Isotropicity/all');
 
end

%% extract dissimilarities for each r1,r2,theta combineation 
 
AllIsoDis = [all_r1_iso' all_r2_iso' all_theta_iso' all_eu_distances_iso' all_similarities_iso'];
[~, order] = sort(AllIsoDis(:, 4));
AllIsoDisSorted = AllIsoDis(order, :);
AllIsoAvg = averageBins(AllIsoDisSorted, 15);
AllIsoDisAvg= AllIsoAvg(:,5);
 
AllDis = [all_r1_no_iso' all_r2_no_iso' all_theta_no_iso' all_eu_no_iso' all_similarities_no_iso'];
[~, order] = sort(AllDis(:, 4));
AllDisSorted = AllDis(order, :);
AllAvg = averageBins(AllDisSorted, 52);
AllDisAvg = AllAvg(:,5);

%% Pearson and errors
pearson_error=zeros(length(all_subjects)+1,7);

for s = 1: length(all_subjects)    
    pearson_error(s,1:6) = calculate_pearson_error(all_subjects(s).similarities);
    
    A = all_subjects(s).similarities(1, :)';
    B = all_subjects(s).similarities(2, :)';
    C = all_subjects(s).similarities(3, :)';
    
    c = corr([A,B])-(corr([A,C])+corr([B,C]))/2;
    pearson_error(s,7) = c(1, 2);
end

all_similarities = [all_subjects.similarities];
pearson_error(length(all_subjects)+1,1:6) = calculate_pearson_error(all_similarities);

% write_as_tsv('results/pearson_error_each_subject.tsv', pearson_error, {'A-B','A-C', 'B-C','A-B','A-C', 'B-C', 'corr(A,B)-(corr(A,C)+corr(B,C))/2'}, {all_subjects.name, 'average'})

[p, h] = signrank(pearson_error(:, 7));

figure;
hold on;
colormap([0 0 0; 0.5 0.5 0.5; 0.7 0.7 0.7]);

subplot(2, 1, 1);
bar(pearson_error(:,1:3), 'grouped', 'BarWidth',1);
pbaspect([40, 10, 1]);
str = {all_subjects.name, 'all'};
set(gca, 'XTickLabel',str, 'XTick',1:numel(str));
xlim([0 17]);


subplot(2, 1, 2);
bar(pearson_error(:,4:6), 'grouped', 'BarWidth',1);
pbaspect([40, 10, 1]);
str = {all_subjects.name, 'all'};
set(gca, 'XTickLabel',str, 'XTick',1:numel(str));
xlim([0 17]);
ylim([0 1]);


% addHeadingAndPrint('Pearson correlations and standard error', 'figures/Pearson Correlation and standard error.ps');


%% find bins with the same value
 
%% plot bins with same value isotropic versus both sessions 
figure;
hold on;
plot(AllDisAvg, 'ro', 'markersize',5);
plot(AllIsoDisAvg,'bo', 'markersize',5);
x = 1:232;
plot([x; x], [AllDisAvg'; AllIsoDisAvg'], '-k');

%% prediction error

errorForIso(AllDisAvg, AllIsoDisAvg)

pearson(AllDisAvg, AllIsoDisAvg)


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

do_plot = 0;
if do_plot
    figure;
end
    
for s = 1:length(all_subjects_iso)
    subject = all_subjects(s);
    
%     % generalising across sessions (same stimuli, same subjects) 
%     for m = 1:length(models)
%         subplot(1,length(models), m);
%         model = models(m);
%         
%         half1 = subject.sessions(1,:);
%         half2 = subject.sessions(2,:);
%         
%         accuracy1 = prediction_accuracy(half1, half2, model, do_plot);
%         accuracy2 = prediction_accuracy(half2, half1, model);
%         across_sessions(s, m) = mean([accuracy1, accuracy2]);
%     end
    
    % generalising across stimuli (fit with a random half of the stimuli using both sessions, test with the other half of the stimuli using both sessions)
    for m = 1:length(models)
        model = models(m);
        
        n = length(subject.sessions);
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

% stderr_across_sessions_sigmoid = (std(across_sessions(:,1)))/number_of_subjects
% stderr_across_sessions_linear = (std(across_sessions(:,2)))/number_of_subjects

% stderr_across_stimuli_sigmoid = (std(across_stimuli(:,1)))/number_of_subjects
% stderr_across_stimuli_linear = (std(across_stimuli(:,2)))/number_of_subjects

stderr_across_sessions_sigmoid = (std(across_sessions(:,1)))/size(across_sessions, 1)
stderr_across_sessions_linear = (std(across_sessions(:,2)))/size(across_sessions, 1)

stderr_across_stimuli_sigmoid = (std(across_stimuli(:,1)))/size(across_sessions, 1)
stderr_across_stimuli_linear = (std(across_stimuli(:,2)))/size(across_sessions, 1)

mkdir_kmj('results/');

csvwrite('results/mean_across_sessions_sigmoid_iso.csv', mean_across_sessions_sigmoid);
csvwrite('results/mean_across_sessions_linear_iso.csv', mean_across_sessions_linear);

csvwrite('results/mean_across_stimuli_sigmoid_iso.csv', mean_across_stimuli_sigmoid);
csvwrite('results/mean_across_stimuli_linear_iso.csv', mean_across_stimuli_linear);

csvwrite('results/stderr_across_sessions_sigmoid_iso.csv', stderr_across_sessions_sigmoid);
csvwrite('results/stderr_across_sessions_linear_iso.csv', stderr_across_sessions_linear);

csvwrite('results/stderr_across_stimuli_sigmoid_iso.csv', stderr_across_stimuli_sigmoid);
csvwrite('results/stderr_across_stimuli_linear_iso.csv', stderr_across_stimuli_linear);

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
