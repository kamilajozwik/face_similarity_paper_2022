% Call functions to perform cross-validated reweighting, save the results,
% and display as a bar graph.

%% WARNING: takes a long time to run full reweighted analysis
% e.g. Main Figure 4B analysis with 16 models, 30 cross-validation folds, 
% and 2000 bootstraps, takes around 3.5hrs using parallel processing
% toolbox on a 2.9GHz 4-core, 8-thread CPU
%   -- To debug, reduce number of cross-val or bootstrap folds.
%   -- If parallel toolbox not available, change "parfor" to "for" in
%      FUNC_bootstrap_wrapper_vectors.m

clear all

nCV = 30; % can make smaller for debugging. At least 20 for full run.
nboots = 2000; % can make smaller for debugging. At least 1000 for full run.
savedir = './analysis/';

%% load human data

load('data/all_data_combined/all_similarities_iso.mat');
load('data/all_data_combined/all_pair_ids_iso.mat');

% get similarities for face pairs
for face_pair = 1:232
    indeces_of_face_pair = all_pair_ids_iso == face_pair;
    
    similarity_across_two_sessions = all_similarities_iso(indeces_of_face_pair);
    similarity_of_face_pair(:,face_pair) = similarity_across_two_sessions;
end

%% load model data
% cycle through model directory, loading predicted dissimilarities from 
% each component and adding them to an all-models cell array

modeldir = 'data/model_predictions_setB/'
modellist = dir([modeldir,'*.csv']);

for thismodel = 1:length(modellist)
    
    % load model-predicted distances
    rawModel = csvread(strcat(modeldir,modellist(thismodel).name));
    if iscolumn(rawModel)
        rawModel = rawModel';
    end
    
    % put these into a nested struct containing everything for this model
    model_struct.rawModel.dists = rawModel; % takes 1xN vector
    model_struct.rawModel.name = modellist(thismodel).name;
    
    component_models{thismodel} = model_struct; % add to whole-model cell array
end

%% set options and call reweighting functions

datafile = reshape(permute(similarity_of_face_pair,[2,1]), [1,232,15]); % reshape and pad for compatibility with rest of code
modelfile = component_models;
tic; WRAPPER_call_for_one_model_facial_similarity(datafile, modelfile, savedir, nboots, nCV); toc

%% load results and begin structuring for plotting

resultsdir = 'analysis/';

load(strcat(resultsdir,'bootstrap_output_layerwise.mat')); % loads as layerwise_results
load(strcat(resultsdir,'bootstrap_output_wholenet.mat')); % loads as wholenet_results
load(strcat(resultsdir,'bootstrap_output_layerwise_sigmoidal.mat')); % loads as layerwise_results
load(strcat(resultsdir,'bootstrap_output_wholenet_sigmoidal.mat')); % loads as wholenet_results
load(strcat(resultsdir,'bootstrap_output_ceilings.mat')); % loads as ceiling_results

lowceil = nanmean(ceiling_results.lower);
uppceil = nanmean(ceiling_results.upper);

%% sort models by performance of the raw or sigmoidal versions

% sort component models from best to worst
% [val, idx] = sort(nanmean(layerwise_results_sigmoidal.raw)); % based on sigmoid
[val, idx] = sort(nanmean(layerwise_results.raw)); % based on untransformed
idx = fliplr(idx);

layer_means = nanmean(layerwise_results.raw);
layer_means = layer_means(idx);
layer_sds = nanstd(layerwise_results.raw);
layer_sds = layer_sds(idx);

layer_means_sigmoidal = nanmean(layerwise_results_sigmoidal.raw);
layer_means_sigmoidal = layer_means_sigmoidal(idx);
layer_sds_sigmoidal = nanstd(layerwise_results_sigmoidal.raw);
layer_sds_sigmoidal = layer_sds_sigmoidal(idx);

groupedperfs = [layer_means, nanmean(wholenet_results.raw_fitted)];
groupedstds = [layer_sds, nanstd(wholenet_results.raw_fitted)];

groupedperfs_sigmoidal = [layer_means_sigmoidal, nanmean(wholenet_results_sigmoidal.raw_fitted)];
groupedstds_sigmoidal = [layer_sds_sigmoidal, nanstd(wholenet_results_sigmoidal.raw_fitted)];

%% PLOT

xlocs = [1:length(groupedperfs)-1,length(groupedperfs)+1];
% prettify names
xnames = {}
for m = 1:length(idx)
    tmp = layerwise_results.name{idx(m)};
    tmp = tmp(1:end-4); % cut off ordering number / file extension
    tmp(tmp=='_') = ' ';
    xnames{m} = tmp;
end
xnames{m+1} = 'weighted';

% add in a thresholding here, so that if sigmoidally-weighted performance
% is worse, it's just not shown:
sigmoidal_bar_heights = diag(groupedperfs_sigmoidal)-diag(groupedperfs);
sigmoidal_bar_heights(sigmoidal_bar_heights<0) = 0;
bars = bar(xlocs, [diag(groupedperfs); sigmoidal_bar_heights]', 'stacked', 'FaceColor', 'flat', 'BarWidth', 0.9)

hold on

% assigning colours to bars (to match layerwise plots)
% cm = parula(7);
for m = 1:(length(groupedperfs)-1)
    bars(m).FaceColor = [0.4, 0.4, 0.4];
    bars(length(groupedperfs)+m).FaceColor = [0.7, 0.7, 0.7];
end
bars(length(groupedperfs)).FaceColor = [0.4, 0.4, 0.4];
bars(length(groupedperfs)*2).FaceColor = [0.7, 0.7, 0.7];

% noise ceiling first in drawing order
patch([0 0 length(groupedperfs)+2 length(groupedperfs)+2],[lowceil, uppceil, uppceil, lowceil],[0.5, 0.5, 0.5],'edgecolor','none','FaceAlpha',0.5)
hold on

for i = 1:length(groupedperfs)
%     errorbar(xlocs(i), groupedperfs(:,i), groupedstds(:,i), 'color','k','LineWidth',2,'LineStyle','none','CapSize',0);
    errorbar(xlocs(i), groupedperfs_sigmoidal(:,i), groupedstds_sigmoidal(:,i), 'color','k','LineWidth',2,'LineStyle','none','CapSize',0);
end

% aesthetics
box off
xlim([0,length(groupedperfs)+1.5])
% ylim([0, 1])
yticks([0:0.2:1])
xticks(xlocs)
xticklabels(xnames);
xtickangle(45);
set(gcf,'color','w');
set(gcf,'Position',[100 100 900 650])
set(gca,'FontSize',18);
ylabel({'\fontsize{20}Model performance';'\fontsize{16}(Pearson corr. with human judgements)'});

%% statistical test indicators: does each model explain sig variance?
% ([un]comment to take into account how models have been sorted)

thresh = 0.05/length(groupedperfs); % bonferonni corrected

allresults = [layerwise_results.raw(:,idx), wholenet_results.raw_fitted];
% allresults = [layerwise_results_sigmoidal.raw(:,idx), wholenet_results_sigmoidal.raw_fitted];

% 1. each model vs lower bound of noise ceiling for trained and fitted nets
% (only ones close to noise ceiling)
diffs = repmat(ceiling_results.lower,[1, size(allresults,2)])-allresults;
for i = 1:size(diffs,2)
    ci = quantile(diffs(:,i), [thresh/2, 1-thresh/2]);
    if ci(1) > 0
%         text(xlocs(i)-0.1, lowceil+0.03, '*', 'FontSize',14);
    else
        text(xlocs(i)-0.15, lowceil+0.12, 'ns', 'FontSize',12);
        ci
    end
end

%% calculate pairwise comparisons: is each model sig better than each other?

% bonferonni corrected 
thresh = 0.05/nchoosek(length(groupedperfs),2);

% each model vs every other
pairwise_sigs = ones(length(groupedperfs),length(groupedperfs))*0.3;

for m = 1:length(groupedperfs)
    for n = 1:length(groupedperfs)
        if n > m % column > row
            diffs = allresults(:,m) - allresults(:,n);
            ci = quantile(diffs, [thresh/2, 1-thresh/2]); % two-tailed
            % check that CI does not contain 0
            if (max(ci) < 0) || (min(ci) > 0) % one model is better than the other
                pairwise_sigs(m,n) = 1;
            end
        else
            pairwise_sigs(m,n) = 0;
        end
    end
end

%% "Golan wings" to indicate significance on main bar plot
% https://github.com/rsagroup/pyrsa/blob/master/pyrsa/vis/model_plot.py

xmax = length(groupedperfs)+1.5; % should correspond to first cell
ymax = uppceil+0.7; % e.g. 0.3 for small sets; 0.5 for large sets; should correspond to first cell
vspacing = 0.032;

figure(1)
hold on
for mm = 1:length(groupedperfs)-1
    for mn = 1:length(groupedperfs)-1
        if pairwise_sigs(mm,mn) == 1
            plot([mm, mn], [ymax-mm*vspacing, ymax-mm*vspacing], '>-', 'markersize', 6, 'color', [0.5, 0.5, 0.5], 'markerfacecolor', [0.5, 0.5, 0.5]);
        end
    end
    % also do the reweighted model comparison, separately
    mn = length(groupedperfs);
    if pairwise_sigs(mm,mn) == 1
        plot([mm, mn+1], [ymax-mn*vspacing, ymax-mn*vspacing], '<-', 'markersize', 6, 'color', [0.5, 0.5, 0.5], 'markerfacecolor', [0.5, 0.5, 0.5]);
    end
end

set(gcf,'renderer','Painters')
% optionally, save figure in vector format
% print('-dpdf','myVectorFile3.pdf','-r300','-painters')
