function [layerwise_results, ceiling_results, wholenet_results] = WRAPPER_call_for_one_model_facial_similarity(datafile, modelfile, savedir, nboots, nCV)

% Entry point for crossvalidated model reweighting. See script
% "reweight_and_plot_setA.m" for an example call.

%% Wrapper for fitting multiple component models using bootstrapped cross-validated reweighting.
% katherine.storrs@gmail.com

% This version customised for the facial similarity project 
% (https://github.com/kamilajozwik/face_similarity_paper_2022), in which
% data are 1D (dissimilarities per unique stimulus pair). 
% This is a special case of the more general procedure for crossvalidated 
% reweighting when data are 2D (full pairwise dissimilarity matrices):
% https://github.com/tinyrobots/reweighted_model_comparison

% INPUT ARGUMENTS:
% Need to provide two sources of data:
%       datafile -- .mat file containing target data to which models will
%                   be fitted. Should be a single array of dimensions N x
%                   O, where N is the number of subjects and O is the
%                   number of data points (observations, e.g. pairs of
%                   stimuli for which dissimilarity was measured)
%       modelfile -- 1 x M cell array, for M models. Each entry containing the
%                   1D list of predicted dissimilarities for all stimuli 
%                   for that model.
%       savedir -- where to save results
%       nboots -- number of bootstrap samples to perform (must be equal to
%                   or less than the number of pre-assigned bootstrap
%                   samples in `samplingfile`
%       nCVs -- number of cross-validation folds to perform within each
%                   bootstrap sample. On each, a test set of subjects and
%                   stimuli is set aside, the model components are fitted
%                   on the training split, and evaluated on the remainder
%
% OUTPUT ARGUMENTS:
% This script will output and save the following structures:
%       layerwise_results -- cell array containing the performance of each
%                           component model, in each crossvalidated
%                           bootstrap fold (nomenclature assumes components
%                           are "layers" of a single model, eg. a neural
%                           network. Components may be any arbitrary 
%                           set of models though.)
%       layerwise_results_sigmoidal -- performance of each component if
%                          allowing a sigmoidal transform to be fitted
%                          within each cv fold, to map the model-predicted
%                          dissimilarities to the human data.
%       ceiling_results -- struct with 2 fields. Each field
%                          contain an (nboots x 1) vector of bootstrap 
%                          estimates of the performances of the ceilings:
%           .lower -- lower bound of noise ceiling
%           .upper -- upper bound of noise ceiling
%       wholenet_results -- estimates of the performance of a combined
%                          model in which all components have been combined
%                          using optimal non-negative linear reweighting.

try mkdir(savedir); end

%% set options...

% % options used by FUNC_compareRefRDM2candRDMs_reweighting
% highlevel_options.reweighting = true; % true = bootstrapped xval reweighting. Otherwise proceeds with standard RSA Toolbox analysis.
% highlevel_options.resultsPath = savedir;
% highlevel_options.barsOrderedByRDMCorr = false;
% highlevel_options.rootPath = pwd;

% options used by FUNC_bootstrap_wrapper
highlevel_options.boot_options.nboots = nboots; 
highlevel_options.boot_options.boot_conds = true;
highlevel_options.boot_options.boot_subjs = true; 

% options used by FUNC_reweighting_wrapper
highlevel_options.rw_options.nTestSubjects = 5; % 5 out of 26
highlevel_options.rw_options.nTestImages = 46; % 46 out of 232
highlevel_options.rw_options.nCVs = nCV; % number of crossvalidation loops within each bootstrap sample (stabilises estimate)

%% analyse
[layerwise_results, layerwise_results_sigmoidal, ceiling_results, wholenet_results, wholenet_results_sigmoidal] = FUNC_bootstrap_wrapper_vectors(datafile, modelfile, highlevel_options);

% add model information back into "layerwise" (per model) results
names = {};
for m = 1:length(modelfile)
    names{m} = modelfile{m}.rawModel.name;
end
layerwise_results.name = names;
layerwise_results_sigmoidal.name = names;

% save bootstrap distributions
save(strcat(savedir,'bootstrap_output_layerwise'), 'layerwise_results');
save(strcat(savedir,'bootstrap_output_layerwise_sigmoidal'), 'layerwise_results_sigmoidal');
save(strcat(savedir,'bootstrap_output_wholenet'), 'wholenet_results');
save(strcat(savedir,'bootstrap_output_wholenet_sigmoidal'), 'wholenet_results_sigmoidal');
save(strcat(savedir,'bootstrap_output_ceilings'), 'ceiling_results');

end
