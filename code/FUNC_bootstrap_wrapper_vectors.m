function [layerwise_results, layerwise_results_sigmoidal, ceiling_results, wholenet_results, wholenet_results_sigmoidal] = FUNC_bootstrap_wrapper_vectors(refData, component_models, highlevel_options)
% Bootstrap resamples conditions and/or subjects and passes the resampled
% data to another function to perform crossvalidated reweighting of the
% component model RDMs.
% -- can feed a "sampling_order" to, to use pre-specified boot samples

% CUSTOMISED FOR FACIAL SIMILARITY PROJECT

% create temporary storages accessible within parfoor loop
tmp_layerwise_raw = zeros([highlevel_options.boot_options.nboots,length(component_models)]);
tmp_layerwise_raw_sigmoidal = zeros([highlevel_options.boot_options.nboots,length(component_models)]);

tmp_wholenet_raw_fitted = zeros([highlevel_options.boot_options.nboots,1]);
tmp_wholenet_raw_fitted_sigmoidal = zeros([highlevel_options.boot_options.nboots,1]);

tmp_ceiling_lower = zeros([highlevel_options.boot_options.nboots,1]);
tmp_ceiling_upper = zeros([highlevel_options.boot_options.nboots,1]);

for boot = 1:highlevel_options.boot_options.nboots %parfor
%     fprintf(' %d ... ',boot)
    if mod(boot,5)==0
        disp(boot)
    end
    
    if highlevel_options.boot_options.boot_conds == true
        cond_ids = datasample(1:size(refData,2),size(refData,2),'Replace',true);
%         cond_ids = sampling_order(boot).cond_ids;
    else
        cond_ids = 1:size(refData,1);
    end
    
    if highlevel_options.boot_options.boot_subjs == true
        subj_ids = datasample(1:size(refData,3),size(refData,3),'Replace',true);
%         subj_ids = sampling_order(boot).subj_ids;
    else
        subj_ids = 1:size(refData,3);
    end
    
    % Calculates results for a single bootstrap sample:
    [layerwise_oneboot, layerwise_oneboot_sigmoidal, ceiling_oneboot] = FUNC_reweighting_wrapper_vectors(refData, component_models, highlevel_options.rw_options, cond_ids, subj_ids); % only needs sampling order info for this boot
    
    % nb "wholenet" (all models, reweighted) is contained within
    % "layerwise" results
    tmp_layerwise_raw(boot,:) = layerwise_oneboot.raw;
    tmp_layerwise_raw_sigmoidal(boot,:) = layerwise_oneboot_sigmoidal.raw;
    
    tmp_wholenet_raw_fitted(boot) = layerwise_oneboot.fitted;
    tmp_wholenet_raw_fitted_sigmoidal(boot) = layerwise_oneboot_sigmoidal.fitted;

    tmp_ceiling_lower(boot) = ceiling_oneboot.lower;
    tmp_ceiling_upper(boot) = ceiling_oneboot.upper;

end

% assign temporary values to structures
layerwise_results.raw = tmp_layerwise_raw;
layerwise_results_sigmoidal.raw = tmp_layerwise_raw_sigmoidal;

wholenet_results.raw_fitted = tmp_wholenet_raw_fitted;
wholenet_results_sigmoidal.raw_fitted = tmp_wholenet_raw_fitted_sigmoidal;

ceiling_results.lower = tmp_ceiling_lower;
ceiling_results.upper = tmp_ceiling_upper;
end