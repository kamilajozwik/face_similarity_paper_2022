function [layerwise_oneboot, layerwise_oneboot_sigmoidal, ceiling_oneboot] = FUNC_reweighting_wrapper_vectors(refData, component_models, rw_options, cond_ids, subj_ids)

% Performs a single bootstrap sample in which multiple crossvalidation
% folds are performed. On each crossval fold, data are split into training 
% and test portions, models are fitted to training portion, and
% tested on test portion. Intended to be embedded within a bootstrap loop,
% which supplies the indices of the selected subjects and conditions on
% this bootstrap

% nb. naming and comments assume that the multiple model components are
% different layers within a neural network - but they could be different
% feature maps within a layer, or anything else.

% CUSTOMISED FOR FACIAL SIMILARITY PROJECT
lsqoptions = optimset('Display','none'); % level of info from fitting procedures

%% extract info from data

nConds = size(refData,1);
nSubjs = size(refData,3);

%% Cross-validation procedure

% create temporary storages for per-crossval fold results for each estimate
loop_store_layerwise.raw = zeros([rw_options.nCVs,length(component_models)]); % one column per layer x nCV rows
loop_store_layerwise_sigmoidal.raw = zeros([rw_options.nCVs,length(component_models)]); % one column per layer x nCV rows

loop_store_ceilings.lower =  zeros([rw_options.nCVs,1]);
loop_store_ceilings.upper =  zeros([rw_options.nCVs,1]);

% cycle through crossvalidation procedure, which splits data into both separate
% stimuli and subject groups. This xval loop has the purpose of stabilising 
% the estimates obtained within each bootstrap sample
for loop = 1:rw_options.nCVs
    
    %% 1. Preparation: split human data into training and test partitions #####################
    
    % STIMULI: We select exactly nTestImages that *are present* in this
    % bootstrap sample (and then sample multiply according to how many
    % times they are present in the sample)
    cond_ids_test = datasample(unique(cond_ids), rw_options.nTestImages, 'Replace', false);
    cond_ids_train = setdiff(unique(cond_ids),cond_ids_test); % use the others for training
   
    % find locations of where these are present in the bootstrapped sample,
    % and append to two lists of cond_id entries we're going to use for
    % training and testing. Note that these change size from boot to boot:
    cond_locs_test = [];
    for i = 1:length(cond_ids_test)
        cond_locs_test = [cond_locs_test, find(cond_ids==cond_ids_test(i))];
    end
    cond_locs_train = [];
    for i = 1:length(cond_ids_train)
        cond_locs_train = [cond_locs_train, find(cond_ids==cond_ids_train(i))];
    end

    % SUBJECTS: apply same logic here, only selecting *available* subjects
    subj_ids_test = datasample(unique(subj_ids), rw_options.nTestSubjects, 'Replace', false);
    subj_ids_train = setdiff(unique(subj_ids),subj_ids_test); % use the others for training

    % find locations of any of these present in the (possibly) bootstrapped sample,
    % and append to two lists of subj_id entries we're going to use for
    % training and testing:
    subj_locs_test = [];
    for i = 1:length(subj_ids_test)
        subj_locs_test = [subj_locs_test, find(subj_ids==subj_ids_test(i))];
    end
    subj_locs_train = [];
    for i = 1:length(subj_ids_train)
        subj_locs_train = [subj_locs_train, find(subj_ids==subj_ids_train(i))];
    end
    
    % training data
    c_sel_train = cond_ids(cond_locs_train);
    s_sel_train = subj_ids(subj_locs_train);
    data_train = refData(:,c_sel_train,s_sel_train);
    data_train = mean(data_train,3); % average over subjects

    % test data
    c_sel_test = cond_ids(cond_locs_test);
    s_sel_test = subj_ids(subj_locs_test);
    data_test = refData(:,c_sel_test,s_sel_test);

    % also create an RDM of ALL subjects' data for test images,
    % for calculating the UPPER bound of the noise ceiling
    data_test_all_subjs = refData(:,c_sel_test,subj_ids); % nb. if bootstrapping Ss, this can contain duplicates
    data_test_all_subjs = mean(data_test_all_subjs,3); % we only ever need the mean
    
    % ...plus an RDM of TRAINING subjects' data for TEST images,
    % for calculating the LOWER bound of the noise ceiling
    data_test_train_subjs = refData(:,c_sel_test,s_sel_train); % nb. if bootstrapping Ss, this can contain duplicates - but cannot overlap w training data
    data_test_train_subjs = mean(data_test_train_subjs,3); % we only ever need the mean
    
    %% Begin layerwise calculations:
    
    for thismodel = 1 % this version hardcoded to only have one model
        
        %% 3. calculate performance of raw and reweighted components #####################
        % for each layer, gather its component RDMs, fit weights, and store a
        % reweighted predicted RDM for the test stimuli
        clear models_train models_test
        clear models_train_sigmoidal models_test_sigmoidal
        for component = 1:length(component_models)
            cModel_train = component_models{component}.rawModel.dists;
            cModel_train = cModel_train./max(cModel_train); % normalise to make sigmoid-fitting easier
            cModel_train = cModel_train(:,c_sel_train); % resample rows and columns
            models_train(:,component) = cModel_train;
            
            cModel_test = component_models{component}.rawModel.dists;
            cModel_test = cModel_test./max(cModel_test); % normalise to make sigmoid-fitting easier
            cModel_test = cModel_test(:,c_sel_test); % resample rows and columns
            models_test(:,component) = cModel_test;
            
            % calculate performance of raw model
            clear tm % very temp storage
            for test_subj = 1:size(data_test,3)
                tm(test_subj) = corr(cModel_test', data_test(:,:,test_subj)', 'Type', 'Pearson');
%                   tm(test_subj) = rankCorr_Kendall_taua(cModel_test', data_test(:,:,test_subj)');   
            end
            % save means of the correlations over held-out subjects
            loop_store_layerwise.raw(loop, component) = mean(tm); 
            
            %% 4. sigmoidal version of each model: 
            % fit a sigmoid to raw model predictions, and evaluate
            % transformed predictions on test data
           
            % fit a logistic
            fo = fitoptions('Method','NonlinearLeastSquares',...
                       'StartPoint',[0.8, 1, -10]);
            params = fit(cModel_train',data_train','a./(1+exp(b+c*x))', fo);
            
            % apply these sigmoidal parameters to get model predictions for
            % test stimuli, and evaluate:
            cModel_train_sigmoidal = params.a./(1+exp(params.b+params.c*cModel_train)); % also make a train version, for fitting per-model weights
            models_train_sigmoidal(:,component) = cModel_train_sigmoidal;
            cModel_test_sigmoidal = params.a./(1+exp(params.b+params.c*cModel_test));
            models_test_sigmoidal(:,component) = cModel_test_sigmoidal;

            clear tm % very temp storage
            for test_subj = 1:size(data_test,3)
                tm(test_subj) = corr(cModel_test_sigmoidal', data_test(:,:,test_subj)', 'Type', 'Pearson');
            end
            loop_store_layerwise_sigmoidal.raw(loop, component) = mean(tm);
        end
        
        % ------------------
        %% 5. do regression to estimate per-model weights

        % main call to fitting library - this could be replaced with
        % glmnet for ridge regression, etc., but GLMnet is not compiled
        % to work in Matlab post ~2014ish.
        weights = lsqnonneg(double(models_train), double(data_train'), lsqoptions);
        weights_sigmoidal = lsqnonneg(double(models_train_sigmoidal), double(data_train'), lsqoptions);
        
        % combine each layer in proportion to the estimated weights
%         model_train_weighted = models_train*weights;
        model_test_weighted = models_test*weights;
        model_test_weighted_sigmoidal = models_test_sigmoidal*weights_sigmoidal;
%         
%         perlayer_fitted_components_train = [perlayer_fitted_components_train, model_train_weighted]; 
%         perlayer_fitted_components_test = [perlayer_fitted_components_test, model_test_weighted]; 
    
        % calculate performance on the held out subjects and images

        % Now we have added the reweighted version to our list of models, 
        % evaluate each one, along with the noise ceiling
        % - need to do this individually against each of the test Ss
        % Note that if bootstrapping Ss, the test subjects may not be
        % unique, but may contain duplicates. This equates to weighting
        % each subject's data according to how frequently it occurs in this
        % bootstrapped sample.
        clear tm % very temp storage
        for test_subj = 1:size(data_test,3)
            tm(test_subj) = corr(model_test_weighted, data_test(:,:,test_subj)', 'Type', 'Pearson');
%             tm(test_subj) = rankCorr_Kendall_taua(model_test_weighted, data_test(:,:,test_subj)');
        end
        % get means of the correlations over held-out subjects
        loop_store_layerwise.fitted(loop, thismodel) = mean(tm); 
        
        % Same for sigmoidal versions of each model:
        clear tm % very temp storage
        for test_subj = 1:size(data_test,3)
            tm(test_subj) = corr(model_test_weighted_sigmoidal, data_test(:,:,test_subj)', 'Type', 'Pearson');
%             tm(test_subj) = rankCorr_Kendall_taua(model_test_weighted, data_test(:,:,test_subj)');
        end
        loop_store_layerwise_sigmoidal.fitted(loop, thismodel) = mean(tm);
    end
    
    %% 6. Estimate noise ceilings just once #####################
    clear tcl tcu % very temp storages
    for test_subj = 1:size(data_test,3)
        % Model for lower bound = correlation between each subject and mean  
        % of test data from TRAINING subjects (this captures a "perfectly fitted" 
        % model, which has not been allowed to peek at any of the training Ss' data)
        tcl(test_subj) = corr(data_test_train_subjs', data_test(:,:,test_subj)', 'Type', 'Pearson');
%         tcl(test_subj) = rankCorr_Kendall_taua(data_test_train_subjs', data_test(:,:,test_subj)');
        % Model for upper noise ceiling = correlation between each subject 
        % and mean of ALL train and test subjects' data, including themselves (overfitted)
        tcu(test_subj) = corr(data_test_all_subjs', data_test(:,:,test_subj)', 'Type', 'Pearson');
%         tcu(test_subj) = rankCorr_Kendall_taua(data_test_all_subjs', data_test(:,:,test_subj)');
    end
    loop_store_ceilings.lower(loop) = mean(tcl);
    loop_store_ceilings.upper(loop) = mean(tcu);

end % end of crossvalidation loops

% average over crossvalidation loops at the end of this bootstrap sample
layerwise_oneboot.fitted = mean(loop_store_layerwise.fitted);
layerwise_oneboot.raw = mean(loop_store_layerwise.raw);

layerwise_oneboot_sigmoidal.fitted = mean(loop_store_layerwise_sigmoidal.fitted);
layerwise_oneboot_sigmoidal.raw = mean(loop_store_layerwise_sigmoidal.raw);

ceiling_oneboot.lower = mean(loop_store_ceilings.lower); % will be split into multiple substructures
ceiling_oneboot.upper = mean(loop_store_ceilings.upper); % will be split into multiple substructures

end
