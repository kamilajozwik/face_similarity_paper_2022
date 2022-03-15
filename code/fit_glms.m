
function [semipartial_betas, add_var_expl, total_var_expl, var_expl] = fit_glms(designMat,dat)

%%INPUTS
% design matrix: [n_models x n_dissimiliarities] OR [n_models x n_dissimiliarities x n_timepoints]
% dat: [1 x n_dissimiliarities] OR [n_stimuli x n_stimuli] OR [n_stimuli x n_stimuli x n_timepoints]

%%OUTPUTS
% var_expl: the variance explained by each model. For model m, we fit GLM
% on X = "model m" and Y = data, the resulting R2 is the variance
% explained.
% add_var_expl: the unique variance explained by each model. For model m,
% we fit GLM on X = "all models but m" and Y = data, then we subtract the
% resulting R2 from the total R2 (fit GLM on X = "all models" and Y = data).

%% check inputs

error_message_dat = 'dat should be [1 x n_dissimiliarities] OR [n_stimuli x n_stimuli] OR [n_stimuli x n_stimuli x n_timepoints]';
if ndims(dat) == 2
    [a, b] = size(dat);
    if a == 1
        vector_data = dat';
        n_dissimilarities = b;
        n_timepoints = 1;
    else
        assert(a == b, error_message_dat);
        n_cond = a;
        n_timepoints = 1;
        vector_data = squareform(dat, 'tovector')';
        n_dissimilarities = length(vector_data);
    end
elseif ndims(dat) == 3
    [n_cond, n_cond_2, n_timepoints] = size(dat);
    assert(n_cond == n_cond_2, error_message_dat);
    n_dissimilarities = n_cond * (n_cond-1) / 2;
    vector_data = RDM2triu(dat);
else
    error(error_message_dat);
end

if ndims(designMat) == 2
    [n_models, n_dissimilarities_2] = size(designMat);
    assert(n_dissimilarities_2 == n_dissimilarities, ['designMat second dimension should have length ' num2str(n_dissimilarities) ' instead was: ' num2str(n_dissimilarities_2)]);
elseif ndims(designMat) == 3
    [n_models, n_dissimilarities_2, n_timepoints_2] = size(designMat);
    assert(n_dissimilarities_2 == n_dissimilarities, ['designMat second dimension should have length ' num2str(n_dissimilarities) ' instead was: ' num2str(n_dissimilarities_2)]);
    assert(n_timepoints_2 == n_timepoints, 'designMat third dimension should match dat third dimension');
else
    error('designMat should be [n_models x n_dissimiliarities] OR [n_models x n_dissimiliarities x n_timepoints]');
end

%%
% ntimes = size(dat,4); %kmj
% dat = squeeze(mean(dat, 3));%kmj

% %convert to upper triangle
% if ndims(dat) == 2 % we never reach this?
%     vector_data = squareform(dat, 'tovector')';
% else
%     vector_data = RDM2triu(dat);
% end
% subdata = RDM2triu_kmj(dat);

% designMat = RDM2triu(designMat); %kmj


%for each time point
betas_joint = nan(size(designMat,1),n_timepoints);
beta_l2 = nan(size(designMat,1),n_timepoints);
betas_joint_wconst = nan(size(designMat,1)+1,n_timepoints);
total_var_expl = nan(1,n_timepoints);
for t=1:n_timepoints
    
    if ndims(dat) == 3
        vector_data_t = vector_data(:,t);
    else
        vector_data_t = vector_data;
    end
   
    %For later reference: fit the full model to the data, estimate explained variance
    SStotal = (length(vector_data_t)-1) * var(vector_data_t);
    if ndims(designMat) == 3
        [tmp_beta, resnorm] = do_regression([ones(1,size(designMat,2));designMat(:,:,t)]',vector_data_t);
    else
        [tmp_beta, resnorm] = do_regression([ones(1,size(designMat,2));designMat]',vector_data_t);
    end
    SS_res_full = resnorm; % for later
    rsq_n = 1 - resnorm/SStotal; % r square formula
    total_var_expl(t) = rsq_n;
    
    betas_joint(:,t) = tmp_beta(2:end); %exclude constant  
    betas_joint_wconst(:,t) = tmp_beta; %including constant  

    
    %for each model, compute a GLM for the whole and P-1 sub-model
    for m=1:size(designMat,1)
    
        %n-1 model definition (adding a constant to model offset)
        if ndims(designMat) == 3
            dm_nm1 = designMat(setdiff(1:size(designMat,1),m),:, t);
        else
            dm_nm1 = designMat(setdiff(1:size(designMat,1),m),:);
        end
        dm_nm1 = [ones(1,size(dm_nm1,2));dm_nm1];
        
        %fit non-negative least squares to data
        [~, resnorm, residual] = do_regression(dm_nm1',vector_data_t);
        
        %% Residual analysis
        %level 2 GLM, fitting the residuals using the previously left-out model
        if ndims(designMat) == 3
            beta_l2(m,t) = do_regression(designMat(m,:,t)',residual);
        else
            beta_l2(m,t) = do_regression(designMat(m,:)',residual);
        end

        
        %% Additional variance explained
        %estimate n-1 model explained variance
        rsq_nm1 = 1 - resnorm/SStotal;
        add_var_expl(m,t) = rsq_n-rsq_nm1; % difference between r square (all models) and r square (all but model m)
        
        
        %% just the variance explained by the model (is this correct?)
        if ndims(designMat) == 3
            mat = designMat(m,:,t);
        else
            mat = designMat(m,:);
        end
        mat = [ones(1,size(mat,2)); mat]'; % add constant term
        [~, resnorm, residual] = do_regression(mat, vector_data_t);
        rsq_n1 = 1 - resnorm/SStotal;
        var_expl(m,t) = rsq_n1;

%         % based on https://en.wikipedia.org/wiki/Coefficient_of_determination#Coefficient_of_partial_determination
%         SS_res_reduced = resnorm;
% %         var_expl(m,t) = (SS_res_reduced - SS_res_full) / SS_res_reduced; % equivalent to:
%         var_expl(m,t) = 1 - SS_res_full / SS_res_reduced;
    end

end

semipartial_betas.beta_l2 = beta_l2;
semipartial_betas.beta_joint = betas_joint;
semipartial_betas.betas_joint_wconst = betas_joint_wconst;

end

function [beta, resnorm, residual] = do_regression(model, data)
    % Non-Negative Least Squares 
    [beta, resnorm, residual] = lsqnonneg(model, data);
    
%     % Ordinary Least Squares
%     [beta, bint, residual] = regress(data, model);
%     resnorm = norm(residual)^2;

end



