%% load models GIST, BFM, VGG-face, VGG-object
model_names_for_read_csv = {'BFM-shape', 'BFM', 'BFM-texture', 'BFM-angle', 'BFM-texture-angle', 'BFM-shape-angle', 'BFM-person-attributes', 'BFM-person-attributes-angle'};% fot plotting
model_names = {'BFM shape', 'BFM', 'BFM texture', 'BFM angle', 'BFM texture angle', 'BFM shape angle', 'BFM person attributes', 'BFM person attributes angle'};% fot plotting

number_of_models = size(model_names_for_read_csv,2);

model = [];

models_dissimilarities_raw = nan(232,number_of_models);

for i = 1: length(model_names_for_read_csv)
    
    model = csvread(['data/model_predictions_revision/main_expt/BFM_submodels/' model_names_for_read_csv{i} '.csv']);
    
    % added normalisation by subtracting mean and dividing by std as model
    % dissilimilarity values were in very different ranges
    models_dissimilarities(:,i) = (model - mean(model)) / std(model);
            
end

%% partial correlations analysis for GIST, Basel face space model and VGG trained on faces and objects 

load('analysis/concatenated_two_sessions_subj_repeated.mat', 'concatenated_two_sessions_subj_repeated');

similarity_of_face_pair_two_sessions = mean(concatenated_two_sessions_subj_repeated,3);

designMat = models_dissimilarities';

number_of_subjects = size(similarity_of_face_pair_two_sessions,1);

clear semipartial_betas add_var_expl total_var_expl var_expl
for subj = 1:number_of_subjects
    subj
%     [semipartial_betas(subj,:), add_var_expl(subj,:), total_var_expl(subj,:)] = fit_glms_no_time_dim(designMat,similarity_of_face_pair_two_sessions(subj,:));
    dat = similarity_of_face_pair_two_sessions(subj,:);
    [semipartial_betas(subj,:), add_var_expl(subj,:), total_var_expl(subj,:), var_expl(subj, :)] = fit_glms(designMat,dat);
   
end

%%
output_folder = 'analysis/variance/all_model_weights/BFM_submodels/';
% plot_variance_results(add_var_expl, model_names, output_folder);
plot_variance_results(total_var_expl, var_expl, add_var_expl, model_names, output_folder);
