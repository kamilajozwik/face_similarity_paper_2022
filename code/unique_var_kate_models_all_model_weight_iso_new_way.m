%% load models 
model_names_for_read_csv = {'VGG-Face_best_layer', 'AAM', 'VGG-BFM-identity_best_layer','VGG-BFM-latents_best_layer', 'Alexnet_best_layer', 'VGG-Object_best_layer', 'GIST',  'BFM', 'HMAX_best_layer', 'eigenface', 'Pixel', 'Configural_0th', '3D_mesh', 'Configural_1st', 'BFM-angle', 'Configural_2nd'};
% fot plotting
model_names = {'VGG-Face best layer', 'Active Appearance Model', 'VGG-BFM-identity best layer','VGG-BFM-latents best layer', 'Alexnet best layer', 'VGG-Object best layer', 'GIST', 'BFM','HMAX best layer', 'Eigenface',  'Pixel',    'Configural 0th',  '3D mesh', 'Configural 1st', 'BFM-angle', 'Configural 2nd'}; 


% % older plotting
% model_names_for_read_csv = {'01_VGG-Face_best','17_VGG-BFM-id-224_best','16_VGG-BFM-id-128_best', '18_VGG-BFM-128-cropped_best', '02_Alexnet_best', '03_GIST', '04_VGG-Object_best', '05_BFM', '06_BFM-shape', '07_BFM-texture', '08_HMAX_best', '09_pixel', '10_config_0th', '11_mesh', '12_config_1st', '13_config_2nd',  '14_BFM-angle', '15_person_attributes'};
% % fot plotting
% model_names = {'VGG-Face best', 'VGG-BFM identity best 224','VGG-BFM identity best 128', 'VGG-BFM latent best 128','Alexnet best','GIST', 'VGG-Object best', 'BFM','BFM-shape', 'BFM-texture','HMAX best',  'pixel',    'config 0th',  'mesh', 'config 1st', 'config 2nd', 'BFM-angle','person attributes'}; 


number_of_models = size(model_names,2);

model = [];

models_dissimilarities_raw = nan(232,number_of_models);

for i = 1: length(model_names)
    
    model = csvread(['data/model_predictions_revision/isotropicity_expt/' model_names_for_read_csv{i} '.csv']);
    
    % added normalisation by subtracting mean and dividing by std as model
    % dissilimilarity values were in very different ranges
    models_dissimilarities(:,i) = (model - mean(model)) / std(model);
            
end

%% partial correlations analysis for GIST, Basel face space model and VGG trained on faces and objects 

load('analysis/similarity_of_face_pair_iso.mat', 'similarity_of_face_pair_iso');

designMat = models_dissimilarities';

number_of_subjects = size(similarity_of_face_pair_iso,1);

clear semipartial_betas add_var_expl total_var_expl var_expl
for subj = 1:number_of_subjects
    subj
    dat = similarity_of_face_pair_iso(subj,:);
    [semipartial_betas(subj,:), add_var_expl(subj,:), total_var_expl(subj,:), var_expl(subj, :)] = fit_glms(designMat,dat);
   
end

% %% partial correlations analysis for GIST, Basel face space model and VGG trained on faces and objects 
% 
% load('analysis/concatenated_two_sessions_subj_repeated.mat', 'concatenated_two_sessions_subj_repeated');
% 
% similarity_of_face_pair_two_sessions = mean(concatenated_two_sessions_subj_repeated,3);
% 
% designMat = models_dissimilarities';
% 
% number_of_subjects = size(similarity_of_face_pair_two_sessions,1);
% 
% for subj = 1:number_of_subjects
%     
%     [semipartial_betas(subj,:), add_var_expl(subj,:), total_var_expl(subj,:)] = fit_glms_no_time_dim(designMat,similarity_of_face_pair_two_sessions(subj,:));
%     
% end
% 
% mean_add_var_expl = mean(add_var_expl);
% 
% std_add_var_expl = std(add_var_expl);

%%
output_folder = 'analysis/variance/all_model_weights_iso/';
plot_variance_results(total_var_expl, var_expl, add_var_expl, model_names, output_folder);
