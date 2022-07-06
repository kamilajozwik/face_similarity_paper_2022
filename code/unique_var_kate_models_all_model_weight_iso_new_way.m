%% load models 
model_names_for_read_csv = {'VGG-Face_best_layer', 'Active_Appearance_Model', 'VGG-BFM-identity_best_layer','VGG-BFM-latents_best_layer', 'Alexnet_best_layer', 'VGG-Object_best_layer', 'GIST',  'BFM', 'HMAX_best_layer', 'Eigenface', 'Pixel', 'Configural_0th', '3D_mesh', 'Configural_1st', 'BFM-angle', 'Configural_2nd'};
% fot plotting
model_names = {'VGG-Face best layer', 'Active Appearance Model', 'VGG-BFM-identity best layer','VGG-BFM-latents best layer', 'Alexnet best layer', 'VGG-Object best layer', 'GIST', 'BFM','HMAX best layer', 'Eigenface',  'Pixel',    'Configural 0th',  '3D mesh', 'Configural 1st', 'BFM-angle', 'Configural 2nd'}; 

number_of_models = size(model_names,2);

model = [];

models_dissimilarities_raw = nan(232,number_of_models);

for i = 1: length(model_names)
    
    model = csvread(['data/model_predictions_setB/' model_names_for_read_csv{i} '.csv']);
    
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

%%
output_folder = 'analysis/variance/all_model_weights_iso/';
plot_variance_results(total_var_expl, var_expl, add_var_expl, model_names, output_folder);
