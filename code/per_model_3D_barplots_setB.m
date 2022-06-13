% Plot 3D barcharts of perceived dissimilarity or
% model-predicted dissimilarity for face pairs at each
% angle and radius-length

clear all

%% PRELIM: load human data

load('data/all_data_combined/all_similarities_iso.mat');
load('data/all_data_combined/all_pair_ids_iso.mat');

% get similarities for face pairs
for face_pair = 1:232
    indeces_of_face_pair = all_pair_ids_iso == face_pair;
    
    similarity_across_two_sessions = all_similarities_iso(indeces_of_face_pair);
    similarity_of_face_pair(:,face_pair) = similarity_across_two_sessions;
    mean_similarity_of_face_pair(face_pair) = mean(similarity_across_two_sessions);
    std_similarity_of_face_pair(face_pair) = std(similarity_across_two_sessions);
end

similarity_of_face_pair_session_one = similarity_of_face_pair((1:2:end),:);

% take mean over all subjects
avg_data = mean(similarity_of_face_pair_session_one);

%% PRELIM: load model data
modeldir = 'data/model_predictions_setB/';

% BFM geometry info
bfm_angle = csvread(strcat(modeldir,'BFM-angle.csv'));
r1 = csvread(strcat(modeldir,'allpcs_r1.csv'));
r2 = csvread(strcat(modeldir,'allpcs_r2.csv'));

% all other models
bfm_euc = csvread(strcat(modeldir,'BFM.csv'));
mesh = csvread(strcat(modeldir,'3D_mesh.csv'));
alexnet = csvread(strcat(modeldir,'Alexnet_best_layer.csv'));
config0 = csvread(strcat(modeldir,'Configural_0th.csv'));
config1 = csvread(strcat(modeldir,'Configural_1st.csv'));
config2 = csvread(strcat(modeldir,'Configural_2nd.csv'));
eigen = csvread(strcat(modeldir,'Eigenface.csv'));
gist = csvread(strcat(modeldir,'GIST.csv'));
hmax = csvread(strcat(modeldir,'HMAX_best_layer.csv'));
pixel = csvread(strcat(modeldir,'Pixel.csv'));
vgg_bfm = csvread(strcat(modeldir,'VGG-BFM-identity_best_layer.csv'));
vgg_bfm_latent = csvread(strcat(modeldir,'VGG-BFM-latents_best_layer.csv'));
vgg_face = csvread(strcat(modeldir,'VGG-Face_best_layer.csv'));
vgg_obj = csvread(strcat(modeldir,'VGG-Object_best_layer.csv'));
bfm_person = csvread(strcat(modeldir,'BFM-person-attributes.csv'));
aam = csvread(strcat(modeldir,'Active_Appearance_Model.csv'));

% collapse together very tiny geometric values to zero
bfm_angle(bfm_angle<0.001) = 0; 
r1(r1<0.001) = 0; 
r2(r2<0.001) = 0; 
bfm_euc(bfm_euc<0.001) = 0; 

%% BFM Euclidean distance (Figure 2A)

% specify here which model to plot and what to title it
model = bfm_euc;

plot_r1_r2_theta(model,bfm_angle,r1,r2)

%% Human perceived dissimilarity (Figure 2B, bottom)

% specify here which model to plot and what to title it
model = avg_data;

plot_r1_r2_theta(model,bfm_angle,r1,r2)
