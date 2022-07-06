# clone this repository from GitHub
```
go to https://github.com/kamilajozwik/face_similarity_paper_2022
```
```
git clone git@github.com:kamilajozwik/face_similarity_paper_2022.git
```
Or download and unzip this repository, naming its root directory `face_similarity_paper_2022`.

# download raw data from https://osf.io/7bh6s/
Human data for Stimulus Set A:
```
unzip "main" and put it into face_similarity_paper_2022/ExperimentalResults/main/
```
Human data for Stimulus Set B:
```
unzip "isotropicity" and put it into face_similarity_paper_2022/ExperimentalResults/isotropicity/
```
Model predictions for Stimulus Set A:
```
unzip "model_predictions_setA" and put it into face_similarity_paper_2022/data/model_predictions_setA
```
Model predictions for Stimulus Set B:
```
unzip "model_predictions_setB" and put it into face_similarity_paper_2022/data/model_predictions_setB
```

# run MATLAB scripts below
Navigate to the `face_similarity_paper_2022` folder in MATLAB, add this and its subfolders to the Matlab search path, and run commands below (tested in MATLAB_R2019b and MATLAB_R2021a but code should work in earlier versions as well).

# to format raw data for analysis
```
extract_data_similarity
extract_data_similarity_iso
```

# to generate figures in the main manuscript
For each of these, open the specified script found within `face_similarity_paper_2022/code/`, and run the cell with the specified title (`"%% ..."`), after first running any preliminary data-processing cells titled `"%% PRELIM: ..."`.

## Figure 2a
```
per_model_3D_barplots
%% BFM Euclidean distance (Figure 2A)
```
## Figure 2b, top
```
per_model_3D_barplots
%% Human perceived dissimilarity (Figure 2B, top)
```
## Figure 2b, bottom
```
per_model_3D_barplots_setB
%% Human perceived dissimilarity (Figure 2B, bottom)
```
## Figure 2c
```
MasterScriptFaceSimilarity
%% plot linear and sigmoid BFS distance on the MEAN data 

MasterScriptFaceSimilarityIsotropicity
%% plot linear and sigmoid BFS distance on the MEAN data 
```
## Figure 3a
```
MasterScriptFaceSimilarity
%% plot sigmoid BFS distance with identity line with standard error 
```
```
MasterScriptFaceSimilarityIsotropicity
%% plot sigmoid BFS distance with identity line with standard error 
```
## Figure 3b
```
MasterScriptFaceSimilarity
%% plot histogram for MEAN eu dist below and above identity line

MasterScriptFaceSimilarityIsotropicity
%% plot histogram for MEAN eu dist below and above identity line
```
## Figure 3c
```
MasterScriptFaceSimilarity
%% plot histogram for MEAN theta below and above identity line

MasterScriptFaceSimilarityIsotropicity
%% plot histogram for MEAN theta below and above identity line
```
## Figure 3d
```
MasterScriptFaceSimilarity
%% plot histogram for MEAN absolute difference between r1 and r2 below and above identity line

MasterScriptFaceSimilarityIsotropicity
%% plot histogram for MEAN absolute difference between r1 and r2 below and above identity line
```
## Figure 3e
```
MasterScriptFaceSimilarity
%% plot AUC

MasterScriptFaceSimilarityIsotropicity
%% plot AUC
```
## Figure 4b
```
reweight_and_plot_setA (top panel)
reweight_and_plot_setB (bottom panel)
```
## Figure 4c
```
unique_var_kate_models_all_model_weight_new_way (top panel)
unique_var_kate_models_all_model_weight_iso_new_way (bottom panel)
```

# to generate figures in the supplementary materials
## Supplementary Figure 1
```
MasterScriptFaceSimilarity
%% arrange face pairs in montage showing every 20 face pairs for similarity judgements using kate's stimuli

MasterScriptFaceSimilarityIsotropicity
%% arrange face pairs in montage showing every 20 face pairs for similarity judgements using kate's stimuli
```
## Supplementary Figure 2
```
MasterScriptFaceSimilarity
%% arrange face pairs in one big montage
```
## Supplementary Figure 3
```
MasterScriptFaceSimilarityIsotropicity
%% arrange face pairs in one big montage
```
## Supplementary Figure 4b
```
MasterScriptFaceSimilarity
%% Isotropicity/uniformity same stimuli test
```
## Supplementary Figures 6, 7, 8, 10b and 11
To create the top panel in each:
```
reweight_and_plot_setA
```
After changing the following variable to point to one of the appropriate datasets:
```
modeldir = 'data/model_predictions_setA/DNN_layers_VGG-face/' (Supp Fig 6, VGG-Face bars only)
modeldir = 'data/model_predictions_setA/DNN_layers_VGG-object/' (Supp Fig 6, VGG-Object bars only)
modeldir = 'data/model_predictions_setA/DNN_layers_VGG-BFM-identity/' (Supp Fig 7)
modeldir = 'data/model_predictions_setA/DNN_layers_VGG-BFM-latents/' (Supp Fig 8)
modeldir = 'data/model_predictions_setA/BFM_submodels/' (Supp Fig 10b)
modeldir = 'data/model_predictions_setA/BFM_num-PCs/' (Supp Fig 11)
```
To create the bottom panel in each:
```
reweight_and_plot_setB
```
After changing the following variable to point to one of the appropriate datasets:
```
modeldir = 'data/model_predictions_setB/DNN_layers_VGG-face/' (Supp Fig 6, VGG-Face bars only)
modeldir = 'data/model_predictions_setB/DNN_layers_VGG-object/' (Supp Fig 6, VGG-Object bars only)
modeldir = 'data/model_predictions_setB/DNN_layers_VGG-BFM-identity/' (Supp Fig 7)
modeldir = 'data/model_predictions_setB/DNN_layers_VGG-BFM-latents/' (Supp Fig 8)
modeldir = 'data/model_predictions_setB/BFM_submodels/' (Supp Fig 10b)
modeldir = 'data/model_predictions_setB/BFM_num-PCs/' (Supp Fig 11)
```
Note: Comment out final cell `%% "Golan wings" to indicate significance on main bar plot` to avoid crowding plot.
## Supplementary Figure 10c
```
unique_var_kate_models_all_model_weight_bfm_sub_new_way
unique_var_kate_models_all_model_weight_bfm_sub_iso_new_way
```
## Supplementary Figure 12
```
intermodel_correlations_plot
```
