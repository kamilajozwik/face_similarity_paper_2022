# clone this repository from GitHub
```
go to https://github.com/kamilajozwik/face_similarity_paper_2022
```
```
git clone git@github.com:kamilajozwik/face_similarity_paper_2022.git
```
Or download and unzip this repository, naming its root directory `face_similarity_paper_2022`.

# download raw data from https://osf.io/7bh6s/
```
unzip "main" and put it into face_similarity_paper_2022/ExperimentalResults/main
```
```
unzip "isotropicity" and put it into face_similarity_paper_2022/ExperimentalResults/isotropicity/
```

# run MATLAB scripts below
Navigate to the `face_similarity_paper_2022` folder in MATLAB, add this and its subfolders to the Matlab search path, and run commands below (tested in MATLAB_R2021a but code should work in earlier versions as well).

# to format raw data for analysis
```
extract_data_similarity
extract_data_similarity_iso
```

# to generate figures in the main manuscript
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
## Figure 4c
```
unique_var_kate_models_all_model_weight_new_way
unique_var_kate_models_all_model_weight_iso_new_way
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
## Supplementary Figure 10c
```
unique_var_kate_models_all_model_weight_bfm_sub_new_way
unique_var_kate_models_all_model_weight_bfm_sub_new_way_iso
```
