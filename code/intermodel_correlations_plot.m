% Load model data and plot correlations between predictions of each model.

clear all

%% load model data
% cycle through model directory, loading predicted dissimilarities from 
% each component and adding them to an all-models cell array

modeldir = 'data/model_predictions_setA/'
modellist = dir([modeldir,'*.csv']);

allmodels = [];
allnames = {};

for thismodel = 1:length(modellist)
    
    % load model-predicted distances
    rawModel = csvread(strcat(modeldir,modellist(thismodel).name));
    if iscolumn(rawModel)
        rawModel = rawModel';
    end
    
    allmodels = [allmodels; rawModel];
    
    tmp = modellist(thismodel).name(1:end-4);
    tmp(tmp=='_') = ' ';
    allnames{thismodel} = tmp;
end

%% 

figure;
RSM = squareform(pdist(allmodels,'correlation'));
imagesc(RSM)
axis square
xticks(1:length(allnames));
xticklabels(allnames);
xtickangle(45);
yticks(1:length(allnames));
yticklabels(allnames);
set(gca,'FontSize',12);
set(gcf,'color','w');
title('Correlation distance')

set(gcf,'renderer','Painters')
% optionally, save figure in vector format
% print('-dpdf','myVectorFile3.pdf','-r300','-painters')
