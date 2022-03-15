function [] = plot_variance_result(add_var_expl, model_names)


[number_of_subjects, number_of_models] = size(add_var_expl);

mean_add_var_expl = mean(add_var_expl);

std_add_var_expl = std(add_var_expl);

stderror_add_var_expl = std(add_var_expl)/sqrt(number_of_subjects);

%% test for significance of unique variance, FDR correction not aplicable

% number_of_models = size(designMat,1);

significance_matrix_add_var = nan(number_of_models, 1);

for i = 1:number_of_models
    
    significance_matrix_add_var(i) = signrank(add_var_expl(:,i));
    % commented out below are just explorations of significance, all of
    % them wrong
% [p(i), h(i)] = signrank(add_var_expl(:,i));
%     [~, significance_matrix_add_var_fdr_corrected(i,:)]= fdr_bh(significance_matrix_add_var(i),0.05,'pdep');
%      [significance_matrix_add_var_fdr_corrected(i,:),significance_matrix_add_var_uncorrected_pvalues(i,:),significance_matrix_add_var_fdr_corrected_pvalues(i,:)] =  fdr_bh(significance_matrix_add_var(i),0.05,'pdep');

end


%% test for significant differences between models for unique variance FDR corrected

significance_matrix_unique_variance = nan(number_of_models, number_of_models);

for i = 1:number_of_models
    for y = 1:number_of_models
        significance_matrix_unique_variance(i,y) = signrank(add_var_expl(:,i),add_var_expl(:,y));
    end 
end

[significance_matrix_unique_variance_fdr_corrected]= fdr_bh(significance_matrix_unique_variance,0.05,'pdep');


%% plot results

max_y_position = max(mean_add_var_expl+stderror_add_var_expl);

[row, column] = find(triu(significance_matrix_unique_variance_fdr_corrected));

significance_matrix_unique_variance_fdr_corrected_indeces = {};
for ind = 1: length(row)
    significance_matrix_unique_variance_fdr_corrected_indeces{ind} = [row(ind),column(ind)];
end

% close all;
figure;

h = bar(mean_add_var_expl,0.85, 'FaceColor', [0.66 0.66 0.66]);
% bar(y,0.4)
h.EdgeColor = 'none';
hold on;

% % horizontal pairwise comparison bars
% sigstar(significance_matrix_unique_variance_fdr_corrected_indeces,max_y_position,[], 1);

hold on;

e = errorbar(1:number_of_models, mean_add_var_expl, stderror_add_var_expl, '.', 'CapSize',0, 'LineWidth', 0.75);
e.Color = 'black';
e.Marker = 'none';

significance_matrix_add_var_model_names = {};
for model_number = 1:number_of_models
    if significance_matrix_add_var(model_number)<0.05
        significance_matrix_add_var_model_names{model_number} = [model_names{model_number}, ' *'];
    else
        significance_matrix_add_var_model_names{model_number} = [model_names{model_number}, '  '];
    end
end

set(gca,'FontName','Arial','TickLength',[0 0],'XTick',...
    1:number_of_models,'XTickLabel',... 
    significance_matrix_add_var_model_names, 'XTickLabelRotation',45);

ylabel('Unique variance explained', 'FontName','Arial');

set(gca,'box', 'off');

% ylim([0 0.02]);

% ax = gca; ax.XAxis.FontSize = 19; ax.YAxis.FontSize = 19;
% ax = gca; ax.XAxis.FontSize = 16; ax.YAxis.FontSize = 16;
ax = gca; ax.XAxis.FontSize = 16; ax.YAxis.FontSize = 15;
% ax = gca; ax.XAxis.FontSize = 12; ax.YAxis.FontSize = 12;
ax.YAxis.Exponent = 0;

% %% test for significant differences between models for unique variance
% 
% significance_matrix_unique_variance = nan(number_of_models, number_of_models);
% 
% for i = 1:number_of_models
%     for y = 1:number_of_models
%         significance_matrix_unique_variance(i,y) = signrank(add_var_expl(:,i),add_var_expl(:,y));
%     end 
% end
% 
% [significance_matrix_unique_variance_fdr_corrected]= fdr_bh(significance_matrix_unique_variance,0.05,'pdep');
% 
% 
% %% plot significant differences between models for unique variance
% figure;
% imagesc(significance_matrix_unique_variance_fdr_corrected);
% c = gray;
% c = flipud(c);
% colormap(c);
% colorbar;
% caxis([0 1])
% 
% set(gca,'FontName','Arial','Fontsize', 16,'TickLength',[0 0],'XTick',1:number_of_models,'XTickLabel',model_names, 'XTickLabelRotation',45, 'xaxisLocation','top','YTick',1:number_of_models,'YTickLabel',model_names);
% 
% grid on;
% xticks([0.5:1:4.5])
% yticks([0.5:1:4.5])
% 
% print(gcf, '-dpsc2', ['animacy/analysis/unique_variance/unique_variance_animacy_dimensions_similarity_judgements_significance'], '-painters');
% print(gcf, '-dpng', ['animacy/analysis/unique_variance/unique_variance_animacy_dimensions_similarity_judgements_significance'], '-painters');
% 


end

