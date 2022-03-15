function [dprime,AUC] = dprime_and_roc(X,Y, do_plot)
% X: data   (232 x 1)
% Y: labels e.g. [1; 1; 2; 1; 2; 2...] (232 x 1)
% do_plot: boolean, whether to plot the ROC curve (default: 0)

if ~exist('do_plot') do_plot = 0; end

model_params = mnrfit(X,Y,'model','nominal');
probabilities = mnrval(model_params,X,'model','nominal');
predicted_label = [];
for i = 1:length(probabilities)
    if probabilities(i,1) > 0.5
        predicted_label(i) = 1;
    else
        predicted_label(i) = 2;
    end
end

correct_label_1 = sum(predicted_label(Y==1) == 1);
total_label_1 = sum(Y==1);
label_2_predicted_as_1 = sum(predicted_label(Y==2) == 1);
total_label_2 = sum(Y==2);

% calulate dprime
hit_rate = correct_label_1 / total_label_1;
false_alarm_rate = label_2_predicted_as_1 / total_label_2;
dprime = norminv(hit_rate) - norminv(false_alarm_rate);


% ROC

% % 1. using roc / plotroc
% targets = [Y==1, Y==2];
% [tpr,fpr,thresholds] = roc(targets', probabilities')
% figure; plotroc(targets', probabilities');

% 2. using perfcurve (also gives AUC = Area Under Curve)
[x,y,T,AUC] = perfcurve(Y',probabilities(:,1),1);
% [x,y,T,AUC] = perfcurve(Y',probabilities(:,2),2); % for the other label (gives different curve but same AUC)

if do_plot
    figure;
    plot(x,y);
    xlabel('False positive rate');
    ylabel('True positive rate');
    title('ROC for Classification by Logistic Regression');
end


end

