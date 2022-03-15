function [ accuracy ] = prediction_accuracy( train_data, test_data, model, do_plot)

    if ~exist('do_plot'); do_plot = 0; end

    [~, ~, ~, ~, sim, ~] = extract_results(test_data);
    
    f = model.fit(train_data);
    predicted = f(model.extract(test_data));
        
        
    R_2 = @(y, f) 1 - sum((y-f).^2) / sum((y - mean(y)).^2);
%      accuracy = R_2(sim', predicted);
%     accuracy = pearson(sim, predicted);
     accuracy = corr(sim', predicted, 'type', 'Pearson');
    
    if do_plot
        [~, ~, ~, eu_train, sim_train, ~] = extract_results(train_data);
        [~, ~, ~, eu_test, sim_test, ~] = extract_results(test_data);
        hold on; plot(eu_train, sim_train, '*g'); plot(eu_test, predicted, '.b');
        title(accuracy);
    end
end

