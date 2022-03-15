function [ f, f_inv ] = sigmoid_distance_fit( data )
% returns funciotn and inverses, the inverse allows getting the Euclidean
% distance given the similarity measure e.g.:
% [sigmoid_bfs_distance, sigmoid_bfs_distance_inv] = sigmoid_distance_fit(all_sessions_iso);
% eu_dist_fits = sigmoid_bfs_distance_inv(similarity)
    [~, ~, ~, eu_d, sim, ~] = extract_results(data);
    
    general_sigmoid = @(param, x) param(1)+(param(2)-param(1)) ./ (1+10.^((param(3)-x) .* param(4)));
    
    % Seeting good initial_params.
    % These numbers come from one of the fits.
    % We do this to avoid the few cases where nlinfit doesn't manage to fit.
    % In those cases we would see the following warning:
    %     Warning: Some columns of the Jacobian are effectively zero at the solution, indicating
    %     that the model is insensitive to some of its parameters.  That may be because those
    %     parameters are not present in the model, or otherwise do not affect the predicted
    %     values.  It may also be due to numerical underflow in the model function, which can
    %     sometimes be avoided by choosing better initial parameter values, or by rescaling or
    %     recentering.  Parameter estimates may be unreliable.
    initial_params = [0.05,0.9,23,0.06];
    
    params = nlinfit(eu_d, sim, general_sigmoid, initial_params);
    f = @(x) general_sigmoid(params, x);
    f_inv = @(y) (-log10((params(2)-params(1))./(y-params(1))-1) ./params(4))+(params(3));
end

