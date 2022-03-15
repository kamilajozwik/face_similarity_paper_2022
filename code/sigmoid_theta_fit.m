function [ f ] = sigmoid_theta_fit( data )
    [r1, r2, theta, ~, sim, ~] = extract_results(data);
    X = [r1; r2; theta]';
    Y = sim';
    
    general_3d_sigmoid = @(b, x) b(1)+(b(2)-b(1)) ./ (1+10.^((b(3) + b(4) * x(:, 1) + b(5) * x(:, 2) + b(6) * x(:, 3)) .* b(7)));
    params = nlinfit(X, Y, general_3d_sigmoid, [0 1 40 -1 -1 -1 1], statset('MaxIter', 100));
    f = @(x) general_3d_sigmoid(params, x);
end

