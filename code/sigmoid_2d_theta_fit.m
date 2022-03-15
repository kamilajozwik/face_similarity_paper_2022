function [ f ] = sigmoid_2d_theta_fit( data )
    [r1, r2, theta, ~, sim, ~] = extract_results(data);
    X = [r1 + r2; theta]';
    Y = sim';
    
    general_2d_sigmoid = @(b, x) b(1)+(b(2)-b(1)) ./ (1+10.^((b(3) + b(4) * x(:, 1) + b(5) * x(:, 2)) .* b(6)));
    b = nlinfit(X, Y, general_2d_sigmoid, [0 1 39 -1 -1 1]);
    f = @(x) general_2d_sigmoid(b, [x(:,1) + x(:,2), x(:,3)]);
end

