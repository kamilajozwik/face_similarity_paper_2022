function [ f ] = linear_theta_fit( data )
    [r1, r2, theta, ~, sim, ~] = extract_results(data);
    X = [ones(1, length(r1)); r1; r2; theta]';
    y = sim';
    
    b = regress(y, X);
    f = @(x) b(1) + b(2)*x(:,1) + b(3)*x(:,2) + b(4)*x(:,3);
end

