function [ p ] = pearson( x, y )
    C=cov(x,y);
    p=C(2)/(std(x)*std(y));
end
