function [ w ] = update_weights( zs, x, alpha, beta, d )
%UPDATE_WEIGHTS Summary of this function goes here
%   Detailed explanation goes here
    l = length(x);  
    U = zeros(d + 1, 1);
    V = 0;
    for i=1:l
        phi_x = phiX(x(i), d);
        U = U + zs(i)*phi_x;
        V = V + phi_x * phi_x';
    end

    %size((beta*U))
    %size(alpha*eye(size(phi_x)) + beta*V)
    den = inv((alpha*eye(size(phi_x)) + beta*V));
    w = den*(beta*U);
end


