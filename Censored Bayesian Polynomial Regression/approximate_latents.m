function [ zs ] = approximate_latents( w, x, y, c, c_star, beta )
%APPROXIMATE_LATENT Summary of this function goes here
%   Detailed explanation goes here
    l = length(x);
    d = length(w) - 1;
    zs = zeros(l, 1);
    for i=1:l
        if c(i) == 0
            zs(i) = y(i);
        else
            mu_i = w' * phiX(x(i), d);           
            gamma = (c_star - mu_i) * beta;
            Z = 0.5*(1 - erf(gamma / sqrt(2)));
            phi_gamma = exp(-0.5*gamma^2) / sqrt(2*pi);
            zs(i) = mu_i + (beta^-1)*phi_gamma / Z;
        end
    end
end

