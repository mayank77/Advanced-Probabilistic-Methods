function [ logp ] = logPosterior( alpha, beta, w, x, zs )
%LOGPOSTERIOR Summary of this function goes here
%   Detailed explanation goes here
    l = length(x);
    A = -0.5 * alpha * (w' * w);
    B = 0;
    d = length(w) - 1;
    for i=1:l
        if zs(i)==5.4678
            err = log(1-(0.5*(1 + erf(sqrt(beta^2/2)*(zs(i) - w' * phiX(x(i), d))))));
        else
            err = -(beta/2)*((zs(i) - w' * phiX(x(i), d))^2);
        end
        B = B + err;
    end
    logp = A+B;
end
