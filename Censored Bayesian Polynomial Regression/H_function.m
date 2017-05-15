function h = H_function(u)
    numerator = logPDF(u, 0, 1);
    r = 1 - normcdf(u);
    if r == 0 % The value from the cumulative distribution function is rounded to 1
        denominator = log(1 - exp(-1.4*u)) - log(u) - u^2/2 - 1.04557; % Approximation by Karagiannidis & Lioumpas (2007)
    else
        denominator = log(r);
    end
    h = exp(numerator - denominator);
end

function p = logPDF(x, mu, sigma)
    p = -log((sqrt(2*pi)*sigma)) - (x - mu)^2/(2*sigma^2);
end