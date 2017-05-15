rng(123123123);

% Simulate data
theta_true = 4;
pi_true = 0.3;
n_samples = 10000;
z = (rand(n_samples,1) < pi_true) + 1; % 2 with probability pi_true
x = zeros(n_samples,1);
for i = 1:n_samples
    if z(i)==1
        x(i) = randn; % N(0,1)
    elseif z(i)==2
        x(i) = randn + theta_true;  % N(theta_true,1)
    end
end


% Parameters of the prior distributions.
alpha0 = 0.5; % pi ~ Beta(alpha0, alpha0)
beta0 = 0.2; % th ~ N(0, 1/beta0)

n_iter = 30;
% To keep track of the estimates of pi and theta in different iterations:
pi_est = zeros(n_iter,1);
th_est = zeros(n_iter,1);

% (1) Initialize global variational parameters
alpha_pi = alpha0; % q(pi) = Beta(pi | alpha_pi, beta_pi)
beta_pi = alpha0;
% q(th) = Normal(eta1, eta2)
eta1 = 0; % First natural parameter of the Gaussian: mu/var
eta2 = -0.05; % Second natural parameter of the Gaussian: -1/(2var) 

% These are the 'standard' Gaussian parameters of the factor q(th)
normal_precision = -2 .* eta2;
normal_mean = eta1 ./ normal_precision;

% (2) Define the step-size schedule
delay = 1;
forget_rate = 0.9;
step_sizes = (1:n_iter + delay).^(-1*forget_rate);
batch_size = 50;


for iter = 1:n_iter
    
    % Allocate:
    pi_par = zeros(batch_size,2); % First and second variational parameters for q(pi) for each sampled data point.
    eta = zeros(batch_size,2); % First and second variational parameters for q(th) for each sampled data point.
    
    % (4) Values that depend on q(pi) and that are needed when computing 
    % the responsibilities. These are the same for all possible samples,
    % and can therefore be calculated once outside the inner loop.
    E_log_pi = psi(alpha_pi) - psi(alpha_pi + beta_pi);
    E_log_pi_c = psi(beta_pi) - psi(alpha_pi + beta_pi);
    
    
    for sampling_round = 1:batch_size
        
        selected_index = ceil(rand*n_samples); % Select randomly a sample whose local parameters to update
        
        % (7):
        E_log_var = (x(selected_index) - normal_mean).^2 + 1/normal_precision;
        
        % (8) Compute the responsibilites, factor q(z)
        log_rho1 = E_log_pi_c - 0.5 .* log(2*pi) - 0.5 .* (x(selected_index).^2);
        log_rho2 = E_log_pi - 0.5 .* log(2*pi) - 0.5 .* E_log_var;
        max_log_rho = max(log_rho1, log_rho2); % Normalize to avoid numerical problems when exponentiating.
        rho1 = exp(log_rho1 - max_log_rho);
        rho2 = exp(log_rho2 - max_log_rho);
        r2 = rho2 ./ (rho1 + rho2);
        r1 = 1 - r2;
        
        
        % (9) Compute intermediate global variational parameters of the
        % factor q(pi)=Beta(par1,par2), assuming the sampled data item is 
        % replicated n_samples times.
        N1 = n_samples .* r1;
        N2 = n_samples .* r2;
        pi_par(sampling_round,1) = N2 + alpha0; % First parameter
        pi_par(sampling_round,2) = N1 + alpha0; % Second parameter
        
        
        % (10) Compute intermediate variational (natural) parameters of the 
        % factor q(theta)=normal(par1,par2), par1=mu/var, par2=-1/(2var), 
        % where mu and var are the mean and variance of the Gaussian
        % distribution. Again assume that the sampled data item is
        % replicated n_samples times.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        beta2inv = (1/(beta0+N2));
        X_bar = (1/N2) * sum(r2 .* x(selected_index));
        m2 = beta2inv*N2*X_bar;
        eta(sampling_round,1) = m2/beta2inv; % EXERCISE % The 1st natural parameter
        eta(sampling_round,2) = (-1)/(2*beta2inv); % EXERCISE % The 2nd natural parameter
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    % (12) Update global variational parameters of factor q(pi)
    aux = mean(pi_par,1); % New estimates, average over sampled data points.
    alpha_pi_new = aux(1); beta_pi_new = aux(2);
    alpha_pi = (1-step_sizes(iter)) .* alpha_pi + step_sizes(iter) .* alpha_pi_new; % Updated estimate (combination of old and new)
    beta_pi = (1-step_sizes(iter)) .* beta_pi + step_sizes(iter) .* beta_pi_new;
    
    % (12) Update global variational parameters of factor q(th)'
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    aux = mean(eta,1); % New estimates, average over sampled data points.
    eta1_new = aux(1); eta2_new = aux(2);
    eta1 = (1-step_sizes(iter)) .* eta1  + step_sizes(iter) .* eta1_new ; % EXERCISE
    eta2 = (1-step_sizes(iter)) .* eta2  + step_sizes(iter) .* eta2_new ; % EXERCISE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % (13) Compute the 'standard' mean and precision parameters of 
    % of the factor q(th).
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    normal_precision = -2 .* eta2; % EXERCISE
    normal_mean = eta1 ./ normal_precision; % EXERCISE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Keep track of the current estimates
    pi_est(iter) = alpha_pi / (alpha_pi + beta_pi);
    th_est(iter) =  -0.5 .* eta1 ./ eta2;
end

disp(num2str([pi_est th_est]));
% With large n_samples, this should converge to the (pi_true, theta_true).
