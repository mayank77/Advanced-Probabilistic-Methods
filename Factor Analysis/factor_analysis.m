clear

% Load data for digit 8
load digit8 
% variable x, 1032*784, rows samples, columns pixels on 28*28 grid
digit8 = x'; % Now the columns are the samples.

% Load data for digit 3
load digit3
% variable x, 1010*784
digit3 = x';

% Plot an example of each digit
figure
subplot(1,4,1); image(reshape(digit8(:,1),[28 28])'); title('Digit 8, orig');
subplot(1,4,2); image(reshape(digit8(:,2),[28 28])'); title('Digit 8, orig');
subplot(1,4,3); image(reshape(digit3(:,1),[28 28])'); title('Digit 3, orig');
subplot(1,4,4); image(reshape(digit3(:,2),[28 28])'); title('Digit 3, orig');

% Divide into training and test data
n_train = 900;
n_test = 50;
d8_train = digit8(:,1:n_train);
d8_test = digit8(:,n_train+1:n_train+n_test);

d3_train = digit3(:,1:n_train);
d3_test = digit3(:,n_train+1:n_train+n_test);

test_data = [d8_test d3_test];
test_labels = [8*ones(1,n_test) 3*ones(1,n_test)];



% Learn the FA model for each digit (M8/3: model for digit 8/3)
opts.maxit = 50; opts.plotprogress=0; opts.tol=0.1;
H = 5; % Assume 5 factors (for example...)

[d8_F, d8_diagPsi, d8_m, d8_loglik] = FA(d8_train, H, opts);
[d3_F, d3_diagPsi, d3_m, d3_loglik] = FA(d3_train, H, opts);
%(see demo_FA.m for a simple example)

% Plot the means and factor loadings (columns of the factor loading matrix)
% for the two models.

figure
subplot(2,3,1); image(reshape(d8_m,[28 28])'); title('D8, mean');
for i = 1:H
    subplot(2,3,1+i); image(reshape(d8_F(:,i),[28 28])'); title(['D8, FA' num2str(i)]); 
end

figure
subplot(2,3,1); image(reshape(d3_m,[28 28])'); title('D3, mean');
for i = 1:H
    subplot(2,3,1+i); image(reshape(d3_F(:,i),[28 28])'); title(['D3, FA' num2str(i)]);
end
% Compute the log-likelihood for observing each test point from digit 3
% model M3

d3_cov = d3_F * d3_F' + diag(d3_diagPsi);
m3_loglikelihood = logmvnpdf(test_data', d3_m', d3_cov);

% Compute the log-likelihood for observing each test point from digit 8
% model M8

d8_cov = d8_F * d8_F' + diag(d8_diagPsi);
m8_loglikelihood = logmvnpdf(test_data', d8_m', d8_cov);

% Normalize to avoid (some of the) numerical problems
max_val = max([m8_loglikelihood; m3_loglikelihood],[],1);
m8_loglikelihood = m8_loglikelihood - max_val;
m3_loglikelihood = m3_loglikelihood - max_val;


% Probability of observing each of test points from M3,
% assuming that both digits are equally probable a priori.
r3 = exp(m3_loglikelihood) ./ (exp(m3_loglikelihood) + exp(m8_loglikelihood));

% Classify each digit in the test data to the model that maximized
% its probability

digit3_indices = (r3 > 0.5);
estimated_labels = 8 .* ones(n_test*2,1);
estimated_labels(digit3_indices) = 3;

% Compute the classification accuracy by comparing the estimated cluster
% labels with the test_labels

accuracy = length(find(estimated_labels==test_labels'))*100 / length(estimated_labels)
