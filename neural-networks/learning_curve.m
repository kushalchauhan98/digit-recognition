input_layer_size = 785;
hidden_layer1_size = 500;
hidden_layer2_size = 100;
num_labels = 10;
load train
lambda = 0;
m = size(X, 1);
m_list = [10, 50, 100, 500, 1000, 5000, 25200];
J_list = zeros(size(m_list));
Jval_list = zeros(size(m_list));
for i=1:size(m_list, 2)
	initial_theta = rand(hidden_layer1_size*input_layer_size + ...
						hidden_layer2_size * (hidden_layer1_size+1) +...
						num_labels * (hidden_layer2_size+1), 1);
	initial_theta = initial_theta*2*0.12 - 0.12;
	options = optimset('GradObj', 'on', 'MaxIter', 100);
	cost = @(p)cost_func(p, X(1:m_list(i), :), y(1:m_list(i), :), ...
						input_layer_size, hidden_layer1_size, ...
						hidden_layer2_size, num_labels, lambda);

	theta = fmincg(cost, initial_theta, options);
	J_list(i) = cost_only(theta, X(1:m_list(i), :), y(1:m_list(i), :),...
						input_layer_size, hidden_layer1_size, ...
						hidden_layer2_size, num_labels, lambda);
	Jval_list(i) = cost_only(theta, Xval, yval, input_layer_size,...
							hidden_layer1_size, hidden_layer2_size,...
							num_labels, lambda);
end
plot(m_list, J_list, m_list, Jval_list);
