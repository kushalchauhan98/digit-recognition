input_layer_size =5
hidden_layer1_size = 3
hidden_layer2_size = 3
num_labels = 3
m = 100;
lambda = 12;
X = sin(rand(100, 5));
y = [1;2;3];
y = [y;y;y;y;y;y;y;y;y;y];
y = [y;y;y;1;2;3;1;2;3;1;2;3; 1];
theta = rand(39, 1);
[J, grad] = cost_func(theta, X, y, input_layer_size, hidden_layer1_size, ...
					hidden_layer2_size, num_labels, lambda);
ngrad = zeros(size(grad));
cost = @(p)cost_only(p, X, y, input_layer_size, hidden_layer1_size, ...
					hidden_layer2_size, num_labels, lambda);
del = zeros(39, 1);
for i=1:39
		del(i) = 0.0001;
	ngrad(i) = (cost(theta + del) - cost(theta - del)) / 0.0002;
	del(i) = 0;
end

[grad ngrad]