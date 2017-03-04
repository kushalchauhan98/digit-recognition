input_layer_size = 785;
hidden_layer1_size = 500;
hidden_layer2_size = 500;
num_labels = 10;
load train

[m, n] = size(X);

lambda = 0;
initial_theta = rand(hidden_layer1_size*input_layer_size + ...
					num_labels * (hidden_layer2_size+1) + ...
					hidden_layer2_size * (hidden_layer1_size+1), 1);
initial_theta = initial_theta*2*0.12 - 0.12;
options = optimset('GradObj', 'on', 'MaxIter', 100);
cost = @(p)cost_func(p, X, y, input_layer_size, hidden_layer1_size, ...
					 hidden_layer2_size, num_labels, lambda);

theta = fmincg(cost, initial_theta, options);
if(0)
	num_iter = 20;
	J_list = zeros(1, num_iter);
	alpha = 0.01;
	for i=1:num_iter
		disp(i);
		fflush(stdout)
		[J_list(i) grad] = cost(theta);
		theta -= alpha * grad;
	end
	plot(1:num_iter, J_list)
end
Theta1 = reshape(theta(1:(input_layer_size*hidden_layer1_size)), ...
				hidden_layer1_size, input_layer_size);			
				
Theta2 = reshape(theta((input_layer_size*hidden_layer1_size + 1):...
						input_layer_size*hidden_layer1_size + ...
						(hidden_layer1_size+1)*hidden_layer2_size), ...
				hidden_layer2_size, (hidden_layer1_size + 1));		
				
Theta3 = reshape(theta((input_layer_size*hidden_layer1_size + ...
				(hidden_layer1_size+1)*hidden_layer2_size + 1):end), ...
				num_labels, (hidden_layer2_size + 1));
				
mtest = size(Xtest, 1);
a2 = zeros(mtest, hidden_layer1_size);
z2 = zeros(mtest, hidden_layer1_size);
a3 = zeros(mtest, hidden_layer2_size);
z3 = zeros(mtest, hidden_layer2_size);
a4 = zeros(mtest, num_labels);
z4 = zeros(mtest, num_labels);

z2 = Xtest*Theta1';
a2 = sigmoid(z2);
a2 = [ones(mtest, 1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);
a3 = [ones(mtest, 1) a3];
z4 = a3*Theta3';
a4 = sigmoid(z4);
[x, ix] = max(a4, [], 2);
accuracy = 100*sum((ix-1)==ytest)/mtest