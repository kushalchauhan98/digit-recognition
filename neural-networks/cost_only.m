function J = cost_only(theta, X, y, input_layer_size, ...
								hidden_layer1_size, hidden_layer2_size,...
								num_labels, lambda)
	Theta1 = reshape(theta(1:(input_layer_size*hidden_layer1_size)), ...
					hidden_layer1_size, input_layer_size);				
	Theta2 = reshape(theta((input_layer_size*hidden_layer1_size + 1):...
							input_layer_size*hidden_layer1_size + ...
							(hidden_layer1_size+1)*hidden_layer2_size), ...
					hidden_layer2_size, (hidden_layer1_size + 1));		
	Theta3 = reshape(theta((input_layer_size*hidden_layer1_size + ...
					(hidden_layer1_size+1)*hidden_layer2_size + 1):end), ...
					num_labels, (hidden_layer2_size + 1));
	
	m = size(X, 1);
	
	a2 = zeros(m, hidden_layer1_size);
	z2 = zeros(m, hidden_layer1_size);
	a3 = zeros(m, hidden_layer2_size);
	z3 = zeros(m, hidden_layer2_size);
	a4 = zeros(m, num_labels);
	z4 = zeros(m, num_labels);
	
	z2 = X*Theta1';
	a2 = sigmoid(z2);
	a2 = [ones(m, 1) a2];
	z3 = a2*Theta2';
	a3 = sigmoid(z3);
	a3 = [ones(m, 1) a3];
	z4 = a3*Theta3';
	a4 = sigmoid(z4);
	J = 0;
	for i=0:(num_labels-1)
		J += sum(-(y==i).*log(a4(:,(i+1))) - (1 - (y==i)).*log(1 - a4(:,(i+1))))/m;
	end
	J += lambda*(sum(sum(Theta1(:, 2:end).^2)) + ...
				sum(sum(Theta2(:, 2:end).^2)) + ...
				sum(sum(Theta3(:, 2:end).^2)))/(2*m);
end