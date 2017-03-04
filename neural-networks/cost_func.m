function [J, grad] = cost_func(theta, X, y, input_layer_size, ...
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
	Theta1_grad = zeros(size(Theta1));
	Theta2_grad = zeros(size(Theta2));
	Theta3_grad = zeros(size(Theta3));
	
	a2 = zeros(m, hidden_layer1_size);
	z2 = zeros(m, hidden_layer1_size);
	d2 = zeros(m, hidden_layer1_size);
	a3 = zeros(m, hidden_layer2_size);
	z3 = zeros(m, hidden_layer2_size);
	d3 = zeros(m, hidden_layer2_size);
	a4 = zeros(m, num_labels);
	z4 = zeros(m, num_labels);
	d4 = zeros(m, num_labels);
	
	z2 = X*Theta1';
	a2 = sigmoid(z2);
	a2 = [ones(m, 1) a2];
	z3 = a2*Theta2';
	a3 = sigmoid(z3);
	a3 = [ones(m, 1) a3];
	z4 = a3*Theta3';
	a4 = sigmoid(z4);
	J = 0;
	Y = zeros(m, num_labels);
	for i=0:(num_labels-1)
		J += sum(-(y==i).*log(a4(:,(i+1))) - (1 - (y==i)).*log(1 - a4(:,(i+1))))/m;
		Y(:, (i+1)) = (y==i);
	end
	J += lambda*(sum(sum(Theta1(:, 2:end).^2)) + ...
				sum(sum(Theta2(:, 2:end).^2)) + ...
				sum(sum(Theta3(:, 2:end).^2)))/(2*m);
			
	d4 += a4 - Y;
	Theta3_grad += d4'*a3/m;
	d3 += d4*Theta3(:, 2:end) .* a3(:,2:end) .*(1-a3(:,2:end));
	Theta2_grad += d3'*a2/m;
	d2 += d3*Theta2(:, 2:end) .* a2(:,2:end) .*(1-a2(:,2:end));
	Theta1_grad += d2'*X/m;
	
	Theta1_grad(:,2:end) += lambda * Theta1(:,2:end)/m;
	Theta2_grad(:,2:end) += lambda * Theta2(:,2:end)/m;
	Theta3_grad(:,2:end) += lambda * Theta3(:,2:end)/m;
	
	grad = [Theta1_grad(:); Theta2_grad(:); Theta3_grad(:)];
end