load("C:\\Users\\Admin\\Desktop\\Digit Recognizer\\Neural Networks\\train")
X = X(:, 2:785);
Xval = Xval(:, 2:785);
Xtest = Xtest(:, 2:785);
X = [X; Xval(1:4200, :)];
y = [y; yval(1:4200, :)];
Xval = [Xval(4201:8400, :); Xtest];
yval = [yval(4201:8400, :); ytest];
X = X/255;
Xval = Xval/255;
p = 0;
accu = 0;
d = 0;
model = svmtrain(y, X, "-q -g 0.03 -c 10");
[p, accu, d] = svmpredict(yval, Xval, model);
Xtest =  csvread("..\\test.csv");
Xtest = Xtest/255;
ytest = zeros(size(Xtest, 1), 1);
[p, accu, d] = svmpredict(ytest, Xtest, model);
index= zeros(size(p));
for i=1:size(p, 1)
	index(i) = i;
end
p = [index p];
csvwrite("predict.csv", p);