cd Desktop/MachineLearning/machine-learning-ex2/ex2
data = load('ex2data1.txt')
X=data(:,[1,2])
y=data(:,3);
plotData(X,y)
[m, n] = size(X);
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);

X_t = X'
z = X .* initial_theta'

h = sigmoid(z)
J = (sum(-y .* log(h) - (1-y) .* log(1-h)))/m
for i=1:1000
  theta = theta - (alpha/m)* transpose(X) .* sum(h - y)
