import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Function to plot the data
def plotData(X, y):
    pos = y == 1
    neg = y == 0
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k', label='Admitted')
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', c='y', label='Not admitted')

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + 2 * np.exp(-z))

# Calculate the cost function
def costFunction(theta, X, y):
    m = len(y)
    h = sigmoid(X @ theta)
    J = -1/m * (y @ np.log(h) + (1 - y) @ np.log(1 - h))
    return J

# Calculate the gradient of the cost function
def gradient(theta, X, y):
    m = len(y)
    h = sigmoid(X @ theta)
    grad = 1/m * X.T @ (h - y)
    return grad

# Plot decision boundary
def plotDecisionBoundary(theta, X, y):
    plotData(X[:, 1:], y)
    plot_x = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
    plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])
    plt.plot(plot_x, plot_y, label='Decision Boundary')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()

# Predict function
def predict(theta, X):
    prob = sigmoid(X @ theta)
    return (prob >= 0.5).astype(int)

# Load data
data = np.loadtxt('exam_scores_data1.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]

# Plotting data
plotData(X, y)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()
input('\nProgram paused. Press enter to continue.\n')

# Add intercept term to X
m, n = X.shape
X = np.concatenate([np.ones((m, 1)), X], axis=1)
initial_theta = np.zeros(n + 1)

# Compute cost and gradient at initial theta
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
print('Cost at initial theta (zeros):', cost)
print('Gradient at initial theta (zeros):', grad)
input('\nProgram paused. Press enter to continue.\n')

# Optimizing using minimize
options = {'gtol': 1e-6}
res = minimize(fun=costFunction, x0=initial_theta, args=(X, y), jac=gradient, method='TNC', options=options)

theta = res.x
cost = res.fun
print('Cost at theta found by minimize:', cost)
print('Theta:', theta)

# Plot decision boundary
plotDecisionBoundary(theta, X, y)
plt.show()
input('\nProgram paused. Press enter to continue.\n')

# Predict and calculate accuracy
exam_scores = np.array([1, 45, 85])
prob = sigmoid(np.dot(exam_scores, theta))
print(f'For a student with scores 45 and 85, we predict an admission probability of {prob:.2%}')

# Calculate training accuracy
p = predict(theta, X)
train_accuracy = np.mean(p == y) * 100
print(f'Train Accuracy: {train_accuracy:.2f}%')
