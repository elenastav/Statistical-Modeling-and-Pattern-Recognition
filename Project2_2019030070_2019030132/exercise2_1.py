import numpy as np
import matplotlib.pyplot as plt

data = {
    'omega1': np.array([[0.1, 1.1], [6.8, 7.1], [-3.5, -4.1], [2, 2.7], [4.1, 2.8],
                        [3.1, 5], [-0.8, -1.3], [0.9, 1.2], [5, 6.4], [3.9, 4]]),
    'omega2': np.array([[7.1, 4.2], [-1.4, -4.3], [4.5, 0], [6.3, 1.6], [4.2, 1.9],
                        [1.4, -3.2], [2.4, -4], [2.5, -6.1], [8.4, 3.7], [4.1, -2.2]]),
    'omega3': np.array([[-3, -2.9], [0.5, 8.7], [2.9, 2.1], [-0.1, 5.2], [-4, 2.2],
                        [-1.3, 3.7], [-3.4, 6.2], [-4.1, 3.4], [-5.1, 1.6], [1.9, 5.1]]),
    'omega4': np.array([[-2, -8.4], [-8.9, 0.2], [-4.2, -7.7], [-8.5, -3.2], [-6.7, -4],
                        [-0.5, -9.2], [-5.3, -6.7], [-8.7, -6.4], [-7.1, -9.7], [-8, -6.3]]),
}

# Plotting the samples
plt.scatter(data['omega1'][:, 0], data['omega1'][:, 1], color='red', label='omega1')
plt.scatter(data['omega2'][:, 0], data['omega2'][:, 1], color='blue', label='omega2')
plt.scatter(data['omega3'][:, 0], data['omega3'][:, 1], color='green', label='omega3')
plt.scatter(data['omega4'][:, 0], data['omega1'][:, 1], color='purple', label='omega4')

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('Samples from Different Classes')
plt.show()


# Define the perceptron algorithm using a while loop
def batch_perceptron(features, labels, max_iterations=100):
    weights = np.zeros(features.shape[1] + 1)
    iteration = 0
    while iteration < max_iterations:
        any_misclassified = False
        for i in range(len(features)):
            feature_with_bias = np.append(features[i], 1)
            activation = np.dot(weights, feature_with_bias)
            if activation * labels[i] <= 0:
                weights += labels[i] * feature_with_bias
                any_misclassified = True
        iteration += 1
        if not any_misclassified:
            break
    return weights, iteration



# Function to plot the decision boundary
def plot_decision_boundary(a, X_train, y_train):
    x_min, x_max = min(X_train[:, 0]) - 1, max(X_train[:, 0]) + 1
    y_min, y_max = min(X_train[:, 1]) - 1, max(X_train[:, 1]) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # Include the bias term in the calculation
    Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel(), np.ones(xx.ravel().shape)], a))
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'Decision Boundary (a = {a})')
    plt.show()


# Prepare the data and apply the perceptron algorithm sequentially for each pair of classes
pair_names = ["Class 1 vs Class 2", "Class 2 vs Class 3", "Class 3 vs Class 4"]
class_pairs = [('omega1', 'omega2'), ('omega2', 'omega3'), ('omega3', 'omega4')]

for i, (class1, class2) in enumerate(class_pairs):
    # Prepare data for this class pair
    X_train = np.vstack((data[class1], data[class2]))  # Combine samples
    y_train = np.hstack((np.ones(len(data[class1])), -np.ones(len(data[class2]))))

    # Train the perceptron
    weights, iteration = batch_perceptron(X_train, y_train)
    print(f"{pair_names[i]}: Number of iterations = {iteration}")

    # Plot the decision boundary
    plot_decision_boundary(weights, X_train, y_train)
