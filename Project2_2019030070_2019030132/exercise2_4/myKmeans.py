import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# Randomly select initial centroids from the dataset.
def initialize_centroids(X, K):
    indices = np.random.choice(X.shape[0], K, replace=False)
    centroids = X[indices]
    return centroids

# Finds the closest centroid for each sample
def find_closest_centroids(X, centroids):
    idx = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        idx[i] = np.argmin(distances)
    return idx

# Compute the mean of samples assigned to each centroid
def compute_centroids(X, idx, K):
    n = X.shape[1]
    centroids = np.zeros((K, n))
    for k in range(K):
        points = X[idx == k]
        if len(points) > 0:
            centroids[k] = np.mean(points, axis=0)
    return centroids

# K-means algorithm for a specified number of iterations
def run_kmeans(X, initial_centroids, max_iters):
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    return centroids, idx

# Initialize centroids randomly
def kmeans_init_centroids(X, K):
    return initialize_centroids(X, K)

# Load the image
image = io.imread('Fruit.png')

# Check the shape of the image
print(f"Original image shape: {image.shape}")

# Handle grayscale images by converting them to RGB
if len(image.shape) == 2:  # Grayscale image
    image = np.stack((image,) * 3, axis=-1)
elif image.shape[2] == 4:  # RGBA image
    image = image[:, :, :3]  # Discard the alpha channel

# Normalize image values in the range 0 - 1
image = image / 255.0

# Size of the image
img_size = image.shape
print(f"Processed image shape: {img_size}")

# Reshape the image to be a Nx3 matrix (N = num of pixels)
X = image.reshape(img_size[0] * img_size[1], 3)
print(f"Reshaped image matrix: {X.shape}")

# Perform K-means clustering
K = 200
max_iters = 10

# Initialize the centroids randomly
initial_centroids = kmeans_init_centroids(X, K)

# Run K-Means
centroids, idx = run_kmeans(X, initial_centroids, max_iters)

# K-Means Image Compression
print('\nApplying K-Means to compress an image.\n')

# Find closest cluster members
idx = find_closest_centroids(X, centroids)

# Recover the image from the indices
X_recovered = centroids[idx]

# Reshape the recovered image into proper dimensions
X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3)

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original')

# Display compressed image side by side
plt.subplot(1, 2, 2)
plt.imshow(X_recovered)
plt.title(f'Compressed, with {K} colors.')

plt.show()
