import numpy as np
from scipy.stats import norm, multivariate_normal
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw


class MyBayesClassifier:
    def __init__(self):
        self.class_priors = {}
        self.class_stats = {}

    def train(self, X, y):
        """
        Train the classifier under the assumption of Gaussian distributions:
        calculate priors and Gaussian distribution parameters for each class.

        Args:
        X (pd.DataFrame): DataFrame with features.
        y (pd.Series): Series with target class labels.
        """
        self.classes_ = np.unique(y)
        for class_label in self.classes_:
            # Filter data by class
            X_class = X[y == class_label]

            # Calculate prior probability for the class
            self.class_priors[class_label] = len(X_class) / len(X)

            # Calculate mean and covariance for the class
            # Adding a small value to the covariance for numerical stability
            self.class_stats[class_label] = {
                'mean': X_class.mean(),
                'cov': X_class.cov() + 1e-4 * np.eye(X_class.shape[1])
            }

    def predict(self, X):
        """
        Predict class labels for each test sample in X.

        Args:
        X (pd.DataFrame): DataFrame with features to predict.

        Returns:
        np.array: Predicted class labels.
        """
        predictions = []
        for _, row in X.iterrows():
            predictions.append(self._predict_instance(row))
        return np.array(predictions)

    def _predict_instance(self, x):
        """
        Private helper to predict the class for a single instance.

        Args:
        x (pd.Series): A single data point's features.

        Returns:
        The predicted class label.
        """
        posteriors = []

        # Calculate the posterior probability for each class
        for class_label in self.classes_:
            prior = self.class_priors[class_label]
            mean = self.class_stats[class_label]['mean']
            cov = self.class_stats[class_label]['cov']
            likelihood = multivariate_normal.pdf(x, mean=mean, cov=cov)
            posterior = prior * likelihood
            posteriors.append(posterior)

        # Choose the class with the highest posterior probability
        prediction = np.argmax(posteriors)
        return self.classes_[prediction]


# Calculate the bounding box
def calculate_bounding_box(image):
    # Find non-zero foreground pixels
    nonzero_pixels = np.nonzero(image)
    # Check if there are any foreground pixels
    if nonzero_pixels[0].size == 0:
        return np.nan  # Return NaN if no foreground pixels found

    # Get minimum and maximum coordinates of foreground pixels
    min_row, min_col = np.min(nonzero_pixels, axis=1)
    max_row, max_col = np.max(nonzero_pixels, axis=1)

    return min_col, min_row, max_col, max_row


# Function to calculate aspect ratio
def aspect_ratio(image):
    """Calculates the aspect ratio of the bounding box around the foreground pixels."""
    try:
        # Extract image data and reshape it (assuming data is in a column named 'image')
        img = image.values.reshape(28, 28)

        # Find non-zero foreground pixels
        nonzero_pixels = np.nonzero(img)

        # Check if there are any foreground pixels
        if nonzero_pixels[0].size == 0:
            return np.nan  # Return NaN if no foreground pixels found

        # Get minimum and maximum coordinates of foreground pixels
        min_col, min_row, max_col, max_row = calculate_bounding_box(img)

        # Calculate bounding box dimensions
        width = max_col - min_col
        height = max_row - min_row

        # Calculate aspect ratio
        aspect_ratio = width / height
        return aspect_ratio

    except (KeyError, ValueError) as e:
        print(f"Error processing image: {e}")
        return np.nan  # Return NaN for rows with errors


def foreground_pixels(image):
    """
    Calculate the pixel density of the image, defined as the
    count of non-zero pixels

    Args:
    image (np.array): A 1D numpy array representing the image.

    Returns:
    int: The pixel density of the image.
    """
    try:
        # Find non-zero foreground pixels
        nonzero_pixels = np.count_nonzero(image)
        if nonzero_pixels == 0:
            print(f"Warning: Couldn't find nonzero pixels.")
            return np.nan  # Return NaN if no foreground pixels found
        return nonzero_pixels
    except ValueError as e:
        print(f"Error processing image: {e}")
        return np.nan  # Return NaN for rows with errors


def calculate_centroid(image):
    """
    Calculate the normalized centroid (center of mass) of the image.

    Returns:
    tuple: The (x, y) coordinates of the centroid normalized by image dimensions.
    """
    try:
        # Extract image data and reshape it (assuming data is in a column named 'image')
        img = image.values.reshape(28, 28)
        rows, cols = img.shape

        # Calculate total mass (sum of non-zero pixels)
        total_mass = np.count_nonzero(img)

        # Calculate x and y coordinates of centroid
        x_center = np.sum(np.arange(cols) * np.sum(img, axis=0)) / total_mass
        y_center = np.sum(np.arange(rows) * np.sum(img, axis=1)) / total_mass

        # Normalize centroid by image dimensions
        centroid_x = x_center / cols
        centroid_y = y_center / rows

        # Create a single scalar as a centroid feature using x + (y * w) where w is the width of the image
        centroid = centroid_x + (centroid_y * cols)
        return centroid
    except ValueError as e:
        print(f"Error processing image: {e}")
        return np.nan  # Return NaN for rows with errors


def min_max_scaling(X, min_val=-1, max_val=1):
    """Scales features to a range between min_val and max_val."""
    X_scaled = (X - X.min()) / (X.max() - X.min())
    return min_val + (max_val - min_val) * X_scaled


def visualize_bounding_box(image, color='red'):
    """Visualizes the bounding box around the digit in an image."""
    bbox = calculate_bounding_box(image)

    # Create a drawing object
    sample_image_img = Image.fromarray(image.astype(np.uint8)).convert('RGB')
    scaling = 10
    sample_image_XL = sample_image_img.resize((28 * scaling, 28 * scaling), resample=Image.NEAREST)

    draw = ImageDraw.Draw(sample_image_img)
    # Draw the rectangle with desired fill color and outline (optional)
    draw.rectangle(bbox, outline=color, width=1)

    #sample_image_XL.show()
    sample_image_XL_bbox = sample_image_img.resize((28 * scaling, 28 * scaling), resample=Image.NEAREST)
    sample_image_XL_bbox.show()


def train_and_evaluate(df_train, df_test, target_train, target_test, data_train, data_test, features):
    # Prepare the training data
    trainData = df_train[features]
    classifier = MyBayesClassifier()
    classifier.train(trainData, target_train)

    # Create the respective features for the test samples
    df_test['aspect_ratio'] = data_test.apply(aspect_ratio, axis=1)
    df_test['aspect_ratio'] = min_max_scaling(df_test['aspect_ratio'])

    if 'fg_pixels' in features:
        df_test['fg_pixels'] = data_test.apply(foreground_pixels, axis=1)
        df_test['fg_pixels'] = min_max_scaling(df_test['fg_pixels'])

    if 'centroid' in features:
        df_test['centroid'] = data_test.apply(calculate_centroid, axis=1)
        df_test['centroid'] = min_max_scaling(df_test['centroid'])

    test_data = df_test[features]
    predictions = classifier.predict(test_data)
    accuracy = accuracy_score(target_test, predictions)
    print(f"Classification accuracy with features {features}:", accuracy)

    for sample in range(3):
        sample_image = data_train.iloc[sample].values.reshape(28, 28)
        visualize_bounding_box(sample_image)


def main():
    nTrainSamples = 10000  # specify 'None' if you want to read the whole file
    df_train = pd.read_csv('data/mnist_train.csv', delimiter=',', nrows=nTrainSamples)
    df_train = df_train[df_train['label'].isin([0, 1, 2])]  # Filter samples for digits 0, 1, and 2
    target_train = df_train.label
    data_train = df_train.iloc[:, 1:]

    nTestSamples = 1000  # specify 'None' if you want to read the whole file
    df_test = pd.read_csv('data/mnist_test.csv', delimiter=',', nrows=nTestSamples)
    df_test = df_test[df_test['label'].isin([0, 1, 2])]  # Filter samples for digits 0, 1, and 2
    target_test = df_test.label
    data_test = df_test.iloc[:, 1:]

    # Step 1: Calculate aspect ratio as the first feature
    df_train['aspect_ratio'] = data_train.apply(aspect_ratio, axis=1)
    df_train['aspect_ratio'] = min_max_scaling(df_train['aspect_ratio'])

    # Train and evaluate using only the aspect ratio
    features = ["aspect_ratio"]
    train_and_evaluate(df_train, df_test, target_train, target_test, data_train, data_test, features)

    # Step 2: Calculate the number of non-zero pixels as the second feature
    df_train['fg_pixels'] = data_train.apply(foreground_pixels, axis=1)
    df_train['fg_pixels'] = min_max_scaling(df_train['fg_pixels'])

    # Train and evaluate using aspect ratio and foreground pixels
    features = ["aspect_ratio", "fg_pixels"]
    train_and_evaluate(df_train, df_test, target_train, target_test, data_train, data_test, features)

    # Step 3: Calculate the centroid feature as the third feature
    df_train['centroid'] = data_train.apply(calculate_centroid, axis=1)
    df_train['centroid'] = min_max_scaling(df_train['centroid'])

    # Train and evaluate using all three features
    features = ["aspect_ratio", "fg_pixels", "centroid"]
    train_and_evaluate(df_train, df_test, target_train, target_test, data_train, data_test, features)

if __name__ == "__main__":
    main()
