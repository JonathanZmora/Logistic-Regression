import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate=0.1, tolerance=0.0001, threshold=0.5, max_iter=10000, multi_class=False):
        self._weights = None
        self._learning_rate = learning_rate
        self._tolerance = tolerance
        self._threshold = threshold
        self._max_iter = max_iter
        self._multi_class = multi_class

    @staticmethod
    def sigmoid(z):
        """ Applies the sigmoid function to the input 'z' and returns the output """
        z = np.clip(z, -709.78, 709.78)  # to avoid overflow
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def cross_entropy(X, y, weights):
        """ Applies the cross entropy loss function to the data and given weights and returns the output"""
        power = -y * np.dot(X, weights)
        power = np.clip(power, -709.78, 709.78)  # to avoid overflow
        loss = (1 / X.shape[0]) * np.sum(np.log(1 + np.exp(power)))
        return loss

    def gradient_descent(self, X, y):
        """
        Applies the gradient descent algorithm to the data and returns the final set of weights.
        The algorithm stops when the cross entropy loss goes below the tolerance or when the number
        of iterations exceeds max_iter, whichever comes first.
        """
        weights = np.zeros(X.shape[1])
        for _ in range(self._max_iter):
            loss = self.cross_entropy(X, y, weights)
            if loss < self._tolerance:
                break
            power = -y * np.dot(X, weights)
            gradient = (-1 / X.shape[0]) * np.dot(X.T, y * self.sigmoid(power))
            weights -= self._learning_rate * gradient
        return weights

    def fit(self, X, y):
        """
        This method trains the model using the given data.
        If the multi_class option is set to True and there are 'num' different classes in y,
        the method will train 'num' different classifiers resulting in 'num' different vectors
        of weights stored in the array self._weights.
        Otherwise, self._weights will contain on vector of weights for binary classification.
        """
        X = np.c_[np.ones(X.shape[0]), X]

        # multi class classification - uses the one vs rest method
        if self._multi_class:
            labels = np.unique(y)
            self._weights = np.empty(shape=(labels.shape[0], X.shape[1]))
            for index, label in enumerate(labels):
                bin_y = np.where(label == y, 1, -1)
                self._weights[index] = self.gradient_descent(X, bin_y)

        # binary classification
        else:
            self._weights = self.gradient_descent(X, y)

    def predict_proba(self, X):
        """
        This method predicts and returns probability estimates for all samples in X.
        For multi_class classification, each model that was trained is used to predict the probabilities,
        thus the probabilities are returned as a matrix where each column represents a different models predictions.
        """
        X = np.c_[np.ones(X.shape[0]), X]

        if self._multi_class:
            probabilities = np.empty(shape=(X.shape[0], self._weights.shape[0]))
            for index, weight_vector in enumerate(self._weights):
                probabilities[:, index] = self.sigmoid(np.dot(X, weight_vector))

        else:
            probabilities = self.sigmoid(np.dot(X, self._weights))

        return probabilities

    def predict(self, X):
        """ Predict class labels for samples in X """
        probabilities = self.predict_proba(X)

        # multi class predictions - uses the one vs rest method and chooses the class with the highest probability
        if self._multi_class:
            predictions = [np.argmax(probabilities[i]) for i in range(probabilities.shape[0])]

        # binary predictions - classifies a sample as 1 if the probability is greater than the threshold
        else:
            predictions = np.array([-1 if p <= self._threshold else 1 for p in probabilities])

        return predictions

    def score(self, X, y):
        """ This method returns the mean accuracy on the given test data and labels """
        y_pred = self.predict(X)
        return np.mean(y == y_pred)

    @property
    def weights(self):
        return self._weights

