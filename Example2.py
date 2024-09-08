import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LogisticRegression import LogisticRegression


def preprocess_dataset():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_scaled = StandardScaler().fit_transform(X)  # scaling the data

    return train_test_split(X_scaled, y, test_size=0.2, shuffle=True)


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=2)

    X_train, X_test, y_train, y_test = preprocess_dataset()

    model = LogisticRegression(learning_rate=0.4, threshold=0.0001, max_iter=100000, multi_class=True)
    model.fit(X_train, y_train)

    print(f'score: {model.score(X_test, y_test):.3f}')
    print('weights:')
    print(*(f'classifier {i + 1}: {model.weights[i]}' for i in range(model.weights.shape[0])), sep='\n')
