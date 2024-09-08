import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from LogisticRegression import LogisticRegression
from BagOfWords import BagOfWords


def preprocess_dataset():
    df = pd.read_csv("..\spam_ham_dataset.csv")
    bow = BagOfWords(n_features=5000)

    X = bow.generate_vectors(df['text'])  # translates the dataset to a numerical one using bag of words
    y = np.array(df['label_num'].apply(lambda x: 1 if x == 1 else -1))  # changing the labels in the data to -1 and 1

    X_reduced = PCA(500).fit_transform(X)  # reducing the number of dimensions from 5000 to 500

    return train_test_split(X_reduced, y, test_size=0.2, shuffle=True)


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=2, threshold=500, edgeitems=5)

    X_train, X_test, y_train, y_test = preprocess_dataset()

    model = LogisticRegression(learning_rate=0.3, tolerance=0.05)
    model.fit(X_train, y_train)

    print(f'score: {model.score(X_test, y_test):.3f}')
    print(f'weights:\n {model.weights}')

    # for the full array of 500 weights, delete the 'threshold' and 'edgeitems'
    # parameters from the np.set_printoptions command at the beginning of the main


