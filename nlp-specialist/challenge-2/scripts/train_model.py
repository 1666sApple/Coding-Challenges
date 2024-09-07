import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.model_selection import train_test_split
from model.dataset import read_file
from model.features import clean_sentence, tokenized_data, preprocess_text, extract_features
from model.train import train_xgb_model
import joblib

def main():
    sentences, labels = read_file()
    clean_sentences = [clean_sentence(sent) for sent in sentences]
    tokenized_sentences = [tokenized_data(sent) for sent in clean_sentences]
    processed_data = [preprocess_text(tokens) for tokens in tokenized_sentences]

    X, vectorizer = extract_features(processed_data)

    label_dict = {'neg': 0, 'ntr': 1, 'pos': 2}
    y = np.array([label_dict[label] for label in labels])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    best_model = train_xgb_model(X_train, y_train)
    filename = 'xgb_model_best.sav'
    joblib.dump(best_model, filename)

    def calculate_accuracy(model, X_test, y_test):
        from sklearn.metrics import accuracy_score, confusion_matrix
        y_pred = model.predict(X_test)
        print("Accuracy: ", accuracy_score(y_test, y_pred))
        print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))

    calculate_accuracy(best_model, X_test, y_test)

if __name__ == "__main__":
    main()
