from typing import List, Any, Dict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import csv
import numpy as np


class WeakClassifier:

    def __init__(self, classifier: str):
        models_types = [DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2, criterion="entropy"),
                        GaussianNB(),
                        LogisticRegression(C=100)]
        if classifier == 'decision-tree':
            model = models_types[0]
        elif classifier == 'gaussian':
            model = models_types[1]
        elif classifier == 'linear-svm':
            model = models_types[2]
        else:
            model = np.random.choice(models_types)
        self.__classifier = model

    def predict(self, features: List[np.ndarray]) -> np.ndarray:
        return np.array(self.__classifier.predict(features))

    def fit(self, features: List[np.ndarray], labels: np.ndarray, features_weight: np.ndarray):
        self.__classifier.fit(X=features, y=labels, sample_weight=features_weight)


class AdaboostModel:
    def __init__(self, n_estimators: int, classifier_model: str):
        self.__n_estimators = n_estimators
        self.__classifier_model = classifier_model
        self.__classifiers = []
        self.__classifiers_weight = []

    def fit(self, features, labels):
        features_weight = np.ones(len(features), dtype=float) / len(features)

        while len(self.__classifiers) < self.__n_estimators:
            classifier = self.__fit_classifier(features, labels, features_weight)
            classifier.fit(features, labels, features_weight)
            labeling_results = AdaboostModel.__get_classifier_accuracy(classifier, features, labels)
            classifier_weight = AdaboostModel.__calculate_classifier_weight(labeling_results, features_weight)
            features_weight = AdaboostModel.__build_new_wights(classifier_weight, features_weight, labeling_results)
            self.__classifiers.append(classifier)
            self.__classifiers_weight.append(classifier_weight)

    def __fit_classifier(self, features: List[np.ndarray], labels: np.ndarray, weights: np.ndarray) -> WeakClassifier:
        weak_classifier = WeakClassifier(self.__classifier_model)
        weak_classifier.fit(features, labels, weights)
        return weak_classifier

    @staticmethod
    def __calculate_classifier_weight(labeling_results: np.ndarray, features_weight: np.ndarray):
        classifier_accuracy = sum(labeling_results * features_weight)
        return 0.5 * np.log2(classifier_accuracy / (1 - classifier_accuracy))

    @staticmethod
    def __get_classifier_accuracy(classifier: WeakClassifier, features: List[np.ndarray], labels: np.ndarray):
        predicted_labels = classifier.predict(features)
        labeling_results = predicted_labels == labels
        return labeling_results

    @staticmethod
    def __build_new_wights(classifier_weight: float, old_weights: np.array, labeling_results: np.ndarray):
        multiplier_constants = np.array([np.exp(-classifier_weight) if result else np.exp(classifier_weight)
                                         for result in labeling_results])
        new_non_normalized_weights = old_weights * multiplier_constants
        new_normalized_weights = new_non_normalized_weights / sum(new_non_normalized_weights)
        return new_normalized_weights

    def predict(self, features: List[List[Any]]) -> np.ndarray:
        predictions_per_feature = list()
        for feature in features:
            predictions_per_feature.append([classifier.predict([feature])[0] for classifier in self.__classifiers])

        weighted_predictions_per_feature = np.array(predictions_per_feature) * np.array(self.__classifiers_weight)
        return np.sign(weighted_predictions_per_feature.sum(axis=1))


if __name__ == '__main__':
    def validation(item):
        if item == 'x' or item == 'positive':
            return 1
        if item == 'o' or item == 'negative':
            return -1
        if item == 'b':
            return 0
        raise ValueError


    feature_values, label_values = list(), list()
    with open('tic-tac-toe.data', newline='') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            feature_values.append(np.array([validation(item) for item in row[:-1]]))
            label_values.append(validation(row[-1]))

    label_values = np.array(label_values)
    features_train, features_test, labels_train, labels_test = train_test_split(feature_values, label_values, test_size=0.2, random_state=42)
    model_x = AdaboostModel(100, '')
    model_x.fit(features_train, labels_train)
    w = model_x.predict(features_test)
    labels_test = np.array(labels_test)
    print(sum(w == labels_test) / len(w == labels_test))

    model_y = AdaBoostClassifier(n_estimators=100)
    model_y.fit(features_train, labels_train)
    w = model_y.predict(features_test)
    labels_test = np.array(labels_test)
    print(sum(w == labels_test)/len(w == labels_test))
