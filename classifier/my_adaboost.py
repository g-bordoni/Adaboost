from typing import List, Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


class MyAdaboostClassifier:
    def __init__(self, n_estimators: int, classifier_model: str):
        self.__n_estimators = n_estimators
        self.__classifier_model = classifier_model
        self.__classifiers = []
        self.__classifiers_weight = []

    def fit(self, features, labels):
        self.__classifiers = []
        self.__classifiers_weight = []
        features_weight = np.ones(len(features), dtype=float) / len(features)

        while len(self.__classifiers) < self.__n_estimators:
            classifier = self.__get_classifier_instance()
            classifier.fit(features, labels, features_weight)
            labeling_results = MyAdaboostClassifier.__get_classifier_accuracy(classifier, features, labels)
            classifier_weight = MyAdaboostClassifier.__calculate_classifier_weight(labeling_results, features_weight)
            MyAdaboostClassifier.__build_new_weights(classifier_weight, features_weight, labeling_results)
            self.__classifiers.append(classifier)
            self.__classifiers_weight.append(classifier_weight)

    def __get_classifier_instance(self):
        if self.__classifier_model == 'stump':
            return DecisionTreeClassifier(max_depth=1, criterion='entropy')
        elif self.__classifier_model == 'weak-logistic-regression':
            return LogisticRegression(C=100)
        raise ValueError('Model invalid')

    @staticmethod
    def __get_classifier_accuracy(classifier: Any, features: List[np.ndarray], labels: np.ndarray):
        predicted_labels = classifier.predict(features)
        labeling_results = predicted_labels == labels
        return labeling_results

    @staticmethod
    def __calculate_classifier_weight(labeling_results: np.ndarray, features_weight: np.ndarray):
        classifier_accuracy = sum(labeling_results * features_weight)
        return 0.5 * np.log(classifier_accuracy / (1 - classifier_accuracy))

    @staticmethod
    def __build_new_weights(classifier_weight: float, weights: np.array, labeling_results: np.ndarray):
        multiplier_constants = np.array([np.exp(-classifier_weight) if result else np.exp(classifier_weight)
                                         for result in labeling_results])
        weights *= multiplier_constants
        weights /= sum(weights)

    def predict(self, features: List[np.ndarray]) -> np.ndarray:
        predictions_per_feature = list()
        for feature in features:
            predictions_per_feature.append([classifier.predict([feature])[0] for classifier in self.__classifiers])

        weighted_predictions_per_feature = np.array(predictions_per_feature) * np.array(self.__classifiers_weight)
        return np.sign(weighted_predictions_per_feature.sum(axis=1)).astype('int32')

    def score(self, features: List[np.ndarray], labels: np.ndarray):
        return accuracy_score(labels, self.predict(features))

