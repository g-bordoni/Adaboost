from typing import List, Any, Dict
from sklearn.tree import DecisionTreeClassifier
import csv
import numpy as np
import pandas as pd


class DataInstance:
    def __init__(self, data):
        self.__data = data
        self.__weight = 1

    def update_weight(self, constant: float):
        self.__weight = int(self.__weight * constant)

    def get_data(self):
        return [self.__data] * self.__weight


class DecisionStump:
    def __init__(self):
        self.__stump = DecisionTreeClassifier(max_depth=1, criterion='entropy')
        self.__decision_wight = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.array(self.__stump.predict(features))

    def fit(self, features: np.ndarray, labels: np.ndarray):
        self.__stump.fit(X=features, y=labels)

    def set_decision_wight(self, wight: float):
        self.__decision_wight = wight

    def get_decision_weight(self) -> float:
        return self.__decision_wight


class AdaboostModel:
    def __init__(self, stumps_number: int):
        self.__stumps_number = stumps_number
        self.__data_instances = []
        self.__decision_stumps = []

    @staticmethod
    def __get_features_and_labels(data) -> (np.ndarray, np.ndarray):
        features, labels = list(), list()
        for item in data:
            features.append(item[0])
            labels.append(item[1])
        return np.array(features), np.array(labels)

    def fit(self, data):
        self.__data_instances = [DataInstance(data_instance) for data_instance in data]

        while len(self.__decision_stumps) < self.__stumps_number:
            decision_stump = self.__fit_stump()
            self.__set_weights(decision_stump)
            self.__decision_stumps.append(decision_stump)

    def predict(self, features: List[List[Any]]) -> np.ndarray:
        prediction_result = list()
        for feature in features:
            prediction_result.append(self.__predict_feature(feature))
        return np.array(prediction_result)

    def __predict_feature(self, feature: List[Any]) -> int:
        stumps_votes_by_predicted_labels = dict()
        for stump in self.__decision_stumps:
            stump_prediction = stump.predict([feature])[0]
            label_votes = stumps_votes_by_predicted_labels.get(stump_prediction, 0.0)
            label_votes += stump.get_decision_weight()
            stumps_votes_by_predicted_labels.update({stump_prediction: label_votes})
        most_voted_label_size = -1
        most_voted_label = None
        for voted_label in stumps_votes_by_predicted_labels:
            if stumps_votes_by_predicted_labels[voted_label] > most_voted_label_size:
                most_voted_label_size = stumps_votes_by_predicted_labels[voted_label]
                most_voted_label = voted_label
        return most_voted_label

    def __get_weighted_data(self) -> List[List[Any]]:
        weighted_data = list()
        for original_instance in self.__data_instances:
            weighted_data.extend(original_instance.get_data())
        return weighted_data

    def __fit_stump(self) -> DecisionStump:
        weighted_data = self.__get_weighted_data()
        features, labels = AdaboostModel.__get_features_and_labels(weighted_data)
        decision_stump = DecisionStump()
        decision_stump.fit(features, labels)
        return decision_stump

    def __set_weights(self, stump: DecisionStump):
        prediction_result_per_data_instance = dict()
        for data_instance in self.__data_instances:
            prediction_result_per_data_instance.update(AdaboostModel.__get_prediction_per_data_instance(data_instance,
                                                                                                        stump))
        new_stump_weight = AdaboostModel.__calculate_stump_weight(prediction_result_per_data_instance)
        stump.set_decision_wight(new_stump_weight)
        AdaboostModel.__set_data_instances_new_weights(new_stump_weight, prediction_result_per_data_instance)

    @staticmethod
    def __get_prediction_per_data_instance(data: DataInstance, model: DecisionStump) -> Dict[DataInstance, np.ndarray]:
        features, labels = AdaboostModel.__get_features_and_labels(data.get_data())
        predicted_labels = model.predict(features)
        labels_accuracy = predicted_labels == labels
        return {data: np.array([sum(labels_accuracy), len(labels_accuracy)])}

    @staticmethod
    def __calculate_stump_weight(instances_results: Dict[DataInstance, np.ndarray]):
        rightness_and_tries = np.array([0, 0])
        for data_instance in instances_results:
            rightness_and_tries += instances_results[data_instance]
        stump_accuracy = rightness_and_tries[0] / rightness_and_tries[1]
        new_stump_weight = 0.5 * np.log2(stump_accuracy / (1 - stump_accuracy))
        return new_stump_weight

    @staticmethod
    def __set_data_instances_new_weights(stump_weight: float, instances_results: Dict[DataInstance, np.ndarray]):
        for data_instance in instances_results:
            correctly_predicted, sample_size = instances_results[data_instance]
            perfectly_predicted = True if correctly_predicted == sample_size else False
            if not perfectly_predicted:
                data_instance.update_weight(np.exp(2 * stump_weight))


def split_folders(items: list, cross_validation_k=5):
    folders = list()
    folder_size = len(items) // cross_validation_k
    while len(folders) < cross_validation_k - 1:
        folder = list()
        while len(folder) < folder_size:
            chosen_index = np.random.randint(len(items))
            folder.append(items[chosen_index])
            items.pop(chosen_index)
        folders.append(folder)
    folders.append(items)

    return folders


if __name__ == '__main__':
    def validation(item):
        if item == 'x' or item == 'positive':
            return 1
        if item == 'o' or item == 'negative':
            return -1
        if item == 'b':
            return 0
        raise ValueError


    with open('tic-tac-toe.data', newline='') as csv_file:
        reader = csv.reader(csv_file)
        data_file = [[np.array([validation(item) for item in row[:-1]]), validation(row[-1])] for row in reader]

    data_file = split_folders(data_file.copy())
    training = data_file[0] + data_file[1] + data_file[2] + data_file[3]
    model_x = AdaboostModel(100)
    model_x.fit(training)

    features, labels = list(), list()
    for item in data_file[4]:
        features.append(item[0])
        labels.append(item[1])
    labels = np.array(labels)
    w = model_x.predict(features)
    l = w == labels

