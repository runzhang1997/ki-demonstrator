from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.tree_export import tree_to_json
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def rules(clf, features, labels, node_index=0):
    node = {}
    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
        # count_labels = zip(clf.tree_.value[node_index, 0], labels)
        # node['name'] = ', '.join(('{} of {}'.format(int(count), label)
        #                          for count, label in count_labels))
        node['type'] = 'leaf'
        node['value'] = clf.tree_.value[node_index, 0].tolist()
        node['error'] = np.float64(clf.tree_.impurity[node_index]).item()
        node['samples'] = clf.tree_.n_node_samples[node_index]
    else:
        feature = features[clf.tree_.feature[node_index]]
        threshold = clf.tree_.threshold[node_index]
        node['type'] = 'split'
        node['label'] = '{} > {}'.format(feature, threshold)
        node['error'] = np.float64(clf.tree_.impurity[node_index]).item()
        node['samples'] = clf.tree_.n_node_samples[node_index]
        node['value'] = clf.tree_.value[node_index, 0].tolist()
        left_index = clf.tree_.children_left[node_index]
        right_index = clf.tree_.children_right[node_index]
        node['children'] = [rules(clf, features, labels, right_index),
                            rules(clf, features, labels, left_index)]

    return node

class Backend(object):
    raw_data = None
    preprocessed_data_nan = None
    preprocessed_data_nan_onehot = None
    model = None

    def __init__(self, config=None):

        if config is None:
            config = self.get_default_config()

        self.generate_data(config)
        self.generate_model(0.8, 100)

    def get_default_config(self):
        config = {

            "samples": 1000,

            "features": {
                'Anzahl der Kavitäten': {"values": (8, 16, 32), "factor": 1, "function":  lambda x: (x - 0.1 * x ** 2) / 0.9},
                'Form der Kavitäten': {"values": ('A', 'B', 'C', 'D'), "factor": 1, "function": lambda x: (np.exp(x) - 1) / (np.exp(1) - 1)},
                'Schieberanzahl': {"values": (1, 2, 3, 4, 5, 6, 7), "factor": 1, "function": lambda x: x},
                'Kanaltyp': {"values": ('Kaltkanal', 'Heißkanal'), "factor": 1, "function": lambda x: x},
            },

            "target": {"name": 'Kosten', "values": (30_000, 75_000)}
        }

        return config

    def get_data(self, mode=0):

        if mode == 0:
            return self.raw_data
        if mode == 1:
            return self.preprocessed_data_nan
        if mode == 2:
            return self.preprocessed_data_nan_onehot
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def generate_data(self, config):
        """
        This method creates the regression dataset with (effective rank) number of singular vectors.
        Some values in some columns will be set to NaN.
        """

        samples_amnt = config["samples"]

        features = config["features"]

        target = config["target"]

        df_X = pd.DataFrame(np.random.random((samples_amnt, len(features))), columns=[f for f in features])
        df_y = pd.DataFrame(np.zeros((samples_amnt)), columns=[target["name"]])

        # calculate possible min and max values for scaling target to realistic values
        y_min = 0
        y_max = 0

        print(df_y.head())

        for feature_name in df_X.columns:

            feature = features[feature_name]

            y_min += feature["factor"] * feature["function"](0)
            y_max += feature["factor"] * feature["function"](1)

            # add contribution of feature to price
            df_y["Kosten"] += feature["factor"] * feature["function"](df_X[feature_name])

            # replace feature values with categories
            amount_categories = len(feature["values"])

            bins = np.linspace(0, 1, amount_categories + 1)

            df_X[feature_name] = pd.cut(df_X[feature_name], bins)

            df_X[feature_name].cat.categories = feature["values"]

        df_y -= y_min
        df_y /= y_max - y_min

        df_y *= target["values"][1] - target["values"][0]
        df_y += target["values"][0]

        # Introduce NaN
        nan_mask = np.random.random(df_X.shape) < .01

        # replace NaN with "" for easier visualization
        df_X = df_X.mask(nan_mask, other="")
        self.raw_data = df_X, df_y

        # drop NaN as first preprocessing step
        df_X_nan = df_X.mask(nan_mask).dropna()
        df_y_nan = df_y.loc[df_X_nan.index]

        self.preprocessed_data_nan = df_X_nan, df_y_nan

        # one-hot encoding of categorical features

        df_X_nan_onehot = df_X_nan
        df_y_nan_onehot = df_y_nan

        column_names = []

        for col in df_X_nan_onehot.columns:

            try:
                df_X_nan_onehot[col].astype(np.float)

                column_names.append(col)

            except ValueError:

                df_onehot = pd.get_dummies(df_X_nan_onehot[col], prefix=col)

                df_X_nan_onehot = df_X_nan_onehot.join(df_onehot)

                for c in df_onehot.columns:
                    column_names.append(c)

        df_X_nan_onehot = df_X_nan_onehot[column_names]

        self.preprocessed_data_nan_onehot = df_X_nan_onehot, df_y_nan_onehot

        self.feature_names = df_X_nan_onehot.columns

    def generate_model(self, train_size, max_depth):
        self.model = DecisionTreeRegressor(max_depth=max_depth)

        df_X, df_y = self.get_data(2)

        X_train, X_test, y_train, y_test = train_test_split(df_X, df_y,
                                                            train_size=train_size)

        self.model.fit(X_train, y_train)

        score = mean_absolute_error(y_test, self.model.predict(X_test))

        return rules(self.model, self.feature_names, None), score

    def highlight_path(self, model_json, path_ids):

        model_json["path_node"] = model_json["node_id"] in path_ids

        del model_json["node_id"]

        if model_json["path_node"]:
            print(model_json["label"])

        if "children" in model_json:
            model_json["children"] = [self.highlight_path(child, path_ids) for child
                                      in model_json["children"]]

        return model_json

    def evaluate_model(self, feature_dict):

        X = []

        for feature in self.feature_names:
            if feature in feature_dict:
                X.append(feature_dict[feature])
            else:
                X.append(0)

        X = np.array(X).reshape(1, -1)

        prediction = self.model.predict(X)[0]

        model_json = rules(self.model, self.feature_names, None)

        decision_path = self.model.decision_path(X)

        path_ids = decision_path.indices

        #model_json = self.highlight_path(model_json, path_ids)

        return prediction, model_json


if __name__ == "__main__":
    backend = Backend()

    backend.generate_model(10, 1, 5)

    backend.get_data()

    X = {
        'Anzahl der Kavitäten': 4,
        'Form der Kavitäten_A31C': 1,
        'Kanaltyp_Kaltkanal': 0,
        'x': 1000,
        'y': 1000,
        'z': 1000,
        'Time': 100000,
    }

    prediction, model_json, decision_path = backend.evaluate_model(X)

    print(prediction)
    print(model_json)
