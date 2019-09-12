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

            "features_relevant": {
                'Anzahl der Kavitäten': ("c", (1, 2, 3)),
                'Form der Kavitäten': (
                    "c", ('A31C', 'A32B', 'A34', 'B42', 'B3', 'C')),

                'Material': ("c", ('PU', 'PE', 'PVC', 'PUT')),
                'Entformungskonzept': ("c", ('A', 'B')),
                'Schieberanzahl': ("c", (1, 2, 3, 4, 5, 6, 7)),
                'Kanaltyp': ("c", ('Kaltkanal', 'Heißkanal')),

                'Größe der Kavitäten': ("n", (1, 7)),
                'Abmaße Werkzeug': ("n", (4, 31)),
                'x': ("n", (10, 40)),
                'y': ("n", (10, 40)),
                'z': ("n", (10, 40)),
                'Temp': ("n", (170, 250)),
                'Time': ("n", (50, 17000)),
            },

            "features_noise": {
                'noise': ("n", (10, 40)),
            },

            "target": ('Kosten', (100, 2000))
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

        # first reset the database and picture subfolder
        # reset()

        samples = config["samples"]

        features_relevant = config["features_relevant"]

        features_noise = config["features_noise"]

        target = config["target"]

        X, y = make_regression(n_samples=samples,
                               n_features=len(features_relevant),
                               n_informative=len(features_relevant),
                               noise=0.2)

        y = y.reshape(-1, 1)

        if len(features_noise):
            X_noise = np.random.rand(X.shape[0], len(features_noise))

            X = np.concatenate((X, X_noise), axis=1)

        X = MinMaxScaler().fit_transform(X)
        y = MinMaxScaler().fit_transform(y)

        # configure features

        feature_names = list(features_relevant.keys()) + list(
            features_noise.keys())

        df_X = pd.DataFrame(X, columns=feature_names)

        for feature in feature_names:
            if feature in features_relevant:
                feature_data = features_relevant[feature]
            elif feature in features_noise:
                feature_data = features_noise[feature]
            else:
                raise ValueError(f"Feature {feature} not found")

            feature_type, feature_config = feature_data

            if feature_type == "n":
                if len(feature_config) != 2:
                    raise ValueError(
                        f"Numerical features must have config of length 2 [{feature}]")

                df_X[feature] *= feature_config[1] - feature_config[0]
                df_X[feature] += feature_config[0]

            elif feature_type == "c":

                amount_categories = len(feature_config)

                df_X[feature] = pd.cut(df_X[feature], amount_categories)

                df_X[feature].cat.categories = feature_config

            else:
                raise ValueError(
                    f"Type {feature_type} for Feature {feature} unknown")

        # configure target

        target_name, target_config = target

        df_y = pd.DataFrame(y, columns=[target_name])

        df_y *= target_config[1] - target_config[0]
        df_y += target_config[0]

        # Introduce NaN

        nan_mask = np.random.random(df_X.shape) < .1

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
