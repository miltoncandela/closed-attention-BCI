import pandas as pd
import numpy as np

# Regressor models
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, r2_score, classification_report, mean_squared_error
from random import seed, sample
from math import sqrt

# Perform 70:30, 80:20, LOO splits
# Get Accuracy, F1-Score, Precision, Recall, R^2, RMSE
# Test with regression models: SVM, MLR, RF, GBM, ANN
# Maybe classification models? LR, KNN, RF, DT, ANN

from statistics import mean, stdev, mode

import warnings
warnings.filterwarnings("ignore")


def get_data(file_name, n_features):
    x_data = pd.read_csv('Paper_Data/no_reference/{}.csv'.format(file_name)).drop(['Second', 'Repetition'], axis=1)

    # only_c3_c4 = [col for col in x_data.columns if (col.count('A1') + col.count('A2')) == 0]
    # x_data = x_data[only_c3_c4]

    # k = 3
    # map_function = lambda x: (1 if (x >= 22) else 0) if k == 2 else (1 if (35 > x >= 22) else 0 if x < 22 else 2)
    y_data = x_data.pop('FAS')
    # y_data_imp = y_data.map(map_function)

    # corr_matrix = x_data.corr().abs()
    # upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # x_data = x_data.drop([column for column in upper_tri.columns if any(upper_tri[column] > 0.9)], axis=1)

    s = pd.Series(index=x_data.drop(['Subject'], axis=1).columns,
                  data=RandomForestRegressor(random_state=50).fit(x_data.drop(['Subject'], axis=1),
                                                                  y_data).feature_importances_).sort_values(ascending=False)

    # The sorted list is used to obtain only the best "n_features" according to the integer selected.
    x_data = x_data.loc[:, list(s.index[:n_features]) + ['Subject']]

    return x_data, y_data


def create_model(model_name, train_x, train_y, k):

    if model_name in ['LOG', 'SVC', 'RFC']:
        map_function = lambda x: (1 if (x >= 22) else 0) if k == 2 else (1 if (35 > x >= 22) else 0 if x < 22 else 2)
        train_y = [map_function(x) for x in train_y]

    if model_name == 'SVR':
        return LinearSVR(random_state=20).fit(train_x, train_y)
    elif model_name == 'MLR':
        return LinearRegression().fit(train_x, train_y)
    elif model_name == 'RFR':
        return RandomForestRegressor(random_state=20).fit(train_x, train_y)
    elif model_name == 'GBM':
        return GradientBoostingRegressor(random_state=20).fit(train_x, train_y)
    elif model_name == 'ANN':
        return MLPRegressor(random_state=20).fit(train_x, train_y)
    elif model_name == 'LOG':
        return LogisticRegression(max_iter=500, random_state=20).fit(train_x, train_y)
    elif model_name == 'RFC':
        return RandomForestClassifier(random_state=20).fit(train_x, train_y)
    elif model_name == 'SVC':
        return LinearSVC(random_state=20).fit(train_x, train_y)


def divide_data(validation_name, x_data, y_data, n_seed):

    ids = x_data.Subject.unique()
    n = len(ids)

    train_ids = ids

    if validation_name == 'LOO':
        test_ids = [ids[n_seed]]
    elif validation_name in ['70:30', '80:20']:
        n_test = int(validation_name[3:])

        seed(n_seed)
        test_ids = sample(set(train_ids), int(n*n_test/100))

    train_ids = list(set(train_ids) - set(test_ids))

    train_index = x[x.Subject.isin(train_ids)].index
    test_index = x[x.Subject.isin(test_ids)].index

    #print(n_seed, train_ids, test_ids)

    x_train_f, x_test_f = x_data.iloc[train_index, :], x_data.iloc[test_index, :]

    def get_index(s):
        """
        :param s: A list with subject ids and their index numbers
        :return: [(index_start, index_end), ...] for n distinct subjects
        """
        l = []
        for distinct_id in s.unique():
            s_sub = s[s == distinct_id]
            l.append((s_sub.index[0], s_sub.index[-1]))
        return l

    train_sub, test_sub = get_index(x_train_f.pop('Subject')), get_index(x_test_f.pop('Subject'))
    # x_train_f, x_test_f = x_data.iloc[train_index, :].drop('Subject', axis=1), x_data.iloc[test_index, :].drop('Subject', axis=1)
    y_train_f, y_test_f = y_data[train_index], y_data[test_index]

    return x_train_f, y_train_f, x_test_f, y_test_f, test_sub


def get_results(model, model_name, dataset_x, dataset_y, k, ids):
    """
    :param ids: [(index_start, index_end), ...] for n distinct subjects
    """

    l_true, l_pred = [], []
    map_function = lambda x: (1 if (x >= 22) else 0) if k == 2 else (1 if (35 > x >= 22) else 0 if x < 22 else 2)

    if model_name in ['SVR', 'MLR', 'RFR', 'GBM', 'ANN']:
        for subject in ids:
            x_sub = dataset_x.loc[list(range(subject[0], subject[1]+1)), :]
            y_sub = dataset_y.loc[subject[0]]
            # y_sub = dataset_y.loc[list(range(subject[0], subject[1]+1))]

            y_pred = mean(model.predict(x_sub))
            # y_pred = model.predict(x_sub)
            # print(y_pred)

            l_true.append(y_sub)
            l_pred.append(y_pred)

        unnest_l = lambda l: [round(x) for y in l for x in y]
        # l_true, l_pred = unnest_l(l_true), unnest_l(l_pred)

        # [x for y in l for x in y]

        # y_true_cat = [map_function(x) for x in l_true]
        # y_pred_cat = [map_function(x) for x in l_pred]

        l_true = [map_function(x) for x in l_true]
        l_pred = [map_function(x) for x in l_pred]

        # df_pred = pd.DataFrame(dict(zip(['N_Pred', 'N_True', 'Cat_Pred', 'Cat_True'], [l_pred, l_true, y_pred_cat, y_true_cat])))

    elif model_name in ['LOG', 'RFC', 'SVC']:
        for subject in ids:
            x_sub = dataset_x.loc[list(range(subject[0], subject[1]+1)), :]
            y_sub = map_function(dataset_y.loc[subject[0]])

            # y_sub = dataset_y.loc[list(range(subject[0], subject[1]+1))]
            y_pred = mode(model.predict(x_sub))

            l_true.append(y_sub)
            l_pred.append(y_pred)

    l_cat = [0, 1] if k == 2 else [0, 1, 2]

    y_true_cat = pd.Categorical(l_true, categories=l_cat, ordered=True)
    y_pred_cat = pd.Categorical(l_pred, categories=l_cat, ordered=True)

    # report = classification_report(y_true_cat, y_pred_cat, output_dict=True)
    # l_performance = ['accuracy', 'f1-score', 'precision', 'recall', 'r2', 'rmse']
    # l_performance = [accuracy_score(y_true_cat, y_pred_cat), report['macro avg']['f1-score'],
    #                  report['macro avg']['precision'], report['macro avg']['recall'],
    #                  r2_score(l_true, l_pred), sqrt(mean_squared_error(l_true, l_pred))]

    l_performance = [accuracy_score(y_true_cat, y_pred_cat)]

    return l_performance


def obtain_accuracy(n_fold, validation_scheme, x, y, model_name, n_class):
    l_acc = []
    for n in range(n_fold):
        seed_transformed = n if validation_scheme == 'LOO' else (n + 1) * int(validation_scheme[:2])
        x_train, y_train, x_test, y_test, test_idx = divide_data(validation_scheme, x, y, seed_transformed)
        model = create_model(model_name, x_train, y_train, n_class)

        l_results_test = get_results(model, model_name, x_test, y_test, n_class, test_idx)
        l_acc.append(l_results_test[0])

    return mean(l_acc), stdev(l_acc)


max_feat = 2
names = ['EEG_Z39M69N100_R']  # ['EEG_Q39M69N100', 'EEG_Q39S69N100', 'EEG_Z39M69N100', 'EEG_Z39S69N100']



exit()

validation_schemes = ['70:30', '80:20']  # ['LOO', '80:20', '70:30']
cross_vals = [10]  # [5, 10]
n_feats = [2]  # range(2, max_feat + 1, 2)
models = ['SVR', 'MLR', 'RFR', 'SVC', 'LOG', 'RFC']

df_results = pd.DataFrame(columns=['CSV', 'N_Fold', 'N_Feat', 'Validation', 'Model', 'N_Class', 'Acc_Test', 'SD_Test'])

for name in names:
    print('{}.csv being read...'.format(name))
    # x_tot, y = get_data(name, max_feat)
    x_tot = pd.read_csv('UMateria/{}.csv'.format(name)).drop(['Second', 'Repetition'], axis=1)
    y = x_tot.pop('RT')
    print('{}.csv file read!'.format(name))
    for cross_val in cross_vals:
        print('Considering {} fold cross-validations...'.format(cross_val))
        for n_feat in n_feats:
            print('Considering {} features...'.format(n_feat))
            x = x_tot.iloc[:, list(range(0, n_feat)) + [x_tot.shape[1] - 1]]
            for validation_scheme in validation_schemes:
                n_fold = len(x.Subject.unique()) if validation_scheme == 'LOO' else cross_val
                print('{} validation scheme with {} folds...'.format(validation_scheme, n_fold))
                for model_name in models:
                    print('Using a {} model...'.format(model_name))
                    for n_class in [2, 3]:
                        mean_acc, sd_acc = obtain_accuracy(n_fold, validation_scheme, x, y, model_name, n_class)

                        data_dict = {'CSV': name, 'N_Fold': cross_val,'N_Feat': n_feat, 'Validation': validation_scheme,
                                     'Model': model_name, 'N_Class': n_class, 'Acc_Test': mean_acc, 'SD_Test': sd_acc}

                        df_results = pd.concat([df_results, pd.DataFrame(columns=list(df_results.columns),
                                                                      data=data_dict,
                                                                      index=[df_results.shape[0]])], axis=0)

                        # print(df_results)

# print(df_results)
df_results.to_csv('Paper_Data/fragile_acc.csv')
