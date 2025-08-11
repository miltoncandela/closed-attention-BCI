
import os
import pandas as pd
import numpy as np
from brainflow.data_filter import DataFilter

from collections import Counter

# EEG Analysis #
from math import floor
from datetime import datetime
from sklearn.impute import KNNImputer

# Comb Features #
import copy
from scipy import stats
from scipy.stats.mstats import winsorize

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Create Model #
from pickle import dump
from statistics import mean
import matplotlib.pyplot as plt
# from collections import Counter
import matplotlib.patches as mpatches
# from scipy.stats.stats import pearsonr
from matplotlib.offsetbox import AnchoredText
from sklearn.linear_model import LinearRegression
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, classification_report
from datetime import datetime, timedelta


def eeg_analysis():
    # Task: From 2-Second windows to 1-Second windows
    spectral_signals = {'Alpha': [8, 12], 'Beta': [12, 30],
                        'Gamma': [30, 50], 'Theta': [4, 8], 'Delta': [1, 4]}

    # EEG
    sr = 256  # Sampling frequency
    ft = 0  # Filter type (0: Butter, 1: Chev, 2: Bessel)
    eta = 4  # Order number of filter
    det = 2  # De-trend operation (0: None, 1: Constant, 2: Linear)
    n_channels = 4  # Number of channels in EEG
    nfft = DataFilter.get_nearest_power_of_two(sr)
    wf = 3  # Windows function (0: No Window, 1: Hanning, 2: Hamming, 3: Blackman Harris)
    o = 6  # 1/2 per minute recording
    channel_to_position = {'A2': ['Right Cushion'], 'A1': ['Left Cushion'], 'C4': ['Top Right'], 'C3': ['Top Left']}
    columns_signals = [s + '_' + str(c) for c in channel_to_position.keys() for s in spectral_signals.keys()]

    df_total = pd.DataFrame()

    for i, folder in enumerate(os.listdir('UMateria/Files/Data2')):
        df_raw = (pd.read_csv('UMateria/Files/Data2/{}/Raw/Crudas.csv'.format(folder), index_col=0)
                  .apply(pd.to_numeric, errors='coerce').dropna(axis=0).reset_index(drop=True))
        df_eeg = pd.DataFrame(columns=columns_signals)

        # Assuming 256 values per second, then
        for n in range(df_raw.shape[0] // 256):
            l_signals = []
            reference = np.sum(np.array(df_raw.iloc[(n * 256):((n + 1) * 256), [0, 1]]), axis=1)/2

            for n_channel in range(n_channels):

                df_unpross = np.array(df_raw.iloc[(n * 256):((n + 1) * 256), n_channel])
                df_unpross = df_unpross - reference

                # df_unpross = np.delete(df_unpross, [0, 2], axis=1)

                # 60 Hz Notch filter
                DataFilter.perform_bandstop(data=df_unpross, sampling_rate=sr, start_freq=58,
                                            stop_freq=62, order=eta, filter_type=ft, ripple=0)

                # 1–50 Hz 4th order Butterworth bandpass filter
                DataFilter.perform_lowpass(data=df_unpross, sampling_rate=sr, cutoff=50,
                                           order=eta, filter_type=ft, ripple=0)
                DataFilter.perform_highpass(data=df_unpross, sampling_rate=sr, cutoff=1,
                                            order=eta, filter_type=ft, ripple=0)

                # Linear de-trend
                DataFilter.detrend(df_unpross, detrend_operation=det)

                psd_data = DataFilter.get_psd_welch(df_unpross, nfft=nfft, overlap=nfft // 2,
                                                    sampling_rate=sr, window=wf)

                for spectral_signal in spectral_signals.keys():
                    l_signals.append(DataFilter.get_band_power(psd_data, spectral_signals[spectral_signal][0],
                                                               spectral_signals[spectral_signal][1]))
            data_signals = pd.DataFrame(dict(zip(columns_signals, l_signals)), index=[n], columns=columns_signals)
            df_eeg = pd.concat([df_eeg, data_signals])

        n = df_eeg.shape[0]
        df_eeg = df_eeg.loc[:, [x for x in df_eeg.columns if x.split('_')[1][0] != 'A']]
        df_eeg['Subject'], df_eeg['Repetition'], df_eeg['Second'] = folder[1:3], folder[4:6], range(1, n + 1)
        df_eeg = df_eeg.iloc[30:(df_eeg.shape[0] - 30),].reset_index(drop=True)
        df_total = pd.concat([df_total, df_eeg], ignore_index=True)

    return df_total


def comb_features():
    """
    The following function pre-process the DataFrame of EEG and ECG features with a set of three functions that:
    Combines features, remove outliers and normalizes data, according to variables defined in main as global variables.

    :return pd.DataFrame: Transformed DataFrame, ready to be fitted by a Machine Learning model and so make predictions.
    """

    def combined_features(df, information_features_combined):
        """
        The following functions takes a DataFrame with both target and source variables, the name of the target and
        information features is included in the list "information_features_combined", because they must be removed
        before doing the combinations between the features available, and so a DataFrame with the original features,
        as well as additional combined features, is returned.

        :param pd.DataFrame df: DataFrame with both source and target variables.
        :param list information_features_combined: List of information and target variables.
        :return pd.DataFrame: A DataFrame with combined features,
        """

        # A separate DataFrame is created to save the information features such as "Second", "ID".
        df_information_combined = df[information_features_combined]
        df = df.drop(information_features_combined, axis=1)
        df_combined = copy.deepcopy(df)

        epsilon = 0.000001
        names = list(df_combined.columns)
        combinations = []

        # The following for loop creates a set of combined features based on the source variables that are available.
        # It iterates over all the features on a separate DataFrame, and it applies a function. The result is further
        # saved on a column with the following encoding:

        # Name_i-I : Inverse on ith feature
        # Name_i-L : Logarithm on ith feature
        # Name_i-M-Name_j : Multiplication of ith feature with feature jth
        # Name_i-D-Name_j : Division of ith feature with feature jth

        # A small number on the form of a epsilon is being used to avoid NANs because some functions are 0 sensitive,
        # such as the the division by 0. Moreover, a separate list "combinations" is used to keep track the combinations
        # of ith and jth features, and so not to generate duplicate features when multiplying ith feature with jth
        # feature and vice versa (as they are the same number).

        for i in range(len(df.columns)):
            # names.append(df.columns[i] + '-I')
            # df_combined = pd.concat((df_combined, np.divide(np.ones(df.shape[0]), df.loc[:, df.columns[i]])),
            #                         axis=1, ignore_index=True)

            # names.append(df.columns[i] + '-L')
            # df_combined = pd.concat((df_combined, pd.Series(np.log(np.abs(np.array(df.loc[:, df.columns[i]]).astype(float)) + 1))),
            #                         axis=1, ignore_index=True)

            for j in range(len(df.columns)):
                if i != j:
                    # current_combination = str(i) + str(j)
                    # if current_combination not in combinations:
                    #     combinations.append(current_combination)
                    #     names.append(df.columns[i] + '-M-' + df.columns[j])
                    #     df_combined = pd.concat((df_combined,
                    #                              np.multiply(df.loc[:, df.columns[i]], df.loc[:, df.columns[j]])),
                    #                             axis=1, ignore_index=True)
                    names.append(df.columns[i] + '-D-' + df.columns[j])
                    df_combined = pd.concat((df_combined,
                                             pd.Series(np.divide(df.loc[:, df.columns[i]],
                                                                 np.array(df.loc[:, df.columns[j]]) + epsilon))),
                                            axis=1, ignore_index=True)

        # The source variables are concatenated with the target variables, infinite numbers are replaced by NANs, and
        # thus any feature with a NAN is removed.
        df_combined.columns = names
        df_combined = df_combined.dropna(axis=0, how='any').replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any')

        # The list of names is updated with the new features that do not generated infinite values.
        names = list(df_combined.columns)

        # Information features are included back into the new DataFrame, and thus the columns' names are updated.
        df_combined = pd.concat((df_combined, df_information_combined), axis=1, ignore_index=True)
        df_combined.columns = names + information_features_combined

        return df_combined

    def remove_outliers(df, method):
        """
        Uses an statistical method to remove outlier rows from the DataFrame df, and filters the valid rows back to a
        new df that would then be returned.

        :param pd.DataFrame df: DataFrame with non-normalized, source variables.
        :param string method: Type of statistical method used.
        :return pd.DataFrame: Filtered DataFrame.
        """

        # The number of initial rows is saved.
        n_pre = df.shape[0]
        second_value = df.Second

        # The calibration DataFrame is obtained using the "Second" column, as a subset of the calibration phase is taken
        # as a base to further filter samples in the P300 test.
        emp_cols = [column for column in df.columns if column[:3] in ['BVP', 'Tem', 'EDA']]
        df_emp = df.loc[:, emp_cols + ['Second']]
        df = df.drop(emp_cols, axis=1)

        df_calibration = (df[(df.Second > second_ranges['scaling'][0]) & (df.Second < second_ranges['scaling'][1])]
                          .drop('Second', axis=1))
        df = df.drop('Second', axis=1)

        # A switch case selects an statistical method to remove rows considered as outliers.
        if method == 'z-score':
            df = df[np.abs(df - df_calibration.mean()) <= (3 * df_calibration.std())].dropna(axis=0, how='any')
        elif method == 'quantile':
            q1 = df_calibration.quantile(q=.25)
            q3 = df_calibration.quantile(q=.75)
            iqr = df_calibration.apply(stats.iqr)
            df = df[~((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).any(axis=1)]
        elif method == 'winsor':
            def using_mstats(s):
                return winsorize(s, limits=[0.01, 0.01])
            df = df.apply(using_mstats, axis=0)

        # The "Second" column is set back to the original DataFrame, according to the removed rows.
        df['Second'] = second_value[df.index]
        df = pd.merge(df, df_emp, on='Second')

        # The difference between the processed and raw rows is printed.
        n_pos = df.shape[0]
        diff = n_pre - n_pos
        print(f'{diff} rows removed {round(diff / n_pre * 100, 2)}%', end=', ')

        return df.reset_index(drop=True)

    def normalize(df, method):
        """
        Uses a normalization method to

        :param pd.DataFrame df: Filtered DataFrame according to the "remove_outliers" method used.
        :param string method: Type of normalization method used.
        :return pd.DataFrame: Normalized DataFrame according to the normalization method selected.
        """

        second_value = df.Second

        # A calibration DataFrame is generated, on which parameters would be created to then be applied onto the whole
        # DataFrame, the ranges of second are usually on the Eyes Open (EO) phase when using the P300 test.
        df_calibration = (df[(df.Second > second_ranges['normalize'][0]) & (df.Second < second_ranges['normalize'][1])]
                          .drop('Second', axis=1))
        df = df.drop('Second', axis=1)
        column_names = df.columns

        # A switch case selects a normalization method to be used on the whole DataFrame.
        scaler = StandardScaler().fit(df_calibration) if method == 'standard' else MinMaxScaler().fit(df_calibration)
        df = pd.DataFrame(scaler.transform(df), columns=column_names)

        # The second column is set back to the original, normalized DataFrame.
        df['Second'] = second_value

        # Furthermore, the DataFrame is subset on only containing samples that correspond to outside the calibration
        # phase, as this is the last function applied to the DataFrame, and thus the DataFrame would be fitted into a
        # Machine Learning regression model.
        df = df[df.Second > second_ranges['normalize'][1]]

        print('Number of valid rows', df.shape[0])

        return df

    # DataFrame with base features is gathered using the "eeg_analysis" function.
    df_feat = eeg_analysis()
    df_feat['Second'] = pd.to_numeric(df_feat.Second)
    print(df_feat)

    # Information features are defined, these consist on columns for indexing such as "Second", and the target variable.
    information_features = ['Second', 'Subject', 'Repetition']

    print(len(df_feat.Subject.unique()), df_feat.Subject.unique())

    # A separate, empty DataFrame is created, as outliers removal, normalization and feature combination is done for
    # each subject, and thus a for loop needs to be used to iterate over all subjects available.
    df_total = pd.DataFrame()

    # The following for loop iterates over all available subjects, and applies the previously defined functions
    # (outlier removal, normalization, feature combination), the resulted DataFrame is then concatenated into a final
    # DataFrame which contains the processed data for each subject.
    for subject in df_feat.Subject.unique():
        print('Current Subject', subject, end=', ')

        # The df_filtered is created, which has the data from the current "subject".
        df_filtered = df_feat[df_feat.Subject == subject].reset_index(drop=True)

        # A separate DataFrame with information features is kept away, as these features are relevant, but they have
        # nothing to do with normalization methods or outlier removal methods. It is worth noting that the "Second"
        # feature is kept, as it is an important indexer in order to define the limits during the calibration phase.
        df_information = df_filtered[information_features[1:]]
        df_filtered = df_filtered.drop(information_features[1:], axis=1)

        # The three functions are applied to the filtered DataFrame, the resulting DataFrame is then saved in a separate
        # DataFrame called "df_processed".
        df_processed = normalize(df_filtered, scaler_method)
        # df_processed = normalize(combined_features(df_filtered.reset_index(drop=True), ['Second']), scaler_method)

        # Information features are manually set to the processed DataFrame, these option was chosen because some of the
        # rows are removed, and so a concatenation between DataFrames would cause an error.
        df_processed['Subject'] = df_information['Subject']
        df_processed['Repetition'] = df_information['Repetition']

        # The processed DataFrame is then concatenated with the empty DataFrame, which would have all subjects' data.
        df_total = pd.concat([df_total, df_processed], axis=0, ignore_index=True)

    df_total['Group'] = df_total.Subject.map(lambda x: 'Control' if int(x) < 11 else 'Index' if int(x) > 29 else 'Model')
    df_total.to_csv('UMateria/{}.csv'.format(file_name), index=False)
    exit()
    print(len(df_total.Subject.unique()), df_total.Subject.unique())

    return df_total


def create_model(tuned=True, mean_plot=False, n_features=10):
    """
    The following function created a predictive model using the processed data obtained from the "comb_features"
    function, as outliers are removed, combined features are created, and data is normalized.

    :param bool tuned: Whether the models would be created with tuned parameters or not (n_features and some subjects
    that are causing noise to the model due to outliers that were not entirely detected.
    :param bool mean_plot: Create mean plot.
    :param int n_features: Default number of features, this could be changed, but preferably < n_subjects.
    :return None: The created model is saved in a .pkl file, the file name depends on whether Empatica's features were
    included in the model or not.
    """

    # Second and Repetition columns are removed, as only source features, FAS and Subject ID are required.
    x = comb_features().drop(['Second', 'Repetition'], axis=1)

    # If the "tuned" parameters is set to True, then subjects that are making noise to each model would be removed.
    if tuned:
        subjects_to_be_removed = remove_subjects['empatica'] if empatica else remove_subjects['non-empatica']
        x = x.drop(x[x.Subject.isin(subjects_to_be_removed)].index, axis=0)
    x = x.dropna(axis=1).reset_index(drop=True)

    print(len(x.Subject.unique()), x.Subject.unique())
    print(Counter(
        [1 if (x >= 22) else 0 for x in [x[x.Subject == sub].head(1).FAS.item() for sub in x.Subject.unique()]]))
    print(Counter([1 if (35 > x >= 22) else 0 if x < 22 else 2 for x in
                   [x[x.Subject == sub].head(1).FAS.item() for sub in x.Subject.unique()]]))
    print([x[x.Subject == sub].head(1).FAS.item() for sub in x.Subject.unique()])

    # Evaluacion para MLR Empatica & EEG (9 Feats y [16])
    # 12 0.9 0.8 0.50
    # 11 0.9 0.7 0.58
    # 10 0.9 0.8 0.46
    # 9 0.9 0.8 0.58
    # 8 0.6 0.4 0.09

    # Evaluacion para MLR EEG (10 Feats y [4, 20])
    #   ID       2       3        R
    #           0.79    0.5     -0.38
    # 4         0.69    0.38    -0.05
    # 4 13      0.58    0.25    -0.68
    # 4    20   0.83    0.5     0.072
    #   13      0.62    0.38    -0.027
    #   13 20   0.66    0.42    0.304
    #      20   0.77    0.53    -0.47
    # 4 13 20   0.63    0.27    -0.07

    # 2-Categories
    # 1 9
    # 0 5

    # 3-Categories
    # 2 4
    # 1 5
    # 0 5

    # Fatigue score is popped from the x DataFrame, as these would be the prediction and thus the y.
    y = x.pop('FAS')

    corr_matrix = x.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    x = x.drop([column for column in upper_tri.columns if any(upper_tri[column] > 0.95)], axis=1)

    #      Con              Sin
    # 0.64 0.42 -0.074  0.71 0.5 -0.031

    # If the "tuned" parameters is set to True, then the right number of features is selected for each model.
    if tuned:
        n_features = number_features['empatica'] if empatica else number_features['non-empatica']

    # RandomForestRegressor is used as a feature selection method, and thus the importance of each feature is sorted.
    s = pd.Series(index=x.drop(['Subject'], axis=1).columns,
                  data=RandomForestRegressor(random_state=50).fit(x.drop(['Subject'], axis=1),
                                                                  y).feature_importances_).sort_values(ascending=False)
    print(s.head(n_features))

    # The sorted list is used to obtain only the best "n_features" according to the integer selected.
    x = x.loc[:, list(s.index[:n_features]) + ['Subject']]

    # The following for loop generates two 2D plots, these plots would depend on the number of classes
    # that FAS score would be encoded, the first variables are colors and strings to keep the same format.
    name_plot = 'EEG & Empatica' if empatica else 'EEG'
    name_file = 'EEG_Empatica' if empatica else 'EEG'
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    cat_decoding = {2: ['No Fatigue', 'Substantial Fatigue'],
                    3: ['No Fatigue', 'Moderate Fatigue', 'Extreme Fatigue']}
    for n_cat in [2, 3]:

        fig = plt.figure()

        if mean_plot:
            # A scatter plot is generated usign the top 2 features, according to the s series.
            x['FAS'] = y
            l_fatigue, x_data, y_data, fatigue_enc = [], [], [], []
            for current_subject in x.Subject.unique():
                l_fatigue.append(x[x.Subject == current_subject].reset_index(drop=True).loc[0, 'FAS'])
                x_data.append(
                    x[x.Subject == current_subject].reset_index(drop=True).loc[0, list(s.index[:n_features])[0]])
                y_data.append(
                    x[x.Subject == current_subject].reset_index(drop=True).loc[0, list(s.index[:n_features])[1]])
                if n_cat == 2:
                    fatigue_enc.append(1 if (l_fatigue[-1] >= 22) else 0)
                elif n_cat == 3:
                    fatigue_enc.append(1 if (35 > l_fatigue[-1] >= 22) else 0 if l_fatigue[-1] < 22 else 2)

            df = pd.DataFrame({'X': x_data, 'Y': y_data, 'FAS': l_fatigue, 'FAS_Enc': fatigue_enc})

            for clas in range(n_cat):
                df_sub = df[df.FAS_Enc == clas]
                plt.scatter(df_sub.X, df_sub.Y, label=cat_decoding[n_cat][clas], color=colors[clas])

            plt.legend()
            y = x.pop('FAS')

        else:
            fas_cat = [1 if (fatigue >= 22) else 0 for fatigue in y] if n_cat == 2 else \
                [1 if (35 > fatigue >= 22) else 0 if fatigue < 22 else 2 for fatigue in y]
            fig = plt.figure()

            # A scatter plot is generated usign the top 2 features, according to the s series.
            plt.scatter(x.loc[:, list(s.index[:n_features])[0]], x.loc[:, list(s.index[:n_features])[1]],
                        color=[colors[cat] for cat in fas_cat])

            # A legend is manually generated using patches of colors, according to default matplotlib's colors.
            plt.legend(handles=[mpatches.Patch(color=c) for c in colors[:n_cat]], labels=cat_decoding[n_cat])

        # The axes object is used to set the title, x-label, and y-label.
        ax = fig.gca()
        ax.set_title('Best features visualization on {}-Class fatigue {}'.format(n_cat, name_plot))
        ax.set_xlabel(list(s.index[:n_features])[0])
        ax.set_ylabel(list(s.index[:n_features])[1])

        # Figure is saved in a .pdf file to mantain the best resolution when including in .tex files.
        fig.savefig('Empatica-Project-ALAS-main/Files/{}_2D_plot_{}.pdf'.format(n_cat, name_file),
                    bbox_inches='tight')
        plt.close()

    # Random Forest
    #  N      2          3       R
    # 20    0.643      0.429    0.06
    # 15    0.643      0.5      0.051
    # 14    0.643      0.5      0.051
    # 13    0.714      0.571    0.092
    # 12    0.714      0.571    0.126
    # 11    0.714      0.571    0.140
    # 10    0.714      0.571    0.17
    # 9     0.714      0.571    0.143
    # 8     0.714      0.571    0.104
    # 7     0.643      0.5      0.09
    # 6     0.643      0.5      0.11
    # 5     0.643      0.429    0.25

    # Linear Regression
    #  N      3          2       R
    # 10    0.786      0.5      -0.383

    # previous_features = set(list(x.drop(['Subject'], axis=1).columns))
    # scores_pearson = [np.abs(pearsonr(y, x[feature])[0]) for feature in x.columns[:N_FEATURES]]
    # p_value = [np.abs(pearsonr(y, x[feature])[1]) for feature in x.columns[:N_FEATURES]]
    # df_correlations = pd.DataFrame({'Feature': x.columns[:N_FEATURES], 'Correlation': scores_pearson, 'P': p_value})
    # df_correlations = (df_correlations[df_correlations.P < 0.05].sort_values('Correlation', ascending=False).round(6)
    #                   .reset_index(drop=True))
    #       Con                 Sin
    # 0.714 0.5 -0.031  0.643 0.429 0.06
    # filtered_features = set(list(df_correlations.Feature))
    # stad_reject_features = previous_features.difference(filtered_features)
    # print(df_correlations.head(df_correlations.shape[0]))
    # x = x.drop(list(previous_features.difference(filtered_features)), axis=1)
    # print('{} Features were rejected'.format(len(stad_reject_features)))
    # print('{} Features were accepted'.format(len(filtered_features)))

    print(x)

    # An empty DataFrame of results is generated, for continuous predictions, as well as 2-class, 3-class categorical.
    df_results = pd.DataFrame(
        columns=['Cont_Pred', 'Cont_True', 'Cat_Pred_2', 'Cat_True_2', 'Cat_Pred_3', 'Cat_True_3'])

    # The following for loop iterates over all subjects, in order to implement a Leave-One-Out (LOO) validation scheme,
    # using approximately 10 subjects, the LOO validation consist on using 90% of the data for training and 10% of the
    # data for testing, as the current subjects' data would be used to test the model, while the rest of the subjects'
    # data would be used for model training.
    for current_subject in x.Subject.unique():
        # Train index are the rows that does not correspond to the current subject's ID, while the test index are the
        # rows that contain the current subject's ID.
        train_index = x[x.Subject != current_subject].index
        test_index = x[x.Subject == current_subject].index

        # Indexes are used to create the set of training and testing data, removing subject as a feature.
        x_train, x_test = x.iloc[train_index, :].drop('Subject', axis=1), x.iloc[test_index, :].drop('Subject', axis=1)
        y_train, y_test = y[train_index], y[test_index]

        # Multiple linear regression is used, and training data is used to train the model.
        model = RandomForestRegressor().fit(x_train, y_train)

        # A raw prediction is done using the training model and the testing rows.
        raw_prediction = model.predict(x_test)

        # The raw prediction is transformed to a final number, because remember that FAS score is assigned to all
        # samples from a subject, and thus we are predicting the FAS score for each row assigned to the subject. So,
        # in order to compare both scores, a single score must be generated, in this case the mean of all predicted
        # scores was used, although, if the mean of score is not in a valid range (0 > x > 50), median is used.
        prediction = np.median(np.round(raw_prediction)) if (0 > mean(raw_prediction) > 50) else round(
            mean(raw_prediction))

        # 2-Class and 3-Class Categorical encoding is applied to the prediction.
        prediction_cat_2 = 1 if (prediction >= 22) else 0
        prediction_cat_3 = 1 if (35 > prediction >= 22) else 0 if prediction < 22 else 2

        # Subject's true FAS score is recovered from the "y_true" pandas series.
        y_true = y_test.head(1).item()

        # 2-Class and 3-Class Categorical encoding is applied to the true value.
        y_true_cat_2 = 1 if (y_true >= 22) else 0
        y_true_cat_3 = 1 if (35 > y_true >= 22) else 0 if y_true < 22 else 2

        # Results are appended into the "df_results" using a dictionary and the zip function.
        df_results = pd.concat([df_results, pd.DataFrame(dict(zip(list(df_results.columns),
                                                                  [prediction, y_true,
                                                                   prediction_cat_2, y_true_cat_2,
                                                                   prediction_cat_3, y_true_cat_3])),
                                                         index=range(1))], axis=0)
        print(current_subject, prediction, y_true, prediction_cat_2, y_true_cat_2, prediction_cat_3, y_true_cat_3)

    # Each subject's ID is set as a new column in the "df_results" DataFrame.
    df_results['Subject'] = x.Subject.unique()
    print(df_results.head(df_results.shape[0]))

    # The DataFrame's columns are transformed into categorical columns, with their respective order.
    df_results.Cat_True_2 = pd.Categorical(df_results.Cat_True_2, categories=[0, 1], ordered=True)
    df_results.Cat_Pred_2 = pd.Categorical(df_results.Cat_Pred_2, categories=[0, 1], ordered=True)
    df_results.Cat_True_3 = pd.Categorical(df_results.Cat_True_3, categories=[0, 1, 2], ordered=True)
    df_results.Cat_Pred_3 = pd.Categorical(df_results.Cat_Pred_3, categories=[0, 1, 2], ordered=True)

    print(accuracy_score(df_results.Cat_True_2, df_results.Cat_Pred_2))
    print(accuracy_score(df_results.Cat_True_3, df_results.Cat_Pred_3))
    print(r2_score(df_results.Cont_True, df_results.Cont_Pred))

    print(classification_report(df_results.Cat_True_2, df_results.Cat_Pred_2))
    print(classification_report(df_results.Cat_True_3, df_results.Cat_Pred_3))

    # A simple plot is generated, that related both continuous FAS predictions and true values.
    ax = df_results.loc[:, ['Cont_Pred', 'Cont_True']].sort_values('Cont_True', ascending=True).reset_index(
        drop=True).plot()

    # Moreover, a title is set, as well as the R-squared obtained using the LOO validation.
    ax.set_title('Predicted and true FAS values, using {} features'.format(name_plot))
    at = AnchoredText("Coefficient of determination $(R^2)$: {}".format(
        round(r2_score(df_results.Cont_True, df_results.Cont_Pred), 2)), loc='lower right')
    ax.add_artist(at)

    # The figure is obtained and thus saved as .pdf file.
    fig = ax.get_figure()
    plt.legend()

    fig.savefig('Empatica-Project-ALAS-main/Files/predictions_{}.pdf'.format(name_file), bbox_inches='tight')

    # A final Linear Regression model is fitted with all the data available, and thus exported as a .pkl file.
    model = LinearRegression().fit(x.drop('Subject', axis=1), y)

    l_features = list(x.drop('Subject', axis=1).columns)

    file = open("Empatica-Project-ALAS-main/Files/features_{}.txt".format(name_file), "w")
    for i in range(len(l_features)):
        if i == len(l_features) - 1:
            file.writelines(l_features[i])
        else:
            file.writelines(l_features[i] + ', ')
    file.close()

    dump(model, open('Empatica-Project-ALAS-main/Files/model_{}.pkl'.format(name_file), 'wb'))


# file_name = 'EEG_ECG_Z39S69N100'
# file_name = 'EEG_Z39S69N100'

empatica = False
norm_method = 'z-score'  # quantile, z-score, winsor
scaler_method = 'standard'  # minmax, standard
second_ranges = {'normalize': [30, 90],
                 'scaling': [30, 90]}

ecg_name = '_ECG' if empatica == True else ''
norm_name = 'Z' if norm_method == 'z-score' else 'Q'
scal_name = 'S' if scaler_method == 'standard' else 'M'
n_valid = 100
file_name = 'EEG{}_{}{}{}{}{}{}N{}_R2'.format(ecg_name, norm_name,
                                           str(second_ranges['scaling'][0])[0], str(second_ranges['scaling'][1])[0],
                                           scal_name,
                                           str(second_ranges['normalize'][0])[0], str(second_ranges['normalize'][1])[0],
                                           n_valid)

# Parameters used when tuned is set to True
remove_subjects = {'empatica': [16],
                   'non-empatica': [4, 20]}
number_features = {'empatica': 10,
                   'non-empatica': 10}
create_model(tuned=False, mean_plot=False, n_features=2)