# This will look at the themens data and classify whether or not the ARTIST autoscaled values are good or bad
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import julian
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
import scipy.stats
import sklearn.model_selection
#from sklearn.experimental import enable_halving_search_cv
#from sklearn.model_selection import HalvingRandomSearchCV
import pickle
import utils
import copy
import pandas as pd
# Try genetic algorithm tuning https://towardsdatascience.com/tune-your-scikit-learn-model-using-evolutionary-algorithms-30538248ac16


# This function will take an DataFrame built from reading in parsed ionogram images and add Themens data that sometimes
# contain data during times not in the parsed ionogram data
def adjust_iono_data(iono_data, themen_timestamps):
    iono_time_index = 0
    new_table_index = 0
    new_table = copy.deepcopy(iono_data)
    for themen_time in themen_timestamps:
        # Add a new blank row if conditions are met
        # 1) We are past the last iono_time_index and there are more themen data
        if iono_time_index >= len(iono_data) or ((
                themen_time.replace(tzinfo=datetime.timezone.utc) < datetime.datetime.fromtimestamp(
                iono_data['datetime'][iono_time_index].timestamp(), tz=datetime.timezone.utc)) and (
                abs(themen_time.replace(tzinfo=datetime.timezone.utc) - datetime.datetime.fromtimestamp(
                    iono_data['datetime'][iono_time_index].timestamp(), tz=datetime.timezone.utc)).seconds > 10)):
            # Add a new row filled with nans at the correct time
            new_row = pd.DataFrame(columns=iono_data.columns).fillna(value=np.nan).append(
                {'station_id': np.unique(iono_data['station_id'])[0], 'datetime': themen_time}, ignore_index=True)
            if new_table_index >= len(new_table):
                new_table = pd.concat([new_table, new_row]).reset_index(drop=True)
            else:
                new_table = pd.concat(
                    [new_table.iloc[0:new_table_index], new_row, new_table.iloc[new_table_index:]]).reset_index(
                    drop=True)
        elif iono_time_index < len(iono_data):
            iono_time_index += 1
        new_table_index += 1
    return new_table


main_dir = 'themens_data_with_hmf'
all_files = os.listdir(main_dir)
threshold_mhz = 0.5
# number of features to use. should be an odd number
# num_features = 9
# window_size = 9
# extra_features = 13
window_hours = 2

all_features = None
all_categories = np.zeros(0, dtype='bool')
all_artist_fof2s = np.array([])
all_artist_hmf2s = np.array([])
all_confidence_scores = np.array([])
all_themens_fof2s = np.array([])
all_themens_hmf2s = np.array([])
all_isprocessed = np.array([])
for file in all_files:
    try:
        print('Loading ' + os.path.join(main_dir, file))
        h5data = h5py.File(os.path.join(main_dir, file), 'r')
        fof2_artist = h5data['ARTIST_foF2'][:]
        hmf2_artist = h5data['ARTIST_hmF2'][:]
        fof2_manual = h5data['Manual_foF2'][:]
        hmf2_manual = h5data['Manual_hmF2'][:]
        if len(fof2_artist) <= 10:
            continue
        timestamps = [julian.from_jd(i) for i in h5data['Julian_dates'][:]]
        # Round the timestamps to the nearest second
        timestamps_rounded = pd.to_datetime(timestamps).round('1s')
        dates = np.unique([timestamp.date() for timestamp in timestamps])
        station_id = file.split('.')[0]
        # Download the data
        params_to_download = ['hmF2', 'hF']
        iono_params = pd.DataFrame()
        for date in dates:
            start_datetime = datetime.datetime.combine(date, datetime.datetime.min.time())
            end_datetime = datetime.datetime.combine(date, datetime.datetime.max.time())
            iono_params = iono_params.append(utils.get_iono_params(station_id, params_to_download, start_datetime,
                                                                   end_datetime), ignore_index=True)
        # Make sure iono_params['datetime'] values are equal to timestamps, if not then fill in or ignore values
        iono_params = adjust_iono_data(iono_params, timestamps)

        #fof2_filtered = utils.filter_data_highbias(fof2_artist, window_size=window_size)
        file_features = []
        file_categories = []
        for i in range(len(fof2_artist)):
            # Save value in all_* arrays
            all_artist_fof2s = np.append(all_artist_fof2s, fof2_artist[i])
            all_artist_hmf2s = np.append(all_artist_hmf2s, hmf2_artist[i])
            all_confidence_scores = np.append(all_confidence_scores, iono_params['cs'].values[i])
            all_themens_fof2s = np.append(all_themens_fof2s, fof2_manual[i])
            all_themens_hmf2s = np.append(all_themens_hmf2s, hmf2_manual[i])

            fof2_a = fof2_artist[i]
            fof2_m = fof2_manual[i]
            hmf2_a = hmf2_artist[i]
            dt = timestamps[i]
            # Calulate the start and end index based on time window size
            start_index = np.min(
                np.where(timestamps_rounded >= timestamps_rounded[i] - datetime.timedelta(hours=window_hours/2))[0])
            mid_index = i
            end_index = np.max(
                np.where(timestamps_rounded <= timestamps_rounded[i] + datetime.timedelta(hours=window_hours / 2))[0])
            # Make sure there are at least 2 points on either side of the sample point
            if mid_index <= start_index + 1 or mid_index >= end_index - 2:
                #all_isprocessed = np.append(all_isprocessed, False)
                all_isprocessed = np.append(all_isprocessed, 0)
                continue
            # Build the feature vector if there are sufficient data around the sample
            if start_index >= 0 and end_index < len(fof2_artist) and (
                    dt - timestamps[start_index]) < datetime.timedelta(hours=2) and (
                    timestamps[end_index] - dt) < datetime.timedelta(hours=2): # \
                    # and \
                    # np.all(np.isfinite(fof2_artist[start_index:end_index+1])) and \
                    # np.all(np.isfinite(fof2_manual[start_index:end_index+1])) and \
                    # np.all(np.isfinite(hmf2_artist[start_index:end_index+1])): #and np.all(np.isfinite(fof2_filtered[start_index:end_index+1])):
                print(fof2_a, fof2_m, dt)
                # features = utils.make_feature_vector(h5data['Julian_dates'][start_index:end_index + 1],
                #                                      fof2_artist[start_index:end_index+1],
                #                                      fof2_filtered=fof2_filtered[start_index:end_index+1],
                #                                      slope=True,
                #                                      hmf_artist=hmf2_artist[start_index:end_index+1],
                #                                      confidence_scores=iono_params['cs'].values[start_index:end_index+1],
                #                                      window_stats=True, mid_index=mid_index-start_index)
                features = utils.make_feature_vector(h5data['Julian_dates'][start_index:end_index + 1],
                                                     fof2_artist[start_index:end_index+1],
                                                     slope=True,
                                                     hmf_artist=hmf2_artist[start_index:end_index+1],
                                                     confidence_scores=iono_params['cs'].values[start_index:end_index+1],
                                                     window_stats=True,
                                                     mid_index=mid_index-start_index)
                # Make sure all features are finite
                if np.all(np.isfinite(features)):
                    try:
                        if all_features is None:
                            all_features = features
                        else:
                            all_features = np.vstack([all_features, features])
                    except:
                        print('There is an error in stacking a new feature the the all_features array')
                    all_categories = np.hstack([all_categories, abs(fof2_m-fof2_a) > threshold_mhz])
                    #all_isprocessed = np.append(all_isprocessed, True)
                    all_isprocessed = np.append(all_isprocessed, 1)
                else:
                    print('Not all features are finite')
                    all_isprocessed = np.append(all_isprocessed, -1)
            else:
                #all_isprocessed = np.append(all_isprocessed, False)
                all_isprocessed = np.append(all_isprocessed, -2)
    except Exception as e:
        print(e)

# Make a plot showing fof2 difference vs confidence score
fig, ax = plt.subplots()
ax.plot(all_confidence_scores, all_artist_fof2s-all_themens_fof2s, '.')
ax.grid()
ax.set_title('{} Points, Various Stations'.format(len(all_artist_fof2s) -
                                                  sum(np.isnan(all_artist_fof2s-all_themens_fof2s))),
             fontdict={'fontsize': 18})
ax.set_xlabel('Confidence Score', fontdict={'fontsize': 16})
ax.set_ylabel('ARTIST foF2 Error (MHz)', fontdict={'fontsize': 16})
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(12)

# Make a plot showing fof2 difference and label the classifications
fig, ax = plt.subplots()
ax.plot(all_confidence_scores[all_isprocessed==1][all_categories==False], all_artist_fof2s[all_isprocessed==1][all_categories==False] -
        all_themens_fof2s[all_isprocessed==1][all_categories==False], '.b', label='Good', markersize=1)
ax.plot(all_confidence_scores[all_isprocessed==1][all_categories], all_artist_fof2s[all_isprocessed==1][all_categories] -
        all_themens_fof2s[all_isprocessed==1][all_categories], '.r', label='Bad', markersize=1)
ax.set_xlabel('Confidence Score', fontdict={'fontsize': 16})
ax.set_ylabel('ARTIST foF2 Error (MHz)', fontdict={'fontsize': 16})
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(12)
ax.grid()
ax.set_title('Classifier and Confidence Score Comparison', fontdict={'fontsize': 18})
ax.legend(prop={'size': 12})

# Do some machine learning
scaler = StandardScaler().fit(all_features)
X = scaler.transform(all_features)
X_train, X_test, y_train, y_test = train_test_split(X, all_categories, test_size=0.3, random_state=1)


# Create a class to handle class_weight hyperparamter optimization
class MyClassWeight:
    rvs_func = None
    def __init__(self, lowlim, upperlim):
        self.rvs_func = scipy.stats.randint(lowlim, upperlim+1)
    def rvs(self, random_state=0):
        return {False: 1, True: self.rvs_func.rvs()}


# Find the optimal classifier using HalvingRandomSearchCV
clf = RandomForestClassifier()
param_distributions = {"max_depth": [3, None],
                       "min_samples_split": scipy.stats.randint(2, 21),
                       "min_samples_leaf": scipy.stats.randint(1, 51),
                       "max_features": scipy.stats.randint(1, len(features)-5),
                       "class_weight": MyClassWeight(5, 300),
                       "n_estimators": scipy.stats.randint(4, 400)}
param_grid = {"max_depth": [3, 5, 10, None],
              "min_samples_split": [2, 4, 8, 13, 16, 21],
              "min_samples_leaf": [1, 10, 25, 50],
              "max_features": [1, 3, 5, 10, 20],
              "class_weight": [{False: 1, True: 5}, {False: 1, True: 50}, {False: 1, True: 100}, {False: 1, True: 150},
                               {False: 1, True: 200}, {False: 1, True: 300}],
              "n_estimators": [4, 10, 50, 100, 200, 300]}
betas = np.power(2.0, [-2, -1, 0, 1, 2, 3, 4])
#betas = [0.05, 0.1, 0.5, 1, 2, 5, 8, 10]
confusion_matrices = []
classifiers = []
n_iter = 40
for beta_score in betas:
    print('Computing classifier for beta score: {}'.format(beta_score))
    scorer = metrics.make_scorer(metrics.fbeta_score, beta=beta_score)
    search = sklearn.model_selection.GridSearchCV(estimator=clf, param_grid=param_grid, verbose=10, scoring=scorer, cv=3)
    # search = sklearn.model_selection.RandomizedSearchCV(clf, param_distributions, verbose=10, scoring=scorer, n_jobs=1,
    #                                                     n_iter=n_iter, cv=3)
    search.fit(X, all_categories)
    classifier = search.best_estimator_
    classifier.fit(X_train, y_train)
    classifiers.append(copy.deepcopy(classifier))
    confusion_matrices.append(metrics.confusion_matrix(y_test, classifier.predict(X_test), normalize='true'))
    filename = os.path.join('classifiers',
                            'scaler_and_classifiers_and_betas_{}_{}_{}.sav'.format(len(features), min(betas),
                                                                                   beta_score))
    with open(filename, 'wb') as fid:
        pickle.dump([scaler, classifiers, betas], fid)
fig, ax = plt.subplots()
ax.plot(betas, np.asarray(confusion_matrices)[:, 0, 1], label='False Positive')
ax.plot(betas, np.asarray(confusion_matrices)[:, 1, 1], label='True Positive')
ax.grid('on')
ax.set_xlabel('Beta Score')
ax.set_ylabel('Rate')
ax.legend()

ax = metrics.ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test, normalize='true', cmap=plt.cm.Blues,
                                                   display_labels=['Good', 'Bad']).ax_
ax.set_title('Confusion Matrix')



#search = HalvingRandomSearchCV(clf, param_distributions, verbose=True, scoring=scorer)
#search.fit(X, all_categories)


#
# n_estimators = [50, 100, 200, 300]
# max_features = [4, 8, 10, 15]
# max_depths = [4, 16, None]
# min_samples_leafs = [4, 16, 32]
# weights = [10, 50, 100, 200]
#
# classifiers = []
# for n_estimator in n_estimators:
#     for max_feature in max_features:
#         for max_depth in max_depths:
#             for min_samples_leaf in min_samples_leafs:
#                 for weight in weights:
#                     classifiers.append(RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimator,
#                                                               max_features=max_feature,
#                                                               min_samples_leaf=min_samples_leaf,
#                                                               class_weight={True: weight, False: 1}))
#
# classifiers = [LinearDiscriminantAnalysis(),
#                QuadraticDiscriminantAnalysis(),
#                SVC(kernel="rbf", class_weight={True: 10, False: 1}, verbose=True),
#                SVC(kernel="rbf", class_weight={True: 20, False: 1}, verbose=True),
#                SVC(kernel="rbf", class_weight={True: 30, False: 1}, verbose=True),
#                SVC(kernel="linear", class_weight={True: 10, False: 1}, verbose=True),
#                SVC(kernel="linear", class_weight={True: 20, False: 1}, verbose=True),
#                SVC(kernel="linear", class_weight={True: 30, False: 1}, verbose=True),
#                MLPClassifier(alpha=1, max_iter=100000, learning_rate='adaptive', learning_rate_init=0.001, tol=1e-7,
#                              verbose=True, n_iter_no_change=20, hidden_layer_sizes=(50, 12, 12, 10, 25)),
#                DecisionTreeClassifier(max_depth=5),
#                KNeighborsClassifier(4),
#                RandomForestClassifier(max_depth=5, n_estimators=100, max_features=4),
#                RandomForestClassifier(max_depth=None, n_estimators=100, max_features=4, class_weight={True: 10, False: 1}),
#                RandomForestClassifier(max_depth=None, n_estimators=100, max_features=8, class_weight={True: 10, False: 1}),
#                RandomForestClassifier(max_depth=None, n_estimators=200, max_features=8, class_weight={True: 10, False: 1}),
#                RandomForestClassifier(max_depth=12, n_estimators=300, max_features=10, class_weight={True: 75, False: 1}),
#                RandomForestClassifier(max_depth=13, n_estimators=300, max_features=10, class_weight={True: 75, False: 1}),
#                RandomForestClassifier(max_depth=14, n_estimators=300, max_features=10, class_weight={True: 75, False: 1}),
#                RandomForestClassifier(max_depth=15, n_estimators=300, max_features=10, class_weight={True: 75, False: 1}),
#                RandomForestClassifier(max_depth=14, n_estimators=300, max_features=11, class_weight={True: 75, False: 1}),
#                RandomForestClassifier(max_depth=14, n_estimators=300, max_features=6, class_weight={True: 75, False: 1}),
#                RandomForestClassifier(max_depth=14, n_estimators=300, max_features=7, class_weight={True: 75, False: 1}),
#                RandomForestClassifier(max_depth=14, n_estimators=300, max_features=5, class_weight={True: 75, False: 1}),
#                RandomForestClassifier(max_depth=14, n_estimators=300, max_features=6, class_weight={True: 100, False: 1}),
#                RandomForestClassifier(max_depth=14, n_estimators=300, max_features=6, class_weight={True: 80, False: 1}),
#                RandomForestClassifier(max_depth=4, n_estimators=100, max_features=4, class_weight={True: 20, False: 1}),
#                RandomForestClassifier(max_depth=10, n_estimators=175, max_features=6, class_weight={True: 20, False: 1}),
#                RandomForestClassifier(max_depth=10, n_estimators=175, max_features=6, class_weight={True: 20, False: 1}, criterion='entropy'),
#                RandomForestClassifier(max_depth=6, n_estimators=100, max_features=6, class_weight={True: 30, False: 1}),
#                RandomForestClassifier(max_depth=5, n_estimators=10, max_features=4, class_weight={True: 30, False: 1}),
#                RandomForestClassifier(class_weight={False: 1, True: 100}, max_features=4, min_samples_leaf=16, n_estimators=200),
#                RandomForestClassifier(class_weight={False: 1, True: 50}, max_depth=16, max_features=8, min_samples_leaf=16),
#                RandomForestClassifier(class_weight={False: 1, True: 50}, max_depth=16, max_features=4, min_samples_leaf=16),
#                RandomForestClassifier(class_weight={False: 1, True: 200}, max_depth=4, max_features=4, min_samples_leaf=16, n_estimators=200),
#                AdaBoostClassifier()]
# print("Always False: {}".format(np.sum(y_test == False)/len(y_test)))
# metrics.ConfusionMatrixDisplay.from_predictions(y_test, np.asarray([False for i in y_test]), cmap=plt.cm.Blues)
# plt.title('Always False Confusion Matrix')
# best_score = 0
# best_score_under_thresh = 0
# false_positive_thresh = 0.1
# false_positives = []
# true_positives = []
# for classifier in classifiers:
#     classifier.fit(X_train, y_train)
#     print("{}: {}".format(classifier, classifier.score(X_test, y_test)))
#     confusion_matrix = metrics.confusion_matrix(y_test, classifier.predict(X_test), normalize='true')
#     if confusion_matrix[1, 1] > best_score:
#         best_score = confusion_matrix[1, 1]
#         print("Best Classifier Index {}: {}".format(classifier, best_score))
#     if confusion_matrix[1, 1] > best_score_under_thresh and confusion_matrix[0, 1] < false_positive_thresh:
#         best_score_under_thresh = confusion_matrix[1, 1]
#         print("Best Under Threshold Classifier Index {}: {}".format(classifier, best_score_under_thresh))
#     false_positives.append(confusion_matrix[0, 1])
#     true_positives.append(confusion_matrix[1, 1])

# Plot the data
metrics.ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
plt.title("{} Confusion Matrix".format(classifier))
#
plt.figure()
plt.plot(timestamps, fof2_artist, '.')
plt.plot(timestamps, fof2_manual, '.')
plt.title('Manual vs ARTIST fOF2')
plt.legend(['ARTIST', 'Manual'])
plt.gca().xaxis.set_major_formatter(mdates.ConciseDateFormatter(plt.gca().xaxis.get_major_locator()))
plt.grid()
plt.ylabel('fOF2 (MHz)')
plt.xlabel('Time (UTC)')
