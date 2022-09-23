# This script is for generating images for a slide for Alex C.
import copy
import datetime
import h5py
import julian
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
import utils

# Setup parameters
station_id = 'WP937'
bad_sample_datetime = datetime.datetime(2021, 8, 1, 0, 50, 8)
window_size = 9
window_hours = 2
scaler_and_classifier_filename = os.path.join('classifiers','scaler_and_classifiers_44_0.1_8.sav')

# Download wallops image
iono_params = utils.get_iono_params(station_id, ['foF2', 'hmF2'], datetime.datetime(2021, 8, 1, 0, 0), datetime.datetime(2021, 8, 2, 0, 0))
index_of_bad_sample = np.argmin(abs(pd.to_datetime(iono_params['datetime'].values) - bad_sample_datetime))
timestamps_rounded = pd.to_datetime(iono_params['datetime'].values).round('1s')

# start_index = index_of_bad_sample - int(window_size / 2)
# end_index = index_of_bad_sample + int(window_size / 2) + 1
start_index = np.min(np.where(timestamps_rounded >= timestamps_rounded[index_of_bad_sample] -
                              datetime.timedelta(hours=window_hours / 2))[0])
mid_index = index_of_bad_sample
end_index = np.max(np.where(timestamps_rounded <= timestamps_rounded[index_of_bad_sample] +
                            datetime.timedelta(hours=window_hours / 2))[0])
# Get the features
fof2_artist = iono_params['foF2'].values
# fof2_filtered = utils.filter_data_highbias(fof2_artist, window_size=window_size)

feature_vector = utils.make_feature_vector(
    pd.to_datetime(iono_params['datetime'].values[start_index:end_index+1]).to_julian_date(),
    fof2_artist[start_index:end_index+1],
    slope=True,
    hmf_artist=iono_params['hmF2'].values[start_index:end_index+1],
    confidence_scores=iono_params['cs'].values[start_index:end_index+1],
    window_stats=True,
    mid_index=mid_index-start_index)

# Load the classifier and scaler
with open(scaler_and_classifier_filename, 'rb') as fid:
    [scaler, classifiers] = pickle.load(fid)
# choose a single classifier if there are multiple
if type(classifiers) is list:
    classifier = classifiers[-3]
else:
    classifier = classifiers

classifier.predict(scaler.transform(feature_vector.reshape(1, -1)))


# Loop through all ionosphere data
fof2_vals = []
hmf_vals = []
datetime_vals = []
classifier_vals = []
confidence_vals = []
for index in range(len(iono_params)):
    if index < int(window_size / 2) or index >= len(iono_params) - np.ceil(window_size / 2):
        continue
    # build feature vector
    # start_index = index - int(window_size / 2)
    # end_index = index + int(window_size / 2) + 1
    start_index = np.min(np.where(timestamps_rounded >= timestamps_rounded[index] -
                                  datetime.timedelta(hours=window_hours / 2))[0])
    mid_index = index
    end_index = np.max(np.where(timestamps_rounded <= timestamps_rounded[index] +
                                datetime.timedelta(hours=window_hours / 2))[0])
    feature_vector = utils.make_feature_vector(
        pd.to_datetime(iono_params['datetime'].values[start_index:end_index+1]).to_julian_date(),
        fof2_artist[start_index:end_index+1],
        slope=True,
        hmf_artist=iono_params['hmF2'].values[start_index:end_index+1],
        confidence_scores=iono_params['cs'].values[start_index:end_index+1],
        window_stats=True,
        mid_index=mid_index-start_index)
    # if np.any(np.isnan(feature_vector)):
    #     continue
    datetime_vals.append(pd.to_datetime(iono_params['datetime'].values[index]))
    fof2_vals.append(iono_params['foF2'].values[index])
    hmf_vals.append(iono_params['hmF2'].values[index])
    confidence_vals.append(iono_params['cs'].values[index])
    if np.any(np.isnan(feature_vector)):
        classifier_vals.append(True)
    else:
        classifier_vals.append(classifier.predict(scaler.transform(feature_vector.reshape(1, -1)))[0])

datetime_vals = np.asarray(datetime_vals)
fof2_vals = np.asarray(fof2_vals)
hmf_vals = np.asarray(hmf_vals)
classifier_vals = np.asarray(classifier_vals)
confidence_vals = np.asarray(confidence_vals)
# Plot
fig, ax = plt.subplots()
ax.plot(datetime_vals[classifier_vals == False], fof2_vals[classifier_vals == False], '.b', label='Good')
ax.plot(datetime_vals[classifier_vals], fof2_vals[classifier_vals], '.r', label='Bad')
ax.grid()
ax.set_title('ARTIST Values Wallops Station', fontdict={'fontsize': 18})
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.set_xlabel('Time (UTC)', fontdict={'fontsize': 16})
ax.set_ylabel('foF2 (MHz)', fontdict={'fontsize': 16})
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(12)
ax.legend(prop={'size': 12})

# Plot confidence scores
fig, ax = plt.subplots()
ax.plot(datetime_vals[classifier_vals == False], confidence_vals[classifier_vals == False], '.b', label='Good')
ax.plot(datetime_vals[classifier_vals], confidence_vals[classifier_vals], '.r', label='Bad')
ax.grid()
ax.set_title('ARTIST Values Wallops Station', fontdict={'fontsize': 18})
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.set_xlabel('Time (UTC)', fontdict={'fontsize': 16})
ax.set_ylabel('Confidence Score', fontdict={'fontsize': 16})
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(12)
ax.legend(prop={'size': 12})


# Load the themens data
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
num_features = 9
window_size = 9

all_features = None
all_categories = np.zeros(0, dtype='bool')
all_predictions = np.zeros(0, dtype='bool')
all_artist_fof2s = np.array([])
all_artist_hmf2s = np.array([])
all_confidence_scores = np.array([])
all_themens_fof2s = np.array([])
all_themens_hmf2s = np.array([])
all_isprocessed = np.zeros(0, dtype='bool')
all_processed_type = np.array([])  # 1= Processed, 2=not processed nan, 3=not enough points to process
for file in all_files:
    try:
        print('Loading ' + os.path.join(main_dir, file))
        h5data = h5py.File(os.path.join(main_dir, file), 'r')
        fof2_artist = h5data['ARTIST_foF2'][:]
        hmf2_artist = h5data['ARTIST_hmF2'][:]
        fof2_manual = h5data['Manual_foF2'][:]
        hmf2_manual = h5data['Manual_hmF2'][:]
        if len(fof2_artist) <= window_size*2:
            continue
        timestamps = [julian.from_jd(i) for i in h5data['Julian_dates'][:]]
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
        timestamps_rounded = pd.to_datetime(iono_params['datetime'].values).round('1s')
        # fof2_filtered = utils.filter_data_highbias(fof2_artist, window_size=window_size)
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
            # start_index = i - int(window_size / 2)
            # end_index = i + int(window_size / 2)
            start_index = np.min(np.where(timestamps_rounded >= timestamps_rounded[i] -
                                          datetime.timedelta(hours=window_hours / 2))[0])
            mid_index = i
            end_index = np.max(np.where(timestamps_rounded <= timestamps_rounded[i] +
                                        datetime.timedelta(hours=window_hours / 2))[0])
            # Build the feature vector if there are sufficient data around the sample
            if start_index >= 0 and end_index < len(fof2_artist) and (
                    dt - timestamps[start_index]) < datetime.timedelta(hours=2) and (
                    timestamps[end_index] - dt) < datetime.timedelta(hours=2) and \
                    np.all(np.isfinite(fof2_artist[start_index:end_index+1])) and \
                    np.all(np.isfinite(fof2_manual[start_index:end_index+1])) and \
                    np.all(np.isfinite(hmf2_artist[start_index:end_index+1])): # and np.all(np.isfinite(fof2_filtered[start_index:end_index+1])):
                print(fof2_a, fof2_m, dt)
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
                        print('uh oh')
                    all_categories = np.hstack([all_categories, abs(fof2_m-fof2_a) > threshold_mhz])
                    all_predictions = np.append(all_predictions, classifier.predict(scaler.transform(features.reshape(1, -1)))[0])
                    all_isprocessed = np.append(all_isprocessed, True)
                    all_processed_type = np.append(all_processed_type, 1)
                else:
                    all_isprocessed = np.append(all_isprocessed, False)
                    all_processed_type = np.append(all_processed_type, 2)
            else:
                all_isprocessed = np.append(all_isprocessed, False)
                all_processed_type = np.append(all_processed_type, 3)
    except:
        print('here')

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
ax.plot(all_confidence_scores[all_isprocessed][all_predictions==False], all_artist_fof2s[all_isprocessed][all_predictions==False] -
        all_themens_fof2s[all_isprocessed][all_predictions==False], '.b', label='Good', markersize=1)
ax.plot(all_confidence_scores[all_isprocessed][all_predictions], all_artist_fof2s[all_isprocessed][all_predictions] -
        all_themens_fof2s[all_isprocessed][all_predictions], '.r', label='Bad', markersize=1)
ax.set_xlabel('Confidence Score', fontdict={'fontsize': 16})
ax.set_ylabel('ARTIST foF2 Error (MHz)', fontdict={'fontsize': 16})
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(12)
ax.grid()
ax.set_title('Classifier and Confidence Score Comparison', fontdict={'fontsize': 18})
ax.legend(prop={'size': 12})


# Make a box-whisker plot of each confidence score error
fig, ax = plt.subplots()
all_fof2_errors = all_artist_fof2s - all_themens_fof2s
valid_indeces = np.isfinite(all_fof2_errors)*np.isfinite(all_confidence_scores)
valid_confidence_scores = all_confidence_scores[valid_indeces]
valid_fof2_errors = all_fof2_errors[valid_indeces]
confidence_scores_set = np.unique(valid_confidence_scores)
box_arrays = []
positions = []
for confidence_score in confidence_scores_set:
    set_indeces = valid_confidence_scores == confidence_score
    box_arrays.append(copy.copy(valid_fof2_errors[set_indeces]))
    positions.append(confidence_score)
ax.boxplot(box_arrays, positions=positions, widths=2, whis=[2.5, 97.5], showfliers=True, flierprops={'markersize': 1, 'marker': '.'})
ax.set_xlabel('Confidence Score', fontdict={'fontsize': 16})
ax.set_ylabel('foF2 Error (MHz)', fontdict={'fontsize': 16})
ax.set_title('ARTIST foF2 Errors vs Confidence Score', fontdict={'fontsize': 18})
ax.grid()
xticks = np.linspace(0, 100, 11)
ax.set_xticks(xticks)
xtick_labels = []
for xtick in xticks:
    xtick_labels.append('{:.0f}'.format(xtick))
ax.set_xticklabels(xtick_labels)

# Make a box-whisker plot showing the test data
# Split the features into test/train
X = scaler.transform(all_features)
X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(X), pd.DataFrame(all_categories), test_size=0.3)
classifier.fit(X_train, y_train)
test_indeces = y_test.index.values
test_predictions = classifier.predict(X_test)
fig, ax = plt.subplots()
test_fof2_errors = all_artist_fof2s[all_isprocessed][test_indeces] - all_themens_fof2s[all_isprocessed][test_indeces]
test_predictions_set = np.unique(test_predictions)
test_box_arrays = [test_fof2_errors[test_predictions], test_fof2_errors[test_predictions==False]]
test_positions = [0, 1]
r = ax.boxplot(test_box_arrays, positions=test_positions, widths=0.33, whis=[2.5, 97.5], showfliers=True, flierprops={'markersize': 1, 'marker': '.'})
ax.set_xlabel('Classifier Prediction', fontdict={'fontsize': 16})
ax.set_ylabel('foF2 Error (MHz)', fontdict={'fontsize': 16})
ax.set_title('ARTIST foF2 Errors vs Classifier Prediction', fontdict={'fontsize': 18})
ax.grid()
ax.set_xticks([0, 1])
ax.set_xticklabels(['Bad', 'Good'])
xtick_labels = []
for xtick in xticks:
    xtick_labels.append('{:.0f}'.format(xtick))
ax.set_xticklabels(xtick_labels)

# Make confusion matrix
ax = metrics.ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test, normalize=None, cmap=plt.cm.Blues,
                                                   display_labels=['Good', 'Bad']).ax_
ax.set_ylabel('True Label', fontdict={'fontsize': 16})
ax.set_xlabel('Predicted Label', fontdict={'fontsize': 16})
ax.set_title('Confusion Matrix', fontdict={'fontsize': 18})
