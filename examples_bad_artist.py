import pandas as pd
import copy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image
import os
from datetime import datetime
import numpy as np
import scipy.signal
from sktime.transformations.series.outlier_detection import HampelFilter


def filter_data(data, window_size=13, npoly=5, outlier_thresh=0.25):
    # Do a SG filter on the data, iteratively remove the largest outliers and refit until no outliers
    data_sg = scipy.signal.savgol_filter(data, window_size, npoly, mode='nearest')
    # Get the difference between data and data_sg
    diff = np.abs(data_sg-data)
    while np.nanmax(diff) > outlier_thresh:
        # Remove the maximum difference and redo the fit
        data[np.nanargmax(diff)] = np.nan
        data_sg = scipy.signal.savgol_filter(data, window_size, npoly, mode='nearest')
        diff = np.abs(data_sg - data)
    return data, data_sg


def filter_data_highbias(data, window_size=13):
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    # Do a running max filter on the data before filtering
    nrows = (data.size - window_size) + 1
    n = data.strides[0]
    s = np.lib.stride_tricks.as_strided(data, shape=(nrows, window_size), strides=(n, n))
    max_middle_segment = np.nanmax(s, 1)
    max_start = np.asarray([np.nanmax(data[0:(i+1)]) for i in range(window_size//2)])
    max_end = np.asarray([np.nanmax(data[-(window_size // 2 - i):]) for i in range(window_size // 2)])
    data_max_win = np.concatenate((max_start, max_middle_segment, max_end))
    return data_max_win


# This will make plots that show examples of poor ARTIST performance
#station_id = 'WP937'    mh_station_id = 'MHJ45'  # 'HAJ43'
#    pf_station_id = 'EI764'  # 'CO764' 'PF765'
station_id = 'EI764'
ionogram_data = pd.read_csv('ionogram_data/ionogram_parameters202108.csv', parse_dates=['datetime'])
station_data_orig = copy.deepcopy(ionogram_data[ionogram_data['station_id'] == station_id])


# Build a feature vector for each data point
# Features will be all relative, so diff between window and filter
def build_feature_vectors(station_data):
    timestamps = station_data_orig['datetime'].values
    fof2s = station_data_orig['fof2'].values
    fof2s_filt = filter_data_highbias(copy.copy(station_data_orig['fof2'].values))
    num_features = 9
    feature_vectors = np.zeros((len(fof2s), num_features))
    is_bad = np.zeros(len(fof2s))
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(len(fof2s)):
        for j in range(num_features):
            point_index = i-(j-int(num_features/2))
            if 0 <= point_index < len(fof2s) and not np.isnan(fof2s[point_index]):
                feature_vectors[i, j] = fof2s_filt[point_index] - fof2s[point_index]
        # See if this is an outlier
        # load the ionogram and plot it
        filename = "{}_{}.png".format(station_data_orig['station_id'].values[i],
                                      pd.Timestamp(station_data_orig['datetime'].values[i]).to_pydatetime().strftime(
                                          '%Y%b%d_%H%M%S'))
        img = Image.open(os.path.join('ionosondeparser',
                                      pd.Timestamp(station_data_orig['datetime'].values[i]).to_pydatetime().strftime(
                                          'images%Y%m%d'), filename))
        ax.imshow(img)
        #plt.show(block=False)
        plt.pause(0.1)
        value = input("1 = Bad, nothing = Good:\n")
        if len(value) > 0:
            is_bad[i] = 1
    return [feature_vectors, is_bad]


[feature_vectors, is_bad] = build_feature_vectors(station_data_orig)



data_orig = station_data_orig['fof2'].values
fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
plt.plot(station_data_orig['datetime'], station_data_orig['fof2'])
# Filter the fof2 data
window_size = 13
npoly = 2
avg_data = np.convolve(station_data_orig['fof2'], np.ones(window_size)/window_size, mode='same')
plt.plot(station_data_orig['datetime'], avg_data)
data_savgolfilt = scipy.signal.savgol_filter(station_data_orig['fof2'].values, window_size, npoly, mode='nearest')
plt.plot(station_data_orig['datetime'], data_savgolfilt)
# Filter the data using the savgol filter and remove outliers
outlier_thresh = 0.1
outlier_inds = np.where(np.abs(data_savgolfilt - data_orig) > outlier_thresh)[0]
data_savgolfilt_outlierfilt = copy.copy(data_orig)
data_savgolfilt_new = copy.copy(data_savgolfilt)
while len(outlier_inds) > 0:
    data_savgolfilt_outlierfilt[outlier_inds] = copy.copy(data_savgolfilt_new[outlier_inds])
    data_savgolfilt_new = scipy.signal.savgol_filter(data_savgolfilt_outlierfilt, window_size, npoly, mode='nearest')
    outlier_inds = np.where(np.abs(data_savgolfilt_new - data_savgolfilt_outlierfilt) > outlier_thresh)[0]
plt.plot(station_data_orig['datetime'], data_savgolfilt_outlierfilt)
# Hampel filter
transformer = HampelFilter(window_length=10, n_sigma=1)
data_hampel = transformer.fit_transform(station_data_orig.reset_index()['fof2'])
plt.plot(station_data_orig['datetime'], data_hampel)
plt.grid()
plt.xlabel('Time (UTC)')
plt.ylabel('foF2 (MHz)')
plt.title('ARTIST Extracted foF2 for {}, ws {}, np {}'.format(station_id, window_size, npoly))
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
for label in ax.get_xticklabels(which='major'):
    label.set(rotation=15, horizontalalignment='right')
plt.figure()
plt.plot(station_data_orig['datetime'], station_data_orig['fof2'].values)
plt.plot(station_data_orig['datetime'], filter_data(copy.copy(station_data_orig['fof2'].values), window_size=31, npoly=5)[0])
plt.plot(station_data_orig['datetime'], filter_data(copy.copy(station_data_orig['fof2'].values), window_size=9, npoly=7, outlier_thresh=0.2)[0])
plt.plot(station_data_orig['datetime'], filter_data(copy.copy(station_data_orig['fof2'].values), window_size=9, npoly=7, outlier_thresh=0.2)[1])
plt.plot(station_data_orig['datetime'], filter_data_highbias(copy.copy(station_data_orig['fof2'].values), window_size=5))


fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
ax.plot(station_data_orig['datetime'], station_data_orig['fof2'].values)
ax.plot(station_data_orig['datetime'], filter_data_highbias(copy.copy(station_data_orig['fof2'].values), window_size=5))
ax.legend(['Original Data', '5 Sample Max Window'])
ax.grid()
ax.set_xlabel('Time (UTC)')
ax.set_ylabel('foF2 (MHz)')
ax.set_title('ARTIST Extracted foF2 for {}'.format(station_id))
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
for label in ax.get_xticklabels(which='major'):
    label.set(rotation=15, horizontalalignment='right')


# Load in the WP937 ionograms and plot the images
# Get all image filenames with the station id
image_dir = os.path.join('ionosondeparser', 'images20210802')
station_filenames = [f for f in os.listdir(image_dir) if f.startswith(station_id)]
ionogram_datetimes = [datetime.strptime(''.join(station_filename.split('.')[0].split('_')[1:3]), '%Y%b%d%H%M%S') for station_filename in station_filenames]
image_index_sequence = [0, 1, 2, 3, 4, 5]
image_fof2_sequence = [7, 7.25, 6.7, 5.8, 5.85, 7.35]
fig, axs = plt.subplots(3, 2, figsize=(6, 7))
fig.suptitle('{} Ionograms {}'.format(station_id, ionogram_datetimes[0].strftime('%Y-%m-%d')))
for i, img_i in enumerate(image_index_sequence):
    img = Image.open(os.path.join(image_dir, station_filenames[img_i]))
    axs[int(i / 2), i % 2].imshow(img)
    axs[int(i / 2), i % 2].set_xticks([])
    axs[int(i / 2), i % 2].set_yticks([])
    axs[int(i / 2), i % 2].set_title('{} foF2 = {:1.3f} MHz'.format(ionogram_datetimes[i].strftime('%H:%M:%S'), image_fof2_sequence[i]))


# Get all image filenames with the station id
image_dir = os.path.join('ionosondeparser', 'images20210801')
station_filenames = [f for f in os.listdir(image_dir) if f.startswith(station_id)]
ionogram_datetimes = [datetime.strptime(''.join(station_filename.split('.')[0].split('_')[1:3]), '%Y%b%d%H%M%S') for station_filename in station_filenames]
image_index_sequence = [0, 1, 2, 3, 4, 5]
image_fof2_sequence = [6.45, 4.45, 5.65, 5.7, 5.6, 5.85]
nrows = 2
ncols = len(image_index_sequence)//nrows
fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*3, 7))
fig.suptitle('{} Ionograms {}'.format(station_id, ionogram_datetimes[0].strftime('%Y-%m-%d')))
for i, img_i in enumerate(image_index_sequence):
    img = Image.open(os.path.join(image_dir, station_filenames[img_i]))
    if nrows > 1:
        axs[i // ncols, i % ncols].imshow(img.crop([0,0,400,600]))
        axs[i // ncols, i % ncols].set_xticks([])
        axs[i // ncols, i % ncols].set_yticks([])
        axs[i // ncols, i % ncols].set_title('{} foF2 = {:1.3f} MHz'.format(ionogram_datetimes[i].strftime('%H:%M:%S'), image_fof2_sequence[i]))
    else:
        axs[i].imshow(img.crop([0,0,400,600]))
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_title('{} foF2 = {:1.3f} MHz'.format(ionogram_datetimes[i].strftime('%H:%M:%S'), image_fof2_sequence[i]))

print('Done.')
