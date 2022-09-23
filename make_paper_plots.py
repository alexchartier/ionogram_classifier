# This script will make the paper plots
import copy
import datetime
import h5py
import julian
import matplotlib.cm
import matplotlib.colors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image, ImageDraw, ImageFont
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn import metrics
from sklearn.model_selection import train_test_split
import utils
import examples_isr
import time


def new_fig(usetex=False, figsize=(5, 4)):
    if usetex:
        plt.rc('text', usetex=usetex)
    else:
        plt.rc('font', family='serif')
    fig = plt.figure(figsize=figsize)
    return fig


def arrowed_line(im, ptA, ptB, width=1, color=(0, 255, 0)):
    """Draw line from ptA to ptB with arrowhead at ptB"""
    # Get drawing context
    draw = ImageDraw.Draw(im)
    # Draw the line without arrows
    draw.line((ptA,ptB), width=width, fill=color)
    # Now work out the arrowhead
    # = it will be a triangle with one vertex at ptB
    # - it will start at 95% of the length of the line
    # - it will extend 8 pixels either side of the line
    x0, y0 = ptA
    x1, y1 = ptB
    # Now we can work out the x,y coordinates of the bottom of the arrowhead triangle
    xb = 0.95*(x1-x0)+x0
    yb = 0.95*(y1-y0)+y0
    # Work out the other two vertices of the triangle
    # Check if line is vertical
    if x0==x1:
       vtx0 = (xb-5, yb)
       vtx1 = (xb+5, yb)
    # Check if line is horizontal
    elif y0==y1:
       vtx0 = (xb, yb+5)
       vtx1 = (xb, yb-5)
    else:
       alpha = np.math.atan2(y1-y0, x1-x0)-90*np.pi/180
       a = 8*np.cos(alpha)
       b = 8*np.sin(alpha)
       vtx0 = (xb+a, yb+b)
       vtx1 = (xb-a, yb-b)
    #draw.point((xb,yb), fill=(255,0,0))    # DEBUG: draw point of base in red - comment out draw.polygon() below if using this line
    #im.save('DEBUG-base.png')              # DEBUG: save
    # Now draw the arrowhead triangle
    draw.polygon([vtx0, vtx1, ptB], fill=color)
    return im


def make_example_ionograms(show=False, same_fig=True):
    ionogram_dir = os.path.join('ionosondeparser', 'images202108')
    good_ionogram_example_file = 'WP937_2021Aug01_012507.png' #'WP937_2021Aug01_000010.png' #'WP937_2021Aug01_012507.png'
    good_ionogram_top_coords = (338, 225) #(364, 225)# (339, 225)
    good_ionogram_bot_coords = (338, 519) #(364, 519) # (339, 519)
    #good_ionogram_text = 'Good Autoscaling'
    good_ionogram_text = 'Accurate Autoscaling'
    bad_ionogram_example_file =  'WP937_2021Aug01_013008.png' #'WP937_2021Aug01_000506.png' #'WP937_2021Aug01_013008.png'
    bad_ionogram_top_coords = (310, 225) #(299, 225) # (311,225)
    bad_ionogram_bot_coords = (310, 519) #(299, 519) # (311,519)
    #bad_ionogram_text = 'Poor Autoscaling'
    bad_ionogram_text = 'Inaccurate Autoscaling'
    if same_fig:
        fig = plt.figure(1, figsize=(13.5, 5))
        grid = ImageGrid(fig, 111, nrows_ncols=(1,2), axes_pad=0.1)
        grid_ind = 0
    for img_file, top_coords, bot_coords, text in zip([good_ionogram_example_file, bad_ionogram_example_file],
                                                      [good_ionogram_top_coords, bad_ionogram_top_coords],
                                                      [good_ionogram_bot_coords, bad_ionogram_bot_coords],
                                                      [good_ionogram_text, bad_ionogram_text]):
        img = Image.open(os.path.join(ionogram_dir, img_file))
        img = arrowed_line(img, top_coords, bot_coords, width=1, color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text(top_coords, text, fill=(0, 0, 0), anchor="ld", font=ImageFont.truetype("arial.ttf", 20))
        if same_fig:
            grid[grid_ind].imshow(img, interpolation=None)
            grid[grid_ind].set_axis_off()
            grid_ind += 1
            if grid_ind == 2:
                fig.savefig(os.path.join('images', 'ExampleIonograms.png'))
        else:
            if show:
                img.show()
            img.save(os.path.join('images', '{}.png'.format(text.replace(' ', ''))))


def make_jan_nmf2_plots_with_isr():
    day_to_analyze = datetime.datetime(2012, 1, 15)
    main_dir = os.path.join('C:\\', 'Users', 'sugargf1', 'Box', 'Data')
    mhisr_dir = os.path.join(main_dir, 'MillstoneHill')
    start_time = day_to_analyze
    stop_time = day_to_analyze + datetime.timedelta(days=1)
    mh_station_id = 'MHJ45'  # 'HAJ43'
    [pfisr_filenames, mhisr_filenames, ionogram_data_filename, pfisr_filename,
     mhisr_filename] = examples_isr.get_filenames_for_day(day_to_analyze, mhisr_dir=mhisr_dir)
    ionogram_data = pd.read_csv(ionogram_data_filename, parse_dates=['datetime'])
    mh_ig_data = ionogram_data[ionogram_data['station_id'] == mh_station_id]
    mh_isr_data = examples_isr.hdf5_to_dataframe(mhisr_filename)
    mh_isr_data = mh_isr_data[mh_isr_data['el'] == 90]
    ax = utils.plot_isr_and_ionosonde_data(mh_isr_data, mh_ig_data,
                                           title='Millstone Hill',
                                           ylabel='Maximum Density [m$^{-3}$]', val='nmf', plot_filter=False,
                                           classifier_filename=os.path.join('classifiers',
                                                                            'scaler_and_classifiers_and_betas_27_0.25_16.0.sav'))
    ax.set_xlim(start_time, stop_time)


def make_august_nmf2_plots_with_classifier_labels(classifier_file_to_load=None, grayscale=False):
    if classifier_file_to_load is None:
        classifier_file_to_load = os.path.join('classifiers', 'scaler_and_classifiers_and_betas_27_0.25_16.0.sav')
    # Setup parameters
    station_id = 'WP937'
    window_hours = 2

    # Download wallops image
    print('Downloading ionogram parameters')
    iono_params = utils.get_iono_params(station_id, ['foF2', 'hmF2'], datetime.datetime(2021, 8, 1, 0, 0), datetime.datetime(2021, 8, 4, 0, 0))
    # Get the features
    fof2_artist = iono_params['foF2'].values

    # Load the classifier and scaler
    print('Loading the classifier and scaler')
    with open(classifier_file_to_load, 'rb') as fid:
        [scaler, classifiers, betas] = pickle.load(fid)

    # Loop through all ionosphere data
    fof2_vals = []
    datetime_vals = []
    classifier_vals = [[] for i in classifiers]
    confidence_vals = []
    timestamps_rounded = pd.to_datetime(iono_params['datetime'].values).round('1s')
    for index in range(len(iono_params)):
        # See if we can build a feature vector
        datetime_vals.append(pd.to_datetime(iono_params['datetime'].values[index]))
        fof2_vals.append(iono_params['foF2'].values[index])
        confidence_vals.append(iono_params['cs'].values[index])
        # build feature vector
        start_index = np.min(np.where(timestamps_rounded >= timestamps_rounded[index] -
                                      datetime.timedelta(hours=window_hours / 2))[0])
        mid_index = index
        end_index = np.max(np.where(timestamps_rounded <= timestamps_rounded[index] +
                                    datetime.timedelta(hours=window_hours / 2))[0])
        if mid_index == end_index or mid_index == start_index:
            # We can't build feature vector,
            print('Index: {} mid_index: {}, end_index: {}'.format(index, mid_index, end_index))
            for iclassifier, classifier in enumerate(classifiers):
                classifier_vals[iclassifier].append(0)
        else:
            feature_vector = utils.make_feature_vector(
                pd.to_datetime(iono_params['datetime'].values[start_index:end_index+1]).to_julian_date(),
                fof2_artist[start_index:end_index+1],
                slope=True,
                hmf_artist=iono_params['hmF2'].values[start_index:end_index+1],
                confidence_scores=iono_params['cs'].values[start_index:end_index+1],
                window_stats=True,
                mid_index=mid_index-start_index)
            if np.any(np.isnan(feature_vector)):
                print('Nans in feature vector')
                print(iono_params.iloc[index].datetime)
                for iclassifier, classifier in enumerate(classifiers):
                    classifier_vals[iclassifier].append(0)
            else:
                for iclassifier, classifier in enumerate(classifiers):
                    if classifier.predict(scaler.transform(feature_vector.reshape(1, -1)))[0]:
                        classifier_vals[iclassifier].append(1)
                    else:
                        classifier_vals[iclassifier].append(-1)
    datetime_vals = np.asarray(datetime_vals)
    fof2_vals = np.asarray(fof2_vals)
    confidence_vals = np.asarray(confidence_vals)
    for beta, classifier_val in zip(betas, classifier_vals):
        vals_to_plot = fof2_vals
        label_to_plot = 'ARTIST foF2 (MHz)'
        classifier_val = np.asarray(classifier_val)
        # Plot
        if grayscale:
            cmap = plt.get_cmap('binary')
            isr_color = 'tab:grey'
        else:
            cmap = plt.get_cmap('viridis')
            isr_color = 'tab:red'
        norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
        fig = new_fig(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)
        indeces = classifier_val == -1
        ax.scatter(datetime_vals[indeces], vals_to_plot[indeces], c=confidence_vals[indeces], linewidths=0.5,
                   cmap=cmap, norm=norm, marker='o', edgecolor='black', label='Accurate', alpha=1, s=30)
        indeces = classifier_val == 1
        ax.scatter(datetime_vals[indeces], vals_to_plot[indeces], c=confidence_vals[indeces], linewidths=0.5,
                   cmap=cmap, norm=norm, marker='*', edgecolor='black', label='Inaccurate', alpha=1, s=60)
        indeces = classifier_val == 0
        ax.scatter(datetime_vals[indeces], vals_to_plot[indeces], c=confidence_vals[indeces], linewidths=0.5,
                   cmap=cmap, norm=norm, marker='d', edgecolor='black', label='Not Classified', alpha=1, s=60)
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cb.set_label('Confidence Score', fontdict={'fontsize': 16})
        ax.grid()
        ax.set_title('Wallops Island $\\beta$={}'.format(beta), fontdict={'fontsize': 18})
        ax.set_xlabel('Time (UTC)', fontdict={'fontsize': 16})
        ax.set_ylabel(label_to_plot, fontdict={'fontsize': 16})
        if label_to_plot == 'ARTIST nmF2 (m$^{-3}$)':
            ax.set_yscale('log')
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(12)
        ax.legend(prop={'size': 12})
        plt.tight_layout()
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))


# This function will take a DataFrame built from reading in parsed ionogram images and add Themens data that sometimes
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


def make_hyperparameter_plots(classifier_file_to_load=None):
    if classifier_file_to_load is None:
        classifier_file_to_load = os.path.join('classifiers', 'scaler_and_classifiers_and_betas_27_0.25_16.0.sav')
    # Load the classifier
    print('Loading ' + classifier_file_to_load)
    with open(classifier_file_to_load, 'rb') as fid:
        [scaler, classifiers, betas] = pickle.load(fid)
    class_weight = []
    max_depth = []
    max_features = []
    min_samples_leaf = []
    min_samples_split = []
    n_estimators = []
    for classifier in classifiers:
        class_weight.append(classifier.class_weight[True])
        max_depth.append(classifier.max_depth)
        max_features.append(classifier.max_features)
        min_samples_leaf.append(classifier.min_samples_leaf)
        min_samples_split.append(classifier.min_samples_split)
        n_estimators.append(classifier.n_estimators)
    plt.figure()
    plt.plot(betas, class_weight)
    plt.title('Class Weight')
    plt.figure()
    plt.plot(betas, max_depth)
    plt.title('Max Depth')
    plt.figure()
    plt.plot(betas, max_features)
    plt.title('max_features')
    plt.figure()
    plt.plot(betas, min_samples_leaf)
    plt.title('min_samples_leaf')
    plt.figure()
    plt.plot(betas, min_samples_split)
    plt.title('min_samples_split')
    plt.figure()
    plt.plot(betas, n_estimators)
    plt.title('n_estimators')


def make_confidence_score_classifier_plots(ax=None):
    # This function will make a plot that uses only the confidence score as a classifier
    # Get themens data
    [all_features, all_categories, all_isprocessed] = load_themens_data(pickle_file='themens_data.p')
    cs_index = 24
    thresholds = np.linspace(0, 100, 11)
    classifications = np.zeros([len(all_features), len(thresholds)])
    confusion_matrices = np.zeros([len(thresholds), 2, 2])
    for i, feature_vector in enumerate(all_features):
        for j, threshold in enumerate(thresholds):
            classification = feature_vector[cs_index] < threshold
            true_classification = all_categories[i]
            classifications[i,j] = feature_vector[cs_index] < threshold
            if true_classification:
                true_index = 1
            else:
                true_index = 0
            if classification:
                guess_index = 1
            else:
                guess_index = 0
            confusion_matrices[j, true_index, guess_index] += 1

    # Normalize confusion_matrices
    confusion_matrices_normalized = copy.copy(confusion_matrices)
    for i in range(len(thresholds)):
        confusion_matrices_normalized[i, 0, 0] /= sum(confusion_matrices[i, 0, :])
        confusion_matrices_normalized[i, 1, 0] /= sum(confusion_matrices[i, 1, :])
        confusion_matrices_normalized[i, 0, 1] /= sum(confusion_matrices[i, 0, :])
        confusion_matrices_normalized[i, 1, 1] /= sum(confusion_matrices[i, 1, :])

    # Make the plot
    if ax is None:
        fig = new_fig()
        ax = fig.add_subplot(1, 1, 1)
    ax.plot(thresholds, [x[1, 1] for x in confusion_matrices_normalized], color='k', ls='solid',
            label='True Positive')
    ax.plot(thresholds, [x[0, 1] for x in confusion_matrices_normalized], color='0.5', ls='dashed',
            label='False Positive')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Classification Rate')
    ax.set_xlabel('Confidence Score Threshold')
    ax.grid('on')
    ax.legend(loc='lower right')
    ax.set_title('Confidence Score Classifier Performance')


def compare_classifiers(classifier_file_to_load=None, do_confidence_scores=False):
    if classifier_file_to_load is None:
        classifier_file_to_load = os.path.join('classifiers', 'scaler_and_classifiers_and_betas_27_0.25_16.0.sav')
    # Load the classifier
    print('Loading ' + classifier_file_to_load)
    with open(classifier_file_to_load, 'rb') as fid:
        [scaler, classifiers, betas] = pickle.load(fid)
    # Get themens data
    [all_features, all_categories, all_isprocessed] = load_themens_data(pickle_file='themens_data.p')
    # [all_features, all_categories, all_isprocessed] = load_themens_data()
    # Split the data into train and test sets
    X = scaler.transform(all_features)
    X_train, X_test, y_train, y_test = train_test_split(X, all_categories, test_size=0.3, random_state=1)
    # Loop through classifiers and calculate the score
    classifier_scores = []
    print("Beta     : False Positive      True Positive")
    for iclassifier, classifier in enumerate(classifiers):
        classifier.fit(X_train, y_train)
        confusion_matrix = metrics.confusion_matrix(y_test, classifier.predict(X_test), normalize='true')
        classifier_scores.append([betas[iclassifier], copy.copy(confusion_matrix)])
        print("Beta {}: {}, {}".format(betas[iclassifier], confusion_matrix[0, 1], confusion_matrix[1, 1]))
    if do_confidence_scores:
        fig = new_fig(figsize=(7, 3.5))
        ax = fig.add_subplot(1, 2, 1)
    else:
        fig = new_fig()
        ax = fig.add_subplot(1, 1, 1)
    ax.plot([x[0] for x in classifier_scores], [x[1][1, 1] for x in classifier_scores], color='k', ls='solid', label='True Positive')
    ax.plot([x[0] for x in classifier_scores], [x[1][0, 1] for x in classifier_scores], color='0.5', ls='dashed', label='False Positive')
    ax.set_xlim(0, 16.1)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Classification Rate')
    ax.set_xlabel(r'$\beta$')
    ax.grid('on')
    ax.legend(loc='lower right')
    ax.set_title('Random Forest Classifier')
    if do_confidence_scores:
        ax = fig.add_subplot(1, 2, 2)
        make_confidence_score_classifier_plots(ax=ax)
        ax.set_ylabel('')
        ax.set_yticklabels({})
        ax.set_xlim([0, 100])
        ax.legend(loc='upper left')
        ax.tick_params(axis='y', length=0)
        ax.set_title('Confidence Score Thresh. Classifier')
        plt.tight_layout()
    print('here')

def make_confidence_score_histograms(classifier_file_to_load=None):
    if classifier_file_to_load is None:
        classifier_file_to_load = os.path.join('classifiers', 'scaler_and_classifiers_and_betas_27_0.25_16.0.sav')
    # Load the classifier
    print('Loading ' + classifier_file_to_load)
    with open(classifier_file_to_load, 'rb') as fid:
        [scaler, classifiers, betas] = pickle.load(fid)
    # Get themens data
    [all_features, all_categories, all_isprocessed] = load_themens_data(pickle_file='themens_data.p')
    af_train, af_test, y_train, y_test = train_test_split(all_features, all_categories, test_size=0.3, random_state=1)
    # Split the data into train and test sets
    X = scaler.transform(all_features)
    X_train, X_test, y_train, y_test = train_test_split(X, all_categories, test_size=0.3, random_state=1)
    # Train the classifier
    classifier_index_to_use = 2
    classifier = classifiers[classifier_index_to_use]
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)

    fig = new_fig(figsize=(5.25, 2.5))
    for i in range(2):
        ax = fig.add_subplot(1, 2, i+1)
        ax.grid('on', zorder=0)
        if i == 0:
            confidence_scores_to_plot = all_features[all_categories, 24]
            ax.set_ylabel('Proportion of Samples')
            ax.set_title('Inaccurate Data')
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=0)
            ax.set_title('Accurate Data')
            confidence_scores_to_plot = all_features[all_categories==False, 24]
        ax.set_ylim(0, 0.75)
        ax.set_xlim(0, 100)
        ax.hist(confidence_scores_to_plot, bins=20, zorder=3,
                weights=np.ones(len(confidence_scores_to_plot)) / len(confidence_scores_to_plot), histtype='bar',
                edgecolor='black', facecolor='gray')
        ax.set_xlabel('Confidence Score')
    plt.tight_layout(pad=0.4)

    # Plot the proportion of each subset that is inaccurate vs accurate
    # Get the bin sizes
    inaccurate_bin_vals = np.histogram(all_features[all_categories, 24], bins=np.linspace(0,100,21))
    accurate_bin_vals = np.histogram(all_features[all_categories==False, 24], bins=np.linspace(0,100,21))
    total_bin_vals = inaccurate_bin_vals[0]+accurate_bin_vals[0]
    inaccurate_bin_rel_vals = inaccurate_bin_vals[0]/total_bin_vals
    accurate_bin_rel_vals = accurate_bin_vals[0]/total_bin_vals

    fig = new_fig()
    ax = fig.add_subplot(1,1,1)
    # h1 = ax.bar(np.linspace(0, 95, 20), accurate_bin_rel_vals, 5, bottom=inaccurate_bin_rel_vals, align='edge', fill=False, hatch='...')
    # h2 = ax.bar(np.linspace(0, 95, 20), inaccurate_bin_rel_vals, 5, align='edge', fill=False, hatch='///')
    h1 = ax.bar(np.linspace(0, 95, 20), accurate_bin_rel_vals, 5, bottom=inaccurate_bin_rel_vals, align='edge',
                facecolor='white', edgecolor='black', hatch='///', zorder=3)
    h2 = ax.bar(np.linspace(0, 95, 20), inaccurate_bin_rel_vals, 5, align='edge', facecolor='gray', edgecolor='black', zorder=3)
    # ax.grid('on', zorder=0)
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Proportion of Samples')
    ax.set_title('Confidence Score foF2 Performance')
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 1])
    ax.legend([h1, h2], ['Accurate', 'Inaccurate'], loc='upper center', bbox_to_anchor=(0.5, 1.35))


def load_only_themens_data():
    """
    Load all themens h5 files and put data into a single pandas dataframe.
    """
    main_dir = 'themens_data_with_hmf'
    all_files = os.listdir(main_dir)
    iono_params = pd.DataFrame()
    for file in all_files:
        print('Loading ' + os.path.join(main_dir, file))
        h5data = h5py.File(os.path.join(main_dir, file), 'r')
        dataFrameData = pd.DataFrame({'ARTIST_foF2': h5data['ARTIST_foF2'][:],
                                      'ARTIST_hmF2': h5data['ARTIST_hmF2'][:],
                                      'Manual_foF2': h5data['Manual_foF2'][:],
                                      'Manual_hmF2': h5data['Manual_hmF2'][:],
                                      'Julian_dates': h5data['Julian_dates'][:],
                                      'Station_ID': file.split('.')[0]})
        iono_params = pd.concat([iono_params, dataFrameData], ignore_index=True)
    return iono_params


def load_themens_data(pickle_file=None):
    if pickle_file is not None:
        with open(pickle_file, 'rb') as fid:
            return pickle.load(fid)

    main_dir = 'themens_data_with_hmf'
    all_files = os.listdir(main_dir)
    threshold_mhz = 0.5
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
            if len(fof2_artist) <= 4:
                continue
            timestamps = [julian.from_jd(i) for i in h5data['Julian_dates'][:]]

            # Round the timestamps to nearest second
            timestamps_rounded = pd.to_datetime(timestamps).round('1s')
            dates = np.unique([timestamp.date() for timestamp in timestamps])
            station_id = file.split('.')[0]

            # Download the data
            params_to_download = ['hmF2', 'hF']
            iono_params = pd.DataFrame()
            for date in dates:
                start_datetime = datetime.datetime.combine(date, datetime.datetime.min.time())
                end_datetime = datetime.datetime.combine(date, datetime.datetime.max.time())
                try:
                    iono_params = iono_params.append(utils.get_iono_params(station_id, params_to_download, start_datetime,
                                                                           end_datetime), ignore_index=True)
                except:
                    print('Error downloading data, waiting a few seconds and try again')
                    time.sleep(10)
                    iono_params = iono_params.append(
                        utils.get_iono_params(station_id, params_to_download, start_datetime,
                                              end_datetime), ignore_index=True)
            # Make sure iono_params['datetime'] values are equal to timestamps, if not then fill in or ignore values
            iono_params = adjust_iono_data(iono_params, timestamps)
            for i in range(len(fof2_artist)):
                # Save value in all_* arrays
                all_artist_fof2s = np.append(all_artist_fof2s, fof2_artist[i])
                all_artist_hmf2s = np.append(all_artist_hmf2s, hmf2_artist[i])
                all_confidence_scores = np.append(all_confidence_scores, iono_params['cs'].values[i])
                all_themens_fof2s = np.append(all_themens_fof2s, fof2_manual[i])
                all_themens_hmf2s = np.append(all_themens_hmf2s, hmf2_manual[i])

                fof2_a = fof2_artist[i]
                fof2_m = fof2_manual[i]
                dt = timestamps[i]
                # Calculate the start and end index based on time window size
                start_index = np.min(
                    np.where(timestamps_rounded >= timestamps_rounded[i] - datetime.timedelta(hours=window_hours / 2))[
                        0])
                mid_index = i
                end_index = np.max(
                    np.where(timestamps_rounded <= timestamps_rounded[i] + datetime.timedelta(hours=window_hours / 2))[
                        0])
                # Make sure there are at least 2 points on either side of the sample point
                if mid_index <= start_index + 1 or mid_index >= end_index - 2:
                    all_isprocessed = np.append(all_isprocessed, -1)
                    continue
                # Build the feature vector if there are sufficient data around the sample
                if start_index >= 0 and end_index < len(fof2_artist) and \
                        (dt - timestamps[start_index]) <= datetime.timedelta(hours=window_hours) and \
                        (timestamps[end_index] - dt) <= datetime.timedelta(hours=window_hours) and \
                        np.any(np.isfinite(fof2_artist[start_index:end_index + 1])) and \
                        np.any(np.isfinite(fof2_manual[start_index:end_index + 1])) and \
                        np.any(np.isfinite(hmf2_artist[
                                           start_index:end_index + 1])):  # and np.all(np.isfinite(fof2_filtered[start_index:end_index+1])):
                    print(fof2_a, fof2_m, dt)
                    features = utils.make_feature_vector(h5data['Julian_dates'][start_index:end_index + 1],
                                                         fof2_artist[start_index:end_index + 1],
                                                         slope=True,
                                                         hmf_artist=hmf2_artist[start_index:end_index + 1],
                                                         confidence_scores=iono_params['cs'].values[
                                                                           start_index:end_index + 1],
                                                         window_stats=True,
                                                         mid_index=mid_index - start_index)
                    # Make sure all features are finite
                    if np.all(np.isfinite(features)):
                        try:
                            if all_features is None:
                                all_features = features
                            else:
                                all_features = np.vstack([all_features, features])
                        except:
                            print('There is an error in stacking a new feature the the all_features array')
                        all_categories = np.hstack([all_categories, abs(fof2_m - fof2_a) > threshold_mhz])
                        all_isprocessed = np.append(all_isprocessed, 1)
                    else:
                        # Not all features are finite
                        all_isprocessed = np.append(all_isprocessed, -2)
                else:
                    # Not sufficient data around ionogram
                    all_isprocessed = np.append(all_isprocessed, -3)
        except Exception as e:
            print('Error reading file')
            print(e)
    return [all_features, all_categories, all_isprocessed]


def make_es_plots():
    station_ids = ['WP937', 'MHJ45', 'EG931', 'PRJ18', 'AL945', 'AU930']
    start_time = datetime.datetime(2021, 7, 13)
    end_time = datetime.datetime(2021, 7, 16)
    ax = utils.plot_foes_vs_time(station_ids, start_time, end_time, plot_den=True)


def make_es_map_plot():
    station_ids = ['WP937', 'MHJ45', 'EG931', 'PRJ18', 'AL945', 'AU930']
    start_time = datetime.datetime(2021, 7, 13)
    end_time = datetime.datetime(2021, 7, 16)
    station_data_all = pd.DataFrame()
    for station_id in station_ids:
        station_data = utils.get_iono_params(station_id, ['foEs'], start_time, end_time)
        station_data_all = pd.concat([station_data_all, station_data]).reset_index(drop=True)
    utils.plot_timeseries_on_world(station_data_all, val_to_plot='foEs', lon_span=40, lat_span=7,
                                   start_at_station=False, show_ids=True,
                                   timeseries_colors=['b.', 'k.', 'g.', 'm.', 'y.'],
                                   station_colors=['b', 'k', 'g', 'm', 'y'], title='foEs for 20210713-20210716')


def make_feature_importance_plot(classifier_filename=None):
    if classifier_filename is None:
        classifier_filename = os.path.join('classifiers', 'scaler_and_classifiers_and_betas_27_0.25_16.0.sav')
    # Load the classifiers
    print('Loading the classifier and scaler and beta')
    with open(classifier_filename, 'rb') as fid:
        [scaler, classifiers, betas] = pickle.load(fid)
    feature_names = [f"feature {i}" for i in range(len(classifiers[0].feature_importances_))]
    for beta, classifier in zip(betas, classifiers):
        fig, ax = plt.subplots()
        importances = classifier.feature_importances_
        forest_importances = pd.Series(importances, index=feature_names)
        std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature Importances for Beta = {}".format(beta))
        ax.set_ylabel("Mean Decrease in Impurity")
        ax.grid('on')
    ax.legend()


def make_isr_plot(grayscale=False):
    # Load the Millstone Hill data
    main_dir = os.path.join('data')
    mhisr_dir = os.path.join(main_dir, 'MillstoneHill')
    mhisr_filename = os.path.join(mhisr_dir, 'mlh120115m.004.hdf5')
    isr_data = utils.hdf5_to_dataframe(mhisr_filename)
    isr_data = isr_data[isr_data['el'] == 90]
    ionosonde_data = utils.get_iono_params('MHJ45', ['foF2', 'foF1', 'foE', 'foEs', 'hmF2'],
                                         datetime.datetime(2012, 1, 15, 0, 0, 0),
                                         datetime.datetime(2012, 1, 16, 0, 0, 0))
    ionosonde_val = np.nanmax(ionosonde_data[['foF2', 'foF1', 'foE', 'foEs']].values, 1)
    isr_val = np.asarray(isr_data['fof2'])
    isr_val_err = (utils.denm3_to_freqhz(
        np.asarray(isr_data['nmf']) + np.asarray(isr_data['dnmf'])) - utils.denm3_to_freqhz(
        np.asarray(isr_data['nmf']) - np.asarray(isr_data['dnmf']))) / 2 * 1e-6

    # Classify the ionosonde_vals
    classifier_filename = os.path.join('classifiers', 'scaler_and_classifiers_and_betas_27_0.25_16.0.sav')
    beta_to_use = 4
    with open(classifier_filename, 'rb') as fid:
        [scaler, classifiers, betas] = pickle.load(fid)
        good_classifier_index = np.where(betas == beta_to_use)[0][0]
        classifier = classifiers[good_classifier_index]

    # Convert ionosonde data into feature_vectors
    feature_vectors_all = utils.ionogram_data_to_feature_vectors(ionosonde_data)
    inaccurate_data = []
    accurate_data = []
    unclassified_data = []
    for feature_vector, timestamp, fof2, confidence_score in zip(feature_vectors_all, ionosonde_data['datetime'],
                                                                 ionosonde_data['foF2'], ionosonde_data['cs']):
        if len(feature_vector) == 0:
            unclassified_data.append([timestamp, fof2, confidence_score])
        elif classifier.predict(scaler.transform(feature_vector.reshape(1,-1))):
            inaccurate_data.append([timestamp, fof2, confidence_score])
        else:
            accurate_data.append([timestamp, fof2, confidence_score])
    inaccurate_data = np.asarray(inaccurate_data)
    accurate_data = np.asarray(accurate_data)
    unclassified_data = np.asarray(unclassified_data)
    if grayscale:
        cmap = plt.get_cmap('binary')
        isr_color = 'tab:grey'
    else:
        cmap = plt.get_cmap('viridis')
        isr_color = 'tab:red'

    norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
    fig = new_fig(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.fill_between(isr_data['datetime'], isr_val - isr_val_err, isr_val + isr_val_err,
                    alpha=1, facecolor=isr_color, edgecolor='black', label='ISR', linestyle='--')
    ax.scatter(accurate_data[:, 0], accurate_data[:, 1], c=accurate_data[:, 2], linewidths=0.5,
               marker='o', edgecolor='black', label='Accurate', alpha=1, s=30, cmap=cmap, norm=norm)
    ax.scatter(inaccurate_data[:, 0], inaccurate_data[:, 1], c=inaccurate_data[:, 2], linewidths=0.5,
               marker='*', edgecolor='black', label='Inaccurate', alpha=1, s=60, cmap=cmap, norm=norm)
    ax.scatter(unclassified_data[:, 0], unclassified_data[:, 1], c=unclassified_data[:, 2], linewidths=0.5,
               marker='d', edgecolor='black', label='Not Classified', alpha=1, s=60, cmap=cmap, norm=norm)
    ax.grid('on')
    ax.legend()
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cb.set_label('Confidence Score', fontdict={'fontsize': 16})
    ax.set_xlim([min(pd.concat([ionosonde_data['datetime'], isr_data['datetime']])),
                 max(pd.concat([ionosonde_data['datetime'], isr_data['datetime']]))])
    ax.set_title('Millstone Hill ISR Comparison', fontdict={'fontsize': 18})
    ax.set_ylabel('foF2 (MHz)', fontdict={'fontsize': 16})
    ax.set_xlabel('Time (UTC)', fontdict={'fontsize': 16})
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    plt.tight_layout()


if __name__ == '__main__':
    compare_classifiers(do_confidence_scores=True)
    plt.show()
    #make_isr_plot()
    make_august_nmf2_plots_with_classifier_labels()
    plt.show()
    make_confidence_score_histograms()
    plt.show()
    make_confidence_score_classifier_plots()
    plt.show()
    make_es_map_plot()
