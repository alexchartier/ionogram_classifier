import datetime
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cf
import numpy as np
import os
import pandas as pd
import geopy.distance
import copy
import pickle
import scipy.stats
import requests
from bs4 import BeautifulSoup as bs
import h5py


def ionogram_data_to_feature_vectors(ionogram_data, window_hours=2):
    """Convert ionogram data to feature vectors to be used by a classifier.

    Keyword arguments:
        window_hours -- the window size centered on the ionogram used to gather data for the feature vector (default 2)
    """
    # Round the timestamps to nearest seconds to handle small discrepancies
    timestamps_rounded = pd.to_datetime(ionogram_data['datetime'].values).round('1s')
    # Loop through all ionosphere data
    feature_vectors = []
    for index in range(len(ionogram_data)):
        # See if we can build a feature vector
        # Get the start, middle, and end indeces of windows we will use for the feature vector
        start_index = np.min(np.where(timestamps_rounded >= timestamps_rounded[index] -
                                      datetime.timedelta(hours=window_hours / 2))[0])
        mid_index = index
        end_index = np.max(np.where(timestamps_rounded <= timestamps_rounded[index] +
                                    datetime.timedelta(hours=window_hours / 2))[0])
        # Make sure there are sufficient data in the windows
        if mid_index <= start_index+2 or end_index <= mid_index+2:
            # We can't build feature vector,
            print('start_index: {} mid_index: {}, end_index: {}'.format(start_index, mid_index, end_index))
            feature_vectors.append([])
        else:
            feature_vector = make_feature_vector(
                pd.to_datetime(ionogram_data['datetime'].values[start_index:end_index + 1]).to_julian_date(),
                ionogram_data['foF2'].values[start_index:end_index + 1],
                slope=True,
                hmf_artist=ionogram_data['hmF2'].values[start_index:end_index + 1],
                confidence_scores=ionogram_data['cs'].values[start_index:end_index + 1],
                window_stats=True,
                mid_index=mid_index - start_index)
            if np.any(np.isnan(feature_vector)):
                print('Nans in feature vector')
                feature_vectors.append([])
            else:
                feature_vectors.append(feature_vector)
    return feature_vectors


def denm3_to_freqhz(den):
    """Convert electron density (m^-3) to frequency (Hz)"""
    return np.sqrt(den) * 8.978662808114986


def freqhz_to_denm3(freq):
    """Convert electron density (m^-3) to frequency (Hz)"""
    return np.square(freq / 8.978662808114986)


def parse_mainh5group(mainh5group):
    """Parse a non-plasmaline incoherent scatter radar h5 file and return a subset of the data as a pandas DataFrame."""
    # Convert the timestamps to datetimes
    timestamps = np.asarray(mainh5group['timestamps'])
    timestamps_datetime = np.asarray([datetime.datetime.utcfromtimestamp(t) for t in timestamps])
    # Get the range data
    if 'gdalt' in mainh5group.keys():
        range_km = np.asarray(mainh5group['gdalt'])
    elif 'range' in mainh5group.keys():
        range_km = np.asarray(mainh5group['range'])
    else:
        raise LookupError('Cannot extract range from h5 file')
    # See which electron density is available
    # First check m^-3 density because it is higher precision than log10(m^-3)
    if 'ne' in mainh5group['2D Parameters'].keys():
        dene = np.asarray(mainh5group['2D Parameters']['ne'])
        dene_err = np.asarray(mainh5group['2D Parameters']['dne'])
    elif 'nel' in mainh5group['2D Parameters'].keys():
        dene = 10 ** np.asarray(mainh5group['2D Parameters']['nel'])
        dene_err = 10 ** np.asarray(mainh5group['2D Parameters']['dnel'])
    elif 'popl' in mainh5group['2D Parameters'].keys():
        dene = 10 ** np.asarray(mainh5group['2D Parameters']['popl'])
        dene_err = 10 ** np.asarray(mainh5group['2D Parameters']['dpopl'])
    else:
        raise LookupError('Cannot find electron density keyword in 2D Parameters')
    # See if az/el are in the parameters
    if 'azm' in mainh5group['1D Parameters'].keys():
        az = mainh5group['1D Parameters']['azm']
    else:
        # Assume 0 azimuth if azm is not a field
        az = np.zeros(timestamps_datetime.shape)
    if 'elm' in mainh5group['1D Parameters'].keys():
        el = mainh5group['1D Parameters']['elm']
    else:
        # Assume 90 degree elevation if elm is not a field
        el = np.ones(timestamps_datetime.shape) * 90
    # Calculate the maximum (over range) electron density (and error), and the height it occurs for each time step
    nmf = np.asarray(
        [np.nanmax(dene[:, i]) if np.any(np.isfinite(dene[:, i])) else np.nan for i in range(dene.shape[1])])
    dnmf = np.asarray([dene_err[np.nanargmax(dene[:, i]), i] if np.any(np.isfinite(dene[:, i])) else np.nan for i in
                       range(dene.shape[1])])
    hmf = np.asarray(
        [range_km[np.nanargmax(dene[:, i])] if np.any(np.isfinite(dene[:, i])) else np.nan for i in range(dene.shape[1])])
    # Convert the maximum electron density to plasma frequency in MHz
    fof2 = denm3_to_freqhz(nmf)*1e-6
    array_vals_dict = {'datetime': timestamps_datetime,
                       'nmf': nmf,
                       'dnmf': dnmf,
                       'hmf': hmf,
                       'fof2': fof2,
                       'az': az,
                       'el': el}
    return pd.DataFrame.from_dict(array_vals_dict)


def parse_plasmaline_h5data(h5data):
    """Parse an incoherent scatter radar plasmaline h5 file and return a subset of the data as a pandas DataFrame."""
    plasma_line_table = h5data['Data']['Table Layout']
    timestamps_datetime = [datetime.datetime(row[0], row[1], row[2], row[3], row[4], row[5]) for row in
                           plasma_line_table]
    data_params = [v[0] for v in h5data['Metadata']['Data Parameters']]
    if b'NEMAX' in data_params:
        nm_col = [v for v in range(len(h5data['Metadata']['Data Parameters'])) if
                  h5data['Metadata']['Data Parameters'][v][0] == b'NEMAX'][0]
        nmfs = [row[nm_col] for row in plasma_line_table]
        dnm_col = [v for v in range(len(h5data['Metadata']['Data Parameters'])) if
                   h5data['Metadata']['Data Parameters'][v][0] == b'DNEMAX'][0]
        dnmfs = [row[dnm_col] for row in plasma_line_table]
    elif b'NEL' in data_params:
        nm_col = [v for v in range(len(h5data['Metadata']['Data Parameters'])) if
                  h5data['Metadata']['Data Parameters'][v][0] == b'NEL'][0]
        nmfs = 10**np.asarray([row[nm_col] for row in plasma_line_table])
        dnm_col = [v for v in range(len(h5data['Metadata']['Data Parameters'])) if
                   h5data['Metadata']['Data Parameters'][v][0] == b'DNEL'][0]
        dnmfs = 10**np.asarray([row[dnm_col] for row in plasma_line_table])
    elif b'POPL' in data_params:
        nm_col = [v for v in range(len(h5data['Metadata']['Data Parameters'])) if
                  h5data['Metadata']['Data Parameters'][v][0] == b'POPL'][0]
        nmfs = 10**np.asarray([row[nm_col] for row in plasma_line_table])
        dnm_col = [v for v in range(len(h5data['Metadata']['Data Parameters'])) if
                   h5data['Metadata']['Data Parameters'][v][0] == b'DPOPL'][0]
        dnmfs = 10**np.asarray([row[dnm_col] for row in plasma_line_table])
    else:
        raise AttributeError('No electron density variable found in h5 file')
    if b'HMAX' in data_params:
        hm_col = [v for v in range(len(h5data['Metadata']['Data Parameters'])) if
                  h5data['Metadata']['Data Parameters'][v][0] == b'HMAX'][0]
        hmfs = [row[hm_col] for row in plasma_line_table]
    elif b'RANGE' in data_params:
        hm_col = [v for v in range(len(h5data['Metadata']['Data Parameters'])) if
                  h5data['Metadata']['Data Parameters'][v][0] == b'RANGE'][0]
        hmfs = [row[hm_col] for row in plasma_line_table]
    else:
        raise AttributeError('No hmf variable found in h5 file')
    if b'FOF2' in data_params:
        fof2_col = [v for v in range(len(h5data['Metadata']['Data Parameters'])) if
                    h5data['Metadata']['Data Parameters'][v][0] == b'FOF2'][0]
        fof2s = [row[fof2_col] for row in plasma_line_table]
    else:
        fof2s = denm3_to_freqhz(nmfs)*1e-6
    az_col = [v for v in range(len(h5data['Metadata']['Data Parameters'])) if
              h5data['Metadata']['Data Parameters'][v][0] == b'AZM'][0]
    azs = [row[az_col] for row in plasma_line_table]
    el_col = [v for v in range(len(h5data['Metadata']['Data Parameters'])) if
              h5data['Metadata']['Data Parameters'][v][0] == b'ELM'][0]
    els = [row[el_col] for row in plasma_line_table]
    array_vals_dict = {'datetime': timestamps_datetime,
                       'nmf': nmfs,
                       'dnmf': dnmfs,
                       'hmf': hmfs,
                       'fof2': fof2s,
                       'az': azs,
                       'el': els}
    return pd.DataFrame.from_dict(array_vals_dict)


def hdf5_to_dataframe(filename):
    """Parse an incoherent scatter radar h5 file and return a subset of data as a pandas DataFrame."""
    dataframe = pd.DataFrame(columns=['datetime', 'nmf', 'dnmf', 'hmf', 'fof2', 'az', 'el'])
    # Load in the data
    h5data = h5py.File(filename, 'r')
    if 'Array Layout' in h5data['Data'].keys():
        array_layout_keys = list(h5data['Data']['Array Layout'].keys())
        if 'timestamps' in array_layout_keys:
            dataframe = pd.concat([dataframe, parse_mainh5group(h5data['Data']['Array Layout'])], ignore_index=True)
        else:
            for array_label in array_layout_keys:
                dataframe = pd.concat([dataframe, parse_mainh5group(h5data['Data']['Array Layout'][array_label])], ignore_index=True)
    else:
        # Assume a plasma line formatted h5 file
        dataframe = pd.concat([dataframe, parse_plasmaline_h5data(h5data)])
    return dataframe


def plot_isr_and_ionosonde_data(isr_data, ionosonde_data, title=None, ylabel=None, xlabel='UTC',
                                val='nmf', plot_masked=True, plot_clevel=False,
                                classifier_filename=None, window_hours=2):
    """Plot the ISR"""
    fig = plt.figure()
    ax = fig.add_subplot()
    if val == 'nmf':
        ionosonde_val = freqhz_to_denm3(np.nanmax(ionosonde_data[['fof2', 'fof1', 'foe', 'foes']].values, 1) * 1e6)
        isr_val = isr_data['nmf']
        isr_val_err = isr_data['dnmf']
        make_semilogy = True
    elif val == 'fof2':
        ionosonde_val = np.nanmax(ionosonde_data[['fof2', 'fof1', 'foe', 'foes']].values, 1)
        isr_val = np.asarray(isr_data['fof2'])
        isr_val_err = (denm3_to_freqhz(np.asarray(isr_data['nmf']) + np.asarray(isr_data['dnmf'])) - denm3_to_freqhz(
            np.asarray(isr_data['nmf']) - np.asarray(isr_data['dnmf']))) / 2 * 1e-6
        make_semilogy = False
    else:
        raise ValueError('Unknown val input value: {}'.format(val))
    ax.plot(isr_data['datetime'], isr_val, color='tab:red', label='ISR')
    ax.fill_between(isr_data['datetime'], isr_val-isr_val_err, isr_val+isr_val_err,
                    alpha=0.5, color='tab:red', label='ISR Error')
    # ax.plot(isr_data['datetime'], isr_val, label='ISR')
    if make_semilogy:
        ax.set_yscale('log')
    if len(ionosonde_data['datetime']) > 0:
        ax.plot(ionosonde_data['datetime'], ionosonde_val, label='ARTIST')
        if plot_masked:
            station_id = np.unique(ionosonde_data.station_id)[0]
            iono_params = get_iono_params(station_id, ['foF2', 'hmF2'],
                                          pd.to_datetime(ionosonde_data.datetime.values[0]),
                                          pd.to_datetime(ionosonde_data.datetime.values[-1]))
            times_julian = pd.to_datetime(iono_params['datetime'].values).to_julian_date()
            timestamps_rounded = pd.to_datetime(iono_params['datetime'].values).round('1s')
            # Download data to get confidence scores
            fof2_artist = iono_params['foF2'].values
            hmf2_artist = iono_params['hmF2'].values
            is_outlier = np.zeros(fof2_artist.shape, dtype='bool')
            # Get the feature vectors for the data
            # Load the classifier
            if classifier_filename is None:
                classifier_filename = os.path.join('classifiers', 'scaler_and_classifiers_and_betas_27_5_10.sav')
            with open(classifier_filename, 'rb') as fid:
                [scaler, classifiers, betas] = pickle.load(fid)
                good_classifier_index = 0
                classifier = classifiers[good_classifier_index]
            for i in range(len(fof2_artist)):
                # Calculate the start and end index using 1 hour window each side
                start_index = np.min(np.where(timestamps_rounded >= timestamps_rounded[i] -
                                              datetime.timedelta(hours=window_hours / 2))[0])
                mid_index = i
                end_index = np.max(np.where(timestamps_rounded <= timestamps_rounded[i] +
                                            datetime.timedelta(hours=window_hours / 2))[0])
                if start_index + 1 < mid_index < end_index - 2:
                    # get the features
                    feature_vector = make_feature_vector(
                        pd.to_datetime(iono_params['datetime'].values[start_index:end_index + 1]).to_julian_date(),
                        fof2_artist[start_index:end_index+1],
                        slope=True,
                        hmf_artist=hmf2_artist[start_index:end_index+1],
                        confidence_scores=iono_params['cs'].values[start_index:end_index+1],
                        window_stats=True,
                        mid_index=mid_index-start_index).reshape(1, -1)
                    feature_vector_scaled = scaler.transform(feature_vector)
                    if np.any(np.isnan(feature_vector_scaled)):
                        is_outlier[i] = True
                    else:
                        is_outlier[i] = classifier.predict(feature_vector_scaled)[0]
            if val == 'fof2':
                val_to_plot = fof2_artist
            elif val == 'nmf':
                val_to_plot = freqhz_to_denm3(fof2_artist*1e6)
            else:
                raise ValueError('Unknown val input value: {}'.format(val))
            ax.plot(iono_params['datetime'][~is_outlier], val_to_plot[~is_outlier], 'b.', label='Good Estimate')
            ax.plot(iono_params['datetime'][is_outlier], val_to_plot[is_outlier], 'r.', label='Bad Estimate')

    if plot_clevel:
        ax2 = ax.twinx()
        ax2.plot(ionosonde_data['datetime'], np.floor(ionosonde_data['clevel']/10), label='C-Level 1')
        ax2.plot(ionosonde_data['datetime'], ionosonde_data['clevel'] % 10, label='C-Level 2')
    ax.legend()
    ax.set_xlim([min(pd.concat([ionosonde_data['datetime'], isr_data['datetime']])),
                 max(pd.concat([ionosonde_data['datetime'], isr_data['datetime']]))])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid('on')
    return ax


def plot_station_data(station_data, classifier=None, scaler=None, val='fof2', window_hours=2, ylabel=None,
                      title=None, xlabel=None):
    fig = plt.figure()
    ax = fig.add_subplot()
    if val == 'nmf':
        ionosonde_val = freqhz_to_denm3(np.nanmax(station_data[['fof2', 'fof1', 'foe', 'foes']].values, 1) * 1e6)
        make_semilogy = True
        if ylabel is None:
            ylabel = 'nmF [m$^{-3}$]'
    elif val == 'fof2':
        ionosonde_val = np.nanmax(station_data[['fof2', 'fof1', 'foe', 'foes']].values, 1)
        make_semilogy = False
        if ylabel is None:
            ylabel = 'foF2 [MHz]'
    else:
        raise ValueError('Unknown val input value: {}'.format(val))
    if make_semilogy:
        ax.set_yscale('log')
    if classifier is None:
        ax.plot(station_data['datetime'], ionosonde_val)
    else:
        # Make the feature vectors for each data point
        timestamps_rounded = pd.to_datetime(station_data['datetime'].values).round('1s')
        fof2_artist = np.nanmax(station_data[['fof2', 'fof1', 'foe', 'foes']].values, 1)
        hmf2_artist = station_data['hmf2'].values
        is_outlier = np.zeros(fof2_artist.shape, dtype='bool')
        for i in range(len(ionosonde_val)):
            # Calculate the start and end index using 1 hour window each side
            start_index = np.min(np.where(timestamps_rounded >= timestamps_rounded[i] -
                                          datetime.timedelta(hours=window_hours / 2))[0])
            mid_index = i
            end_index = np.max(np.where(timestamps_rounded <= timestamps_rounded[i] +
                                        datetime.timedelta(hours=window_hours / 2))[0])
            if start_index + 1 < mid_index < end_index - 2:
                # get the features
                feature_vector = make_feature_vector(
                    pd.to_datetime(station_data['datetime'].values[start_index:end_index + 1]).to_julian_date(),
                    fof2_artist[start_index:end_index + 1],
                    slope=True,
                    hmf_artist=hmf2_artist[start_index:end_index + 1],
                    confidence_scores=station_data['cs'].values[start_index:end_index + 1],
                    window_stats=True,
                    mid_index=mid_index - start_index).reshape(1, -1)
                feature_vector_scaled = scaler.transform(feature_vector)
                if np.any(np.isnan(feature_vector_scaled)):
                    is_outlier[i] = True
                else:
                    is_outlier[i] = classifier.predict(feature_vector_scaled)[0]
        if val == 'fof2':
            val_to_plot = fof2_artist
        elif val == 'nmf':
            val_to_plot = freqhz_to_denm3(fof2_artist*1e6)
        else:
            raise ValueError('Unknown val input value: {}'.format(val))
        ax.plot(station_data['datetime'][~is_outlier], val_to_plot[~is_outlier], 'b.', label='Good Estimate')
        ax.plot(station_data['datetime'][is_outlier], val_to_plot[is_outlier], 'r.', label='Bad Estimate')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid()


def plot_world_grid(results, projection=ccrs.PlateCarree(), lon_span=10, lat_span=5, val_to_plot='fof2_error',
                    title=None, borders=False, coastline=True):
    fig_data = new_map_figure(borders=borders, coastline=coastline, title=title, projection=projection)


# Create a new figure generating function
def new_map_figure(borders=False, coastline=True, title=None, projection=ccrs.PlateCarree()):
    fig = plt.figure(figsize=(13.4, 5.4))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    if borders:
        ax.add_feature(cf.BORDERS, color='gray')
    if coastline:
        ax.add_feature(cf.COASTLINE, color='gray')
    ax.set_global()
    if title is not None:
        ax.set_title(title)
    figure_data = {'fig': fig, 'ax': ax, 'lats_lons': []}
    return figure_data


# Use PlateCarre projection by default since it is equi-rectangluar
def plot_timeseries_on_world(results, projection=ccrs.PlateCarree(), lon_span=10, lat_span=5, val_to_plot='fof2_error',
                             title=None, borders=False, coastline=True, start_at_station=False, station_size=6,
                             station_colors='r', timeseries_colors='b', show_ids=False, allow_overlap=True):
    """Plot the times series data on a world map
    results: pandas DataFrame containing data
    projection: The map projection you want to use
    lon_span: Double, the maximum span in longitude (x) the time series data will be scaled to match
    lat_span: Double, The maximum span in latitude (y) the time series data will be scaled to match
    val_to_plot: A string containing the column header of the results DataFrame used as the time series data
    title: Title for the plot
    borders: Boolean, plot country borders on the map
    coastline: Boolean, plot coastline on the map
    start_at_station: Boolean, station latitude equals first time series value (true) or 0 (false)
    station_size: Marker size of a station
    station_color: Color of station marker
    timeseries_color: Color of timeseries data
    show_ids: Boolean, add station ids to the plot
    allow_overlap: Boolean, allow station data to overlap on map
    """
    if title is None:
        title = val_to_plot
    # Create the first figure and axes add to list
    figs_data = [new_map_figure(borders=borders, coastline=coastline, title=title, projection=projection)]
    fig_data = figs_data[0]
    fig = fig_data['fig']
    ax = fig_data['ax']
    # Loop through each station in the results table
    station_ids = results['station_id'].unique()
    results_max_duration_ns = (results['datetime'].max() - results['datetime'].min()).total_seconds()*1e9
    results_val_normalization = results[val_to_plot].max() - results[val_to_plot].min()
    for i_station, station_id in enumerate(station_ids):
        if type(timeseries_colors) is list:
            timeseries_color = timeseries_colors[i_station]
        else:
            timeseries_color = timeseries_colors
        if type(station_colors) is list:
            station_color = station_colors[i_station]
        else:
            station_color = station_colors
        results_station = results[results['station_id'] == station_id]
        # Ignore nans
        results_station = results_station[np.isfinite(results_station[val_to_plot])]
        if len(results_station) < 2:
            continue
        station_lat = results_station['lat'].unique()[0]
        station_lon = results_station['lon'].unique()[0]
        if station_lon > 180:
            station_lon -= 360
        # See if we want to plot multiple figures (if min_distance is defined)
        if not allow_overlap:
            # See which figure we can plot the data in
            fig_data_ind = 0
            while fig_data_ind < len(figs_data):
                make_new_fig = False
                for lat_lon in figs_data[fig_data_ind]['lats_lons']:
                    if np.abs(lat_lon[0]-station_lat) < lat_span and np.abs(lat_lon[1]-station_lon) < lon_span:
                        make_new_fig = True
                        break
                # Stop before incrementing index if we don't need to make a new figure
                if not make_new_fig:
                    break
                # Increment index
                fig_data_ind += 1
            # Create the new figure if needed
            if make_new_fig:
                figs_data.append(new_map_figure(borders=borders, coastline=coastline))
            # get the axes
            fig_data = figs_data[fig_data_ind]
            fig_data['lats_lons'].append([station_lat, station_lon])
            ax = fig_data['ax']
        # Append the lat/lon of the station plotted to the figure data dictionary
        ax.scatter(station_lon, station_lat, marker='o', c=station_color, s=station_size, transform=projection)
        # Convert times to lons
        timeseries_x = station_lon + ((results_station['datetime'] - results['datetime'].min()).values.astype(
            np.double)) / results_max_duration_ns * lon_span
        timeseries_y = station_lat + (results_station[val_to_plot]/results_val_normalization*lat_span).values
        # y shift data so that data start at station latitude
        if start_at_station:
            timeseries_y -= (timeseries_y[0] - station_lat)
        ax.plot(timeseries_x, timeseries_y, timeseries_color, transform=projection, linewidth=0.5)
        # Connecting timeseries line to station
        ax.plot([station_lon, timeseries_x[0]], [station_lat, timeseries_y[0]], station_color)
        if show_ids:
            ax.axes.text(station_lon, station_lat, station_id, horizontalalignment='right',
                         verticalalignment='center')


# This function will make a movie using the plot_errors_on_world function
def make_movie(results, clim=[-4, 4], movie_name='animation.gif', is_binned=False, val='fof2_error', gridlines=True):
    fig = plt.figure(figsize=(13.4, 5.4))

    # Define the animation function inside so that we can use the figure
    def movie_func(i, results, clim, fig, is_binned=False, val='fof2_error', gridlines=True):
        unique_datetimes = results['datetime'].unique()
        datetime_to_plot = datetime.datetime.utcfromtimestamp(unique_datetimes[i].astype(datetime.datetime) * 1e-9)
        title = datetime_to_plot.strftime('SAMI3 fOF2 Errors for %Y-%m-%d %H:%M:%S UTC')
        plot_vals_on_world(results[results['datetime'] == datetime_to_plot], title=title, clim=clim, fig=fig,
                           is_binned=is_binned, val=val, gridlines=gridlines)

    my_animation = animation.FuncAnimation(fig, movie_func, frames=range(len(results['datetime'].unique())),
                                           fargs=(results, clim, fig, is_binned, val, gridlines), repeat=False)

    # Save the movie in the appropriate format (.gif if specified)
    if movie_name.endswith('.gif'):
        animation_writer = animation.PillowWriter(fps=1)
    else:
        animation_writer = animation.FFMpegWriter(fps=1)
    my_animation.save(movie_name, writer=animation_writer)
    my_animation.pause()


# This function will plot some values (default fof2 errors) on a world map
def plot_vals_on_world(results, title='', clim=None, fig=None, cmap=plt.get_cmap('bwr'), is_binned=False,
                         val='fof2_error', gridlines=False):
    if fig is None:
        fig = plt.figure(figsize=(13.4, 5.4))
        # Move the figure to top left corner
        move_figure(fig, 0, 0)
    else:
        # Clear the figure
        fig.clf()
    # Set the clim to the max and min by default
    if clim is None:
        clim = np.array([-1, 1])*np.max([abs(results['fof2_error'].min()), abs(results['fof2_error'].max())])
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cf.BORDERS, color='gray')
    ax.add_feature(cf.COASTLINE, color='gray')
    if gridlines:
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    if is_binned:
        # Generate lat/lon mesh
        nlats = 19
        nlons = 37
        lon_array = np.linspace(-180, 180, nlons)
        lat_array = np.linspace(-90, 90, nlats)
        [lat_mesh, lon_mesh] = np.meshgrid(lat_array, lon_array)
        values_mesh = [[[] for i in range(nlats-1)] for j in range(nlons-1)]
        # Find the indeces where to place the lat/lon bins
        for index, row in results.iterrows():
            lon_index = np.where(row['lon'] - lon_array < 0)[0][0] - 1
            lat_index = np.where(row['lat'] - lat_array < 0)[0][0] - 1
            values_mesh[lon_index][lat_index].append(row[val])
        # Loop through values_mesh and get the mean values
        value_mesh = np.ones(lat_mesh.shape)*np.nan
        for lon_index in range(nlons-1):
            for lat_index in range(nlats-1):
                if len(values_mesh[lon_index][lat_index]) > 0:
                    value_mesh[lon_index, lat_index] = np.mean(values_mesh[lon_index][lat_index])
        sh = ax.pcolormesh(lon_array, lat_array, value_mesh.T, transform=ccrs.PlateCarree(),
                           vmin=clim[0], vmax=clim[1], cmap=cmap)
    else:
        sh = ax.scatter(results['lon'], results['lat'], c=results['fof2_error'], vmin=clim[0], vmax=clim[1], cmap=cmap,
                        s=np.abs(results['fof2_error']) / np.array(results['fof2_measured']) * 100, edgecolors='k')
        # make legend with dummy points
        for a in [25, 50, 100]:
            ax.scatter([], [], c='k', alpha=0.5, s=a, label=str(a) + '%')
        ax.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='lower left')
    # create colorbar and legend
    plt.colorbar(sh, label='Error (MHz)')
    ax.set_title(title)
    return ax


# This function will move a matplotlib figure's upper left corner to pixel (x, y)
# Python makes this complicated because there are multiple backends for matplotlib
def move_figure(f, x, y):
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


def get_station_timeseries_vals(results, val='fof2_error'):
    # Get the station ids
    station_ids = results['station_id'].unique()
    # Loop through station ids and get time series data for each station
    val_timeseries = pd.DataFrame(columns=['station_id', 'datetime', val])
    for station_id in station_ids:
        # get the fof2 and time for the station id
        station_results = results[results['station_id'] == station_id]

        for result in results:
            if station_id in result['station_ids']:
                # Get the index of station_id
                ind = np.where([r == station_id for r in result['station_ids']])[0]
                val_timeseries = val_timeseries.append({'station_id': station_id,
                                                        'datetime': result['datetime'],
                                                        val: result[val][ind[0]]})
                val_timeseries[station_id]['datetime'] = \
                    np.append(val_timeseries[station_id]['datetime'], result['datetime'])
                val_timeseries[station_id][val] = \
                    np.append(val_timeseries[station_id][val], result[val][ind[0]])
    return val_timeseries


def make_timeseries_color_plot(results, color_val='fof2_measured', y_val='lon', y_label=None,
                               cb_title=None, title=None, clim=None, station_labels_on=True,
                               markersize=5, cmap=plt.get_cmap('viridis')):
    # Make sure that the color_val and y_val are valid keys in results table
    for val in [color_val, y_val]:
        if val not in results.keys():
            print(val + ' not a valid key')
            return
    # Set default values for colorbar
    if clim is None:
        clim = np.percentile(results[color_val].dropna(), [2.5, 97.5])
    if cb_title is None:
        cb_title = color_val
    if title is None:
        title = results['datetime'].min().strftime('%Y-%m-%d')
    if y_label is None:
        y_label = y_val

    # Get all station ids
    station_ids = results['station_id'].unique()

    # Extract val from all stations
    y_vals = np.array([])
    for station_id in station_ids:
        y_vals = np.append(y_vals, results[results['station_id'] == station_id][y_val].unique())

    # Generate figure with labels
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.grid('on')
    ax.set_xlabel('Time (UTC)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylabel(y_label)
    ax.set_title(title)

    for istation, station_id in enumerate(station_ids):
        results_subset = results[results['station_id'] == station_id]
        # Ignore nans
        results_subset = results_subset[np.isfinite(results_subset[color_val])]
        # Plot with the yshift
        # ax.plot(results_subset['datetime'], results_subset[y_val], '--k', linewidth=0.5)
        sh = ax.scatter(results_subset['datetime'], results_subset[y_val], markersize,
                        results_subset[color_val], vmin=clim[0], vmax=clim[1], cmap=cmap)
        # Add text to plot to label the station id
        if station_labels_on:
            ax.axes.text(np.max(results_subset['datetime']) + datetime.timedelta(hours=0.25),
                         results_subset[y_val].unique(), station_id, horizontalalignment='left',
                         verticalalignment='center')
        if istation == 0:
            ch = plt.colorbar(sh)
            if cb_title is not None:
                ch.ax.set_ylabel(cb_title)


def make_timeseries_plots(results, val_to_plot='fof2_error', ybuffer=5, yticks_on=False, max_stations_per_plot=None,
                          clim=None, plot_zero_line=False):
    # Order the stations by longitude
    station_ids = results['station_id'].unique()
    # Put all stations on a single plot by default
    if max_stations_per_plot is None:
        max_stations_per_plot = len(station_ids)

    # Extract lat and lon of all stations
    station_lats = np.array([])
    station_lons = np.array([])
    for station_id in station_ids:
        station_lats = np.append(station_lats, results[results['station_id'] == station_id]['lat'].unique())
        station_lons = np.append(station_lons, results[results['station_id'] == station_id]['lon'].unique())
    # Sort by longitude
    sorted_inds = np.argsort(station_lons)
    for istation, station_id in enumerate(station_ids[sorted_inds]):
        # Create a new figure every x images
        if istation % max_stations_per_plot == 0:
            # Make a new figure
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.grid('on')
            ax.set_xlabel('Time (UTC)')
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            # Set the ylabel
            if val_to_plot == 'val_to_plot':
                ylabel = 'fOF2 Error (MHz)'
                if clim is None:
                    clim = [-2, 2]
            elif val_to_plot == 'fof2_measured':
                ylabel = 'fOF2 Measured (MHz)'
                if clim is None:
                    clim = [2, 10]
            else:
                ylabel = val_to_plot
                if clim is None:
                    clim = [-2, 2]
            ax.set_ylabel(ylabel)
            if not yticks_on:
                ax.yaxis.set_ticklabels([])
            # Reset yshift
            yshift = 0
        results_subset = results[results['station_id'] == station_id]
        # Ignore nans
        results_subset = results_subset[np.isfinite(results_subset[val_to_plot])]
        # continue only if there are more than 1 data point
        if len(results_subset) < 2:
            continue
        # Plot with the yshift
        ax.plot(results_subset['datetime'], results_subset[val_to_plot] + yshift, 'k', linewidth=0.5)
        if plot_zero_line:
            ax.plot([results_subset['datetime'].values[0], results_subset['datetime'].values[-1]],
                    [yshift, yshift], '--k', linewidth=0.5)
        # Add text to plot to label the station id
        ax.axes.text(np.max(results_subset['datetime']) + datetime.timedelta(hours=0.25),
                     yshift, station_id, horizontalalignment='left', verticalalignment='center')
        # update the yshift
        yshift += np.max(results_subset[val_to_plot]) + ybuffer


def filter_station_outliers(data, station_id=None):
    if station_id is None:
        # Make sure that there is just a single unique station id
        if len(data['station_id'].unique()) > 1:
            raise ValueError("If station_id is not specified, there must only be a single station_id in data")
        filtered_data = copy.deepcopy(data)
    else:
        filtered_data = data[data['station_id'] == station_id]
    # Go through the time series data and
    for key in filtered_data.keys():
        if key != 'datetime' and key != 'station_id' and key != 'lat' and key != 'lon':
            print(key)
            plt.plot(filtered_data['datetime'], filtered_data[key])
    plt.plot(filtered_data['datetime'], filtered_data['fof2'])
    plt.plot(filtered_data['datetime'], filtered_data['fof1p'])


# This will build a feature vector used for the classifier
def make_feature_vector(times_julianday, fof2_artist, fof2_filtered=None, slope=True, hmf_artist=None,
                        confidence_scores=None, window_stats=True, mid_index=None):
    if mid_index is None:
        mid_index = int(len(times_julianday) / 2)
    # Check if a filtered fof2 was passed
    if fof2_filtered is None:
        # See if we want to calculate window stats or just use the results
        if window_stats:
            features = np.hstack([fof2_artist[mid_index],
                                  np.nanmean(fof2_artist[0:mid_index]),
                                  np.nanstd(fof2_artist[0:mid_index]),
                                  np.nanmean(fof2_artist[mid_index+1:]),
                                  np.nanstd(fof2_artist[mid_index+1:])])
        else:
            features = fof2_artist
    else:
        fof2_filtered_minus_artist = fof2_filtered - fof2_artist
        if window_stats:
            try:
                features = np.hstack([fof2_artist[mid_index],
                                      np.nanmean(fof2_filtered_minus_artist[0:mid_index]),
                                      np.nanstd(fof2_filtered_minus_artist[0:mid_index]),
                                      np.nanmean(fof2_filtered_minus_artist[mid_index + 1:]),
                                      np.nanstd(fof2_filtered_minus_artist[mid_index + 1:])])
            except:
                print('error with stacking features')
        else:
            features = np.hstack([fof2_filtered_minus_artist,
                                  fof2_artist[mid_index]])
    if slope:
        finite_inds = np.isfinite(fof2_artist)
        finite_inds_start = finite_inds[0:mid_index+1]
        finite_inds_end = finite_inds[mid_index:]
        if sum(finite_inds) >= 2:
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(times_julianday[finite_inds],
                                                                                 fof2_artist[finite_inds])
        else:
            slope = np.nan
            r_value = np.nan
            p_value = np.nan
            std_err = np.nan
        if sum(finite_inds_start) >= 2:
            slope_start, intercept_start, r_value_start, p_value_start, std_err_start = scipy.stats.linregress(
                times_julianday[0:mid_index + 1][finite_inds_start], fof2_artist[0:mid_index + 1][finite_inds_start])
        else:
            slope_start = np.nan
            r_value_start = np.nan
            p_value_start = np.nan
            std_err_start = np.nan
        if sum(finite_inds_end) >= 2:
            slope_end, intercept_end, r_value_end, p_value_end, std_err_end = scipy.stats.linregress(
                times_julianday[mid_index:][finite_inds_end], fof2_artist[mid_index:][finite_inds_end])
        else:
            slope_end = np.nan
            r_value_end = np.nan
            p_value_end = np.nan
            std_err_end = np.nan
        features = np.hstack([features, slope, r_value, p_value, std_err,
                              slope_start, r_value_start, p_value_start, std_err_start,
                              slope_end, r_value_end, p_value_end, std_err_end])
    if hmf_artist is not None:
        if window_stats:
            # Calculate the hmf stats
            finite_inds = np.isfinite(hmf_artist)
            if sum(finite_inds) >= 2:
                slope_hmf, intercept_hmf, r_value_hmf, p_value_hmf, std_err_hmf = scipy.stats.linregress(
                    times_julianday[finite_inds], hmf_artist[finite_inds])
            else:
                slope_hmf = np.nan
                r_value_hmf = np.nan
                p_value_hmf = np.nan
                std_err_hmf = np.nan
            features = np.hstack([features, hmf_artist[mid_index], np.nanstd(hmf_artist), np.nanmean(hmf_artist),
                                  slope_hmf, r_value_hmf, p_value_hmf, std_err_hmf])
        else:
            slope_hmf, intercept_hmf, r_value_hmf, p_value_hmf, std_err_hmf = scipy.stats.linregress(times_julianday,
                                                                                                     hmf_artist)
            features = np.hstack([features, hmf_artist, slope_hmf, r_value_hmf, p_value_hmf, std_err_hmf])
    if confidence_scores is not None:
        # Add confidence scores
        if window_stats:
            features = np.hstack([features, confidence_scores[mid_index], np.nanmin(confidence_scores[0:mid_index]),
                                  np.nanmin(confidence_scores[mid_index+1:])])
        else:
            features = np.hstack([features, confidence_scores])
    return features


# To access this servlet use format: DIDBGetValues?ursiCode=CODE&charName=NAME&fromDate=yyyy.mm.dd%20(...)%20hh:mm:ss&toDate=yyyy.mm.dd%20(...)%20hh:mm:ss
# For example: DIDBGetValues?ursiCode=DB049&charName=foF2&fromDate=2007.06.25%20(...)%2000:00:00&toDate=2007.06.26%20(...)%2000:00:00
# or: DIDBGetValues?ursiCode=DB049&charName=foF2&fromDate=2007.06.25&toDate=2007.06.26
def get_iono_params(station_id, params, start_datetime, end_datetime):
    # List of acceptable params, see https://giro.uml.edu/didbase/scaled.php for more info
    params_ok = ['foF2', 'foF1', 'foE', 'foEs', 'fbEs', 'foEa', 'foP', 'fxI', 'MUFD', 'MD', 'hF2', 'hF', 'hE', 'hEs',
                 'hEa', 'hP', 'TypeEs', 'hmF2', 'hmF1', 'zhalfNm', 'yF2', 'yF1', 'yE', 'scaleF2', 'B0', 'B1', 'D1',
                 'TEC', 'FF', 'FE', 'QF', 'QE', 'fmin', 'fminF', 'fminE', 'fminEs', 'foF2p']
    # Create the params string
    if type(params) is list:
        params_str = ''
        for param in params:
            params_str += param + ','
        params_list = params
    else:
        params_str = params
        params_list = [params]
    # Validate each parameter
    for param in params_list:
        if param not in params_ok:
            raise(ValueError('Invalid parameter: ' + param))

    url = 'https://lgdc.uml.edu/common/DIDBGetValues?ursiCode={}&charName={}&fromDate={}&toDate={}'.\
        format(station_id, params_str, start_datetime.strftime('%Y.%m.%d+%H:%M:%S'), end_datetime.strftime('%Y.%m.%d+%H:%M:%S'))
    # Download the data
    print('Downloading data from: {}'.format(url))
    page = requests.get(url)
    page_lines = page.text.splitlines()
    past_header = False
    # Create a pandas table to hold data
    results = pd.DataFrame(columns=['station_id', 'lat', 'lon', 'datetime', 'cs', *params_list])
    for line in page_lines:
        if not past_header:
            if line[0:5] == '#Time':
                line_header = line.split()
                past_header = True
            elif line[0:10] == '# Location':
                lat_str = line.split()[3]
                station_lat = float(lat_str[0:-1])
                if lat_str[-1] == 'S':
                    station_lat = -station_lat
                lon_str = line.split()[4][0:-1]
                station_lon = float(lon_str[0:-1])
                if lon_str[-1] == 'W':
                    station_lon = -station_lon
        else:
            # Extract the data
            line_split = line.split()
            line_dict = {'station_id': station_id,
                         'lat': station_lat,
                         'lon': station_lon,
                         'datetime': datetime.datetime.strptime(line_split[0], '%Y-%m-%dT%H:%M:%S.%fZ'),
                         'cs': float(line_split[1])}
            param_num = 0
            for i, line_el in enumerate(line_split):
                # Skip the first and last elements
                if i <= 1 or i == (len(line_split)-1) or line_header[i] == 'QD':
                    continue
                # Parse the value
                try:
                    line_dict[line_header[i]] = float(line_el)
                except ValueError:
                    line_dict[line_header[i]] = np.nan
                param_num += 1
            results = results.append(line_dict, ignore_index=True)
    return results


def plot_foes_vs_time(station_ids, start_time, end_time, plot_den=False):
    if type(station_ids) is str:
        station_ids = [station_ids]
    fig, ax = plt.subplots()
    for station_id in station_ids:
        station_data = get_iono_params(station_id, ['foEs'], start_time, end_time)
        if plot_den:
            ax.semilogy(station_data['datetime'], freqhz_to_denm3(station_data['foEs']*1e6), '.', label=station_id)
            ylabel = 'Es Density [m$^{-3}$]'
        else:
            ax.plot(station_data['datetime'], station_data['foEs'], '.', label=station_id)
            ylabel = 'Es Frequency [MHz]'
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.legend()
    ax.grid('on')
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel(ylabel)
    return ax


def make_2d_time_plot(values, use_local_time=True, only_hours=False, yval='fof2_measured',
                      cval='lat', title=None, vmin=None, vmax=None):
    val_label_dict = {'fof2_measured': 'foF2 Measured (MHz)',
                      'fof2_error': 'foF2 Error (MHz)',
                      'lat': 'Latitude (deg)',
                      'lon': 'Longitude (deg)',
                      'fof2_error_percent': 'foF2 Error Percent',
                      'fof2_sami3': 'foF2 SAMI3 (MHz)'}
    if use_local_time:
        time_array = pd.to_datetime(values['datetime'].values) + np.asarray(
            [datetime.timedelta(days=lon_val / 360) for lon_val in values['lon'].values])
        xlabel = 'Local Time'
    else:
        time_array = pd.to_datetime(values['datetime'].values)
        xlabel = 'UTC'
    if only_hours:
        time_array = time_array.hour + time_array.minute/60 + time_array.second/3600
    ylabel = val_label_dict[yval]
    clabel = val_label_dict[cval]
    if 'fof2_error_percent' == yval:
        yvals_to_plot = values['fof2_error'].values/values['fof2_measured'].values*100
    elif 'fof2_sami3' == yval:
        yvals_to_plot = values['fof2_error'].values + values['fof2_measured'].values
    else:
        yvals_to_plot = values[yval].values
    if 'fof2_error_percent' == cval:
        cvals_to_plot = values['fof2_error'].values / values['fof2_measured'].values * 100
    elif 'fof2_sami3' == cval:
        cvals_to_plot = values['fof2_error'].values + values['fof2_measured'].values
    else:
        cvals_to_plot = values[cval].values
    fig = plt.figure()
    ax = fig.add_subplot()
    p = ax.scatter(time_array, yvals_to_plot, s=2, c=cvals_to_plot, vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel, fontdict={'fontsize': 16})
    ax.set_ylabel(ylabel, fontdict={'fontsize': 16})
    ax.set_title(title, fontdict={'fontsize': 16})
    cba = fig.colorbar(p)
    cba.ax.set_ylabel(clabel, fontdict={'fontsize': 14})
    if not only_hours:
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid('on')
    return ax


def make_sami_artist_line_plot_single_lat(values, use_local_time=True, only_hours=False):
    ylabel = 'foF2 (MHz)'
    if use_local_time:
        time_array = pd.to_datetime(values['datetime'].values) + np.asarray(
            [datetime.timedelta(days=lon_val / 360) for lon_val in values['lon'].values])
        xlabel = 'Local Time'
    else:
        time_array = pd.to_datetime(values['datetime'].values)
        xlabel = 'UTC'
    if only_hours:
        time_array = time_array.hour + time_array.minute/60 + time_array.second/3600
    sami_fof2 = values['fof2_error'].values + values['fof2_measured'].values
    artist_fof2 = values['fof2_measured'].values
    lats = list(set(values['lat']))
    for lat in lats:
        fig = plt.figure()
        ax = fig.add_subplot()
        inds = values['lat'] == lat
        title = 'SAMI3 and ARTIST for {} deg Latitude'.format(lat)
        ax.plot(time_array[inds], sami_fof2[inds], '.', label='SAMI3')
        ax.plot(time_array[inds], artist_fof2[inds], '.', label='ARTIST')
        if not only_hours:
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.grid('on')
        ax.set_xlabel(xlabel, fontdict={'fontsize': 16})
        ax.set_ylabel(ylabel, fontdict={'fontsize': 16})
        ax.set_title(title, fontdict={'fontsize': 16})
        ax.legend()
    return ax
