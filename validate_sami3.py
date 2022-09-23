import argparse
import os
import sys
import json
import netCDF4 as nc
import numpy as np
import geopy.distance
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import datetime
import pickle
import pandas as pd
import utils


# Set physical constants
eps = 8.85418782e-12
me = 9.10938356e-31
elem_charge = 1.60217662e-19

# Set the maximum allowed time threshold between sami and ionogram
max_time_thresh_sec = 10*60


def _build_arg_parser(Parser, *args):
    """Arugment parser for the main script
    """
    scriptname = os.path.basename(sys.argv[0])

    formatter = argparse.RawDescriptionHelpFormatter(scriptname)
    width = formatter._width

    title = "extract_parameters"
    copyright = "Copyright (c) 2021 JHU/APL"
    shortdesc = "Validate sami3 with ionogram data."
    desc = "\n".join(
        (
            "*" * width,
            "*{0:^{1}}*".format(title, width - 2),
            "*{0:^{1}}*".format(copyright, width - 2),
            "*{0:^{1}}*".format("", width - 2),
            "*{0:^{1}}*".format(shortdesc, width - 2),
            "*" * width,
        )
    )

    usage = (
        "%s [-s sami_input] [-i ionogram_file] [-o output_file] [-m movie_name] [-p make_plots] [-ts timeseries_plot]" % scriptname
    )

    # parse options
    parser = Parser(
        description=desc,
        usage=usage,
        prefix_chars="-+",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-s",
        "--sami_input",
        dest="sami_input",
        default=None,
        help="""Sami file to analyze""",
    )

    parser.add_argument(
        "-i",
        "--ionogram_file",
        dest="ionogram_file",
        default=None,
        help="""Ionogram file to use for data validation""",
    )

    parser.add_argument(
        "-o",
        "--output_file",
        dest="output_file",
        default=None,
        help="""File to save the output""",
    )

    parser.add_argument(
        "-p",
        "--plot",
        dest="make_plots",
        action="store_true",
        help="""Generate plots. (default: False)""",
    )

    parser.add_argument(
        "-m",
        "--movie",
        dest="movie_name",
        default=None,
        help="""File to save the movie.""",
    )

    parser.add_argument(
        "-ts",
        "--timeseries",
        dest="timeseries_plot",
        action="store_true",
        help="""Generate time series plots. (default: False)""",
    )

    return parser


def ionogram_json_to_table(ionogram_dict):
    # Create a pandas DataFrame table from ionogram data json
    table_columns = list(ionogram_dict[list(ionogram_dict.keys())[0]].keys())
    table_columns.append('station_id')
    table_columns.append('datetime')
    ionogram_table = pd.DataFrame(columns=table_columns)
    for key in ionogram_dict.keys():
        station_data = ionogram_dict[key]
        station_data['datetime'] = datetime.datetime.strptime(key.split('_')[1] + key.split('_')[2], '%Y%b%d%H%M%S')
        station_data['station_id'] = key.split('_')[0]
        ionogram_table = ionogram_table.append(station_data, ignore_index=True)
    return ionogram_table


# This will linearly interpolate a value using the nearest 4 latitude and longitude grid points
def interpolate_lat_lon_values(lats, lons, values, interp_lat, interp_lon):
    # Find the indeces bounding that lats and lons
    min_ind_lat = np.argmin(np.abs(lats[:] - interp_lat))
    min_ind_lon = np.argmin(np.abs(lons[:] - interp_lon))
    if lats[min_ind_lat] < interp_lat:
        lat_bound_inds = [min_ind_lat, min_ind_lat + 1]
    else:
        lat_bound_inds = [min_ind_lat - 1, min_ind_lat]
    if lons[min_ind_lon] < interp_lon:
        lon_bound_inds = [min_ind_lon, min_ind_lon + 1]
    else:
        lon_bound_inds = [min_ind_lon - 1, min_ind_lon]
    # Do a weighted average over the 4 different cells
    distances = np.zeros((2, 2))
    for ilat in range(2):
        for ilon in range(2):
            distances[ilat, ilon] = geopy.distance.distance((interp_lat, interp_lon), (
                lats[lat_bound_inds].data[ilat], lons[lon_bound_inds].data[ilon])).km
    # Make weights proportional to 1/distance (closer points have higher weight)
    dist_inv = 1 / distances
    weights = dist_inv / np.sum(dist_inv)
    if len(values.shape) == 3:
        interp_values = np.zeros((len(values),))
    elif len(values.shape) == 2:
        interp_values = 0
    for ilat in range(2):
        for ilon in range(2):
            if len(values.shape) == 3:
                interp_values += values[:, lat_bound_inds[ilat], lon_bound_inds[ilon]] * weights[ilat, ilon]
            elif len(values.shape) == 2:
                interp_values += values[lat_bound_inds[ilat], lon_bound_inds[ilon]] * weights[ilat, ilon]
    return interp_values


window_size = 9
window_hours = 2
scaler_and_classifier_filename = os.path.join('classifiers', 'scaler_and_classifiers_and_betas_27_0.1_10.sav')
filter_outliers = True

if __name__ == '__main__':
    parser = _build_arg_parser(argparse.ArgumentParser)
    args = parser.parse_args()

    # Load in the ionogram data
    if args.ionogram_file is None:
        # We want to download files for sami
        should_download_ionodat = True
    else:
        should_download_ionodat = False
        if args.ionogram_file.endswith('.csv'):
            ionogram_data = pd.read_csv(args.ionogram_file, parse_dates=['datetime'])
        else:
            with open(args.ionogram_file, 'r') as json_file:
                ionogram_data = ionogram_json_to_table(json.load(json_file))
        station_ids = ionogram_data['station_id'].unique()

    # Load in the SAMI3 data
    # See if the sami_input is a file or a directory
    if args.sami_input.endswith('.nc'):
        sami_files = [args.sami_input]
    else:
        # find all sami files in the directory
        sami_files = [os.path.join(args.sami_input, f) for f in os.listdir(args.sami_input) if f.endswith('.nc')]
    # Get all datetimes from sami files
    sami_datetimes = []
    for sami_file in sami_files:
        sami_datetimes.append(datetime.datetime.strptime(
                os.path.basename(sami_file).split('-')[0] + os.path.basename(sami_file).split('-')[1].split('_')[0],
                '%Y%j%H%M'))

    # Download data for all station_ids for sami times
    ionogram_downloaded_data = pd.DataFrame(columns=['station_id', 'datetime', 'cs', 'foF2'])
    for station_id in station_ids:
        ionogram_downloaded_data = ionogram_downloaded_data.append(
            utils.get_iono_params(station_id, ['foF2'], min(sami_datetimes), max(sami_datetimes)), ignore_index=True)
    # Merge the data
    ionogram_merged = ionogram_data.merge(ionogram_downloaded_data, how='left')

    # Load in the scaler and classifier
    with open(scaler_and_classifier_filename, 'rb') as fid:
        [scaler, classifiers, betas] = pickle.load(fid)
    # choose a single classifier if there are multiple
    if type(classifiers) is list:
        classifier = classifiers[3]
    else:
        classifier = classifiers

    # Create a pandas table to store the data for each sami3 output file
    results = pd.DataFrame(columns=['fof2_measured', 'fof2_error', 'lat', 'lon', 'station_id', 'datetime', 'cs', 'isgood'])
    fig = None
    for sami_file in sami_files:
        print('Validating sami file: ' + os.path.basename(sami_file))
        sami_datetime = datetime.datetime.strptime(
                os.path.basename(sami_file).split('-')[0] + os.path.basename(sami_file).split('-')[1].split('_')[0],
                '%Y%j%H%M')
        # Make sure there are data within x times of sami to continue
        if np.min(np.abs((ionogram_data['datetime'] - sami_datetime).values.astype('timedelta64[s]'))).astype('double') > max_time_thresh_sec:
            print('No close ionogram data found. Skipping...')
            continue
        sami_ds = nc.Dataset(sami_file)
        dene = np.squeeze(sami_ds['dene'])  # Squeeze because first dimension is time and it has just one element
        nmf2 = np.squeeze(sami_ds['nmf2'])
        # Loop through each station to find the closest data to the sami time
        for station_id in station_ids:
            # Get the minimum time difference between sami3 and the station
            ionogram_data_station = ionogram_merged[ionogram_merged['station_id'].eq(station_id)]
            station_times_julian = pd.to_datetime(ionogram_data_station['datetime'].values).to_julian_date()
            # station_fof2_filtered = utils.filter_data_highbias(ionogram_data_station['fof2'].values, window_size=window_size)
            timediffs = np.abs((ionogram_data_station['datetime'] - sami_datetime).values.astype('timedelta64[s]'))
            timediff_min = np.min(timediffs)
            timestamps_rounded = pd.to_datetime(ionogram_data_station['datetime'].values).round('1s')
            # Make sure the minimum time is within 5 minutes of sami file
            if timediff_min.astype('double') > max_time_thresh_sec:
                continue
            data_index = np.argmin(timediffs)
            # Make sure we can generate a feature vector for the data point
            if data_index < int(window_size/2) or data_index >= len(ionogram_data_station) - np.ceil(window_size/2):
                continue
            station_data = ionogram_data_station.iloc[data_index]
            # Make there there was an extracted fof2 value
            if station_data['fof2'] is np.nan or station_data['fof2'] == 'N/A':
                continue
            # Generate the feature vector for the data point
            # start_index = data_index - int(window_size / 2)
            # end_index = data_index + int(window_size / 2) + 1
            start_index = np.min(np.where(timestamps_rounded >= timestamps_rounded[data_index] -
                                          datetime.timedelta(hours=window_hours / 2))[0])
            mid_index = data_index
            end_index = np.max(np.where(timestamps_rounded <= timestamps_rounded[data_index] +
                                        datetime.timedelta(hours=window_hours / 2))[0])
            if mid_index >= end_index-2 or mid_index < start_index+2:
                continue
            feature_vector = utils.make_feature_vector(
                station_times_julian[start_index:end_index+1],
                ionogram_data_station['fof2'].values[start_index:end_index+1],
                slope=True,
                hmf_artist=ionogram_data_station['hmf2'].values[start_index:end_index+1],
                confidence_scores=ionogram_data_station['cs'].values[start_index:end_index+1],
                window_stats=True,
                mid_index=mid_index-start_index).reshape(1, -1)
            # Make sure there are no nans in the feature vector
            if np.any(np.isnan(feature_vector)):
                continue
            # Scale and classify feature vector
            if filter_outliers:
                feature_vector_scaled = scaler.transform(feature_vector)
                is_outlier = classifier.predict(feature_vector_scaled)
                if is_outlier:
                    print('Outlier: {} {}'.format(station_data['station_id'], station_data['datetime']))
                    continue

            # Fix -180 < station_data['lon'] < 180
            if station_data['lon'] > 180:
                station_data['lon'] -= 360
            # Interpolate to get the sami predicted electron density profile at the station location
            dene_prof = interpolate_lat_lon_values(np.array(sami_ds['lat']), np.array(sami_ds['lon']), dene,
                                                   station_data['lat'], station_data['lon'])
            # Get the fof2 from the dene_prof (convert from cm^3 to MHz)
            pfreq_prof = np.sqrt(dene_prof * (100 ** 3) * (elem_charge ** 2) / (me * eps)) * 1e-6 / (2*np.pi)
            fof2_sami = np.max(pfreq_prof)
            if args.make_plots:
                plt.figure()
                plt.plot(pfreq_prof, sami_ds['alt'])
                plt.grid()
                plt.title('SAMI3 Profile for {} ({}, {})\nfOF2: {:1.3f} MHz, {:1.3f} MHz'.format(station_data['station_id'],
                                                                                                 station_data['lat'],
                                                                                                 station_data['lon'],
                                                                                                 fof2_sami,
                                                                                                 station_data['fof2']))
                plt.xlabel('Plasma Frequency (MHz)')
                plt.ylabel('Altitude (km)')
            # Get the error
            results = results.append({'fof2_measured': station_data['fof2'],
                                      'fof2_error': fof2_sami - station_data['fof2'],
                                      'lat': station_data['lat'],
                                      'lon': station_data['lon'],
                                      'station_id': station_data['station_id'],
                                      'datetime': sami_datetime,
                                      'cs': station_data['cs']}, ignore_index=True)
        # Plot the errors on a map
        # if args.make_plots:
        #     if fig is None:
        #         fig = plt.figure(figsize=(13.4, 5.4))
        #         # Move the figure to top left corner
        #         move_figure(fig, 0, 0)
        #     plot_errors_on_world(results, clim=[-4, 4], fig=fig,
        #                          title='fOF2 Error (SAMI3 - Measured)\n{}'.format(os.path.basename(sami_file)))

    if args.movie_name is not None:
        print('Making movie')
        utils.make_movie(results, movie_name=args.movie_name, is_binned=True, gridlines=True)



    utils.make_sami_artist_line_plot_single_lat(results)
    utils.make_2d_time_plot(results, use_local_time=True, only_hours=False, yval='fof2_measured', cval='lat',
                      title='Measured vs Local Time\n{} - {}'.format(results.datetime.min().strftime('%Y%m%d'),
                                                                   results.datetime.max().strftime('%Y%m%d')))
    utils.make_2d_time_plot(results, use_local_time=False, only_hours=False, yval='fof2_measured', cval='lat',
                      title='Measured vs UTC Time\n{} - {}'.format(results.datetime.min().strftime('%Y%m%d'),
                                                                   results.datetime.max().strftime('%Y%m%d')))
    utils.make_2d_time_plot(results, use_local_time=False, only_hours=False, yval='fof2_error', cval='lat',
                      title='Error vs UTC\n{} - {}'.format(results.datetime.min().strftime('%Y%m%d'),
                                                           results.datetime.max().strftime('%Y%m%d')))
    utils.make_2d_time_plot(results, use_local_time=True, only_hours=False, yval='fof2_error', cval='lat',
                      title='Error vs Local Time\n{} - {}'.format(results.datetime.min().strftime('%Y%m%d'),
                                                           results.datetime.max().strftime('%Y%m%d')))
    utils.make_2d_time_plot(results, use_local_time=False, only_hours=False, yval='lat', cval='fof2_error',
                      title='Error vs UTC\n{} - {}'.format(results.datetime.min().strftime('%Y%m%d'),
                                                           results.datetime.max().strftime('%Y%m%d')))
    utils.make_2d_time_plot(results, use_local_time=True, only_hours=False, yval='lat', cval='fof2_error',
                      title='Error vs Local Time\n{} - {}'.format(results.datetime.min().strftime('%Y%m%d'),
                                                           results.datetime.max().strftime('%Y%m%d')))
    utils.make_2d_time_plot(results, use_local_time=False, only_hours=False, yval='lat', cval='fof2_measured',
                      title='Measured vs UTC\n{} - {}'.format(results.datetime.min().strftime('%Y%m%d'),
                                                           results.datetime.max().strftime('%Y%m%d')))
    utils.make_2d_time_plot(results, use_local_time=True, only_hours=False, yval='lat', cval='fof2_measured',
                      title='Measured vs Local Time\n{} - {}'.format(results.datetime.min().strftime('%Y%m%d'),
                                                           results.datetime.max().strftime('%Y%m%d')))
    utils.make_2d_time_plot(results, use_local_time=False, only_hours=False, yval='lat', cval='fof2_error_percent',
                      title='Error vs UTC\n{} - {}'.format(results.datetime.min().strftime('%Y%m%d'),
                                                           results.datetime.max().strftime('%Y%m%d')))
    utils.make_2d_time_plot(results, use_local_time=True, only_hours=False, yval='lat', cval='fof2_error_percent',
                            title='Percent Error vs Local Time\n{} - {}'.format(
                                results.datetime.min().strftime('%Y%m%d'), results.datetime.max().strftime('%Y%m%d')),
                            vmin=-75, vmax=75)
    utils.make_2d_time_plot(results, use_local_time=True, only_hours=False, yval='lat', cval='fof2_error',
                            title='Absolute Error vs Local Time\n{} - {}'.format(
                                results.datetime.min().strftime('%Y%m%d'), results.datetime.max().strftime('%Y%m%d')))

    local_times = pd.to_datetime(results['datetime'].values) + np.asarray(
        [datetime.timedelta(days=lon_val / 360) for lon_val in results['lon'].values])
    local_hours = local_times.hour + local_times.minute/60 + local_times.second/3600
    # Simple local time x axis, error y-axis
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(local_hours, results['fof2_error'], s=2)
    ax.set_xlabel('Local Time (Hour of Day)', fontdict={'fontsize': 16})
    ax.set_ylabel('foF2 Error (MHz)', fontdict={'fontsize': 16})
    ax.set_title('Error vs Local Time.: {} - {}'.format(results.datetime.min().strftime('%Y%m%d'),
                                                      results.datetime.max().strftime('%Y%m%d')),
                 fontdict={'fontsize': 18})
    ax.set_xlim(0, 24)
    ax.grid('on')

    # Simple local time x axis, measured y-axis
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(local_hours, results['fof2_measured'], s=2)
    ax.set_xlabel('Local Time (Hour of Day)', fontdict={'fontsize': 16})
    ax.set_ylabel('foF2 Measured (MHz)', fontdict={'fontsize': 16})
    ax.set_title('foF2 vs Local Time: {} - {}'.format(results.datetime.min().strftime('%Y%m%d'),
                                                      results.datetime.max().strftime('%Y%m%d')),
                 fontdict={'fontsize': 18})
    ax.set_xlim(0, 24)
    ax.grid('on')

    fig = plt.figure()
    ax = fig.add_subplot()
    p = ax.scatter(local_hours, results['fof2_error'], s=2, c=results['lat'].values)
    ax.set_xlabel('Local Time (Hour of Day)', fontdict={'fontsize': 16})
    ax.set_ylabel('foF2 Error (MHz)', fontdict={'fontsize': 16})
    ax.set_title('Error vs Local Time: {} - {}'.format(results.datetime.min().strftime('%Y%m%d'),
                                                      results.datetime.max().strftime('%Y%m%d')),
                 fontdict={'fontsize': 18})
    ax.set_xlim(0, 24)
    ax.grid('on')
    cba = fig.colorbar(p)
    cba.ax.set_ylabel('Latitude (deg)')

    fig = plt.figure()
    ax = fig.add_subplot()
    p = ax.scatter(local_hours, results['fof2_measured'], s=2, c=results['lat'].values)
    ax.set_xlabel('Local Time (Hour of Day)', fontdict={'fontsize': 16})
    ax.set_ylabel('foF2 Measured (MHz)', fontdict={'fontsize': 16})
    ax.set_title('foF2 vs Local Time: {} - {}'.format(results.datetime.min().strftime('%Y%m%d'),
                                                      results.datetime.max().strftime('%Y%m%d')),
                 fontdict={'fontsize': 18})
    ax.set_xlim(0, 24)
    ax.grid('on')
    cba = fig.colorbar(p)
    cba.ax.set_ylabel('Latitude (deg)')

    fig = plt.figure()
    ax = fig.add_subplot()
    p = ax.scatter(local_hours, results['lat'], s=2, c=results['fof2_error'].values)
    ax.set_xlabel('Local Time (Hour of Day)', fontdict={'fontsize': 16})
    ax.set_ylabel('Latitude (deg)', fontdict={'fontsize': 16})
    ax.set_title('Latitude vs Local Time\n{} - {}'.format(results.datetime.min().strftime('%Y%m%d'),
                                                          results.datetime.max().strftime('%Y%m%d')),
                 fontdict={'fontsize': 18})
    ax.set_xlim(0, 24)
    ax.grid('on')
    cba = fig.colorbar(p)
    cba.ax.set_ylabel('foF2 Error (deg)')

    fig = plt.figure()
    ax = fig.add_subplot()
    p = ax.scatter(local_hours, results['lat'], s=2, c=results['fof2_measured'].values)
    ax.set_xlabel('Local Time (Hour of Day)', fontdict={'fontsize': 16})
    ax.set_ylabel('Latitude (deg)', fontdict={'fontsize': 16})
    ax.set_title('Latitude vs Local Time\n{} - {}'.format(results.datetime.min().strftime('%Y%m%d'),
                                                          results.datetime.max().strftime('%Y%m%d')),
                 fontdict={'fontsize': 18})
    ax.set_xlim(0, 24)
    ax.grid('on')
    cba = fig.colorbar(p)
    cba.ax.set_ylabel('foF2 Measured (deg)')

    fig = plt.figure()
    ax = fig.add_subplot()
    p = ax.scatter(results['datetime'], results['lat'], s=2, c=results['fof2_measured'].values)
    ax.set_xlabel('UTC', fontdict={'fontsize': 16})
    ax.set_ylabel('Latitude (deg)', fontdict={'fontsize': 16})
    ax.set_title('Latitude vs UTC\n{} - {}'.format(results.datetime.min().strftime('%Y%m%d'),
                                                          results.datetime.max().strftime('%Y%m%d')),
                 fontdict={'fontsize': 18})
    ax.grid('on')
    cba = fig.colorbar(p)
    cba.ax.set_ylabel('foF2 Measured (deg)')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    p = ax.scatter(local_hours, results['lat'].values, results['fof2_error'], s=1, c=results['fof2_error'])
    ax.set_xlabel('Local Time (Hours)')
    ax.set_ylabel('Latitude (deg)')
    ax.set_zlabel('foF2 Error (MHz)')
    fig.colorbar(p)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(local_hours, results['lat'].values, results['fof2_measured'], s=1)
    ax.set_xlabel('Local Time (Hours)')
    ax.set_ylabel('Latitude (deg)')
    ax.set_zlabel('foF2 Measured (MHz)')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #ax.scatter(results['datetime'].values, results['lat'].values, results['lon'].values, c=results['fof2_measured'])
    ax.scatter(local_hours, results['lat'].values, results['fof2_error'])
    ax.set_xlabel('Local Time (Hours)')
    ax.set_ylabel('Latitude (deg)')
    ax.set_zlabel('foF2 Error (MHz)')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #ax.scatter(results['datetime'].values, results['lat'].values, results['lon'].values, c=results['fof2_measured'])
    ax.scatter(local_hours, results['lat'].values, results['lon'].values, c=results['fof2_error'])
    ax.set_xlabel('Local Time (Hours)')
    ax.set_ylabel('Latitude (deg)')
    ax.set_zlabel('Longitude (deg)')
    plt.colorbar()

    if args.output_file is not None:
        # Save the output
        pickle.dump(results, open(args.output_file, "wb"))

    if args.timeseries_plot is not None:
        print('Making time-series plot')
        utils.make_timeseries_plots(results, val_to_plot='fof2_measured')
        utils.make_timeseries_plots(results, val_to_plot='fof2_error')
        utils.make_timeseries_color_plot(results, color_val='fof2_measured', cb_title='fOF2 Measured (MHz)',
                                         y_val='lon', y_label='Longitude (deg)',
                                         title=results['datetime'].min().strftime('%Y-%m-%d'),
                                         station_labels_on=False)
        utils.plot_timeseries_on_world(results, lon_span=40, lat_span=15, val_to_plot='fof2_error',
                                       start_at_station=False, station_size=20, show_ids=True)

    print('Done.')
