# This will plot the time series of fOF2 for ISR, ionogram, and SAMI data
import argparse
import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import pandas as pd
import utils
import pickle


def _build_arg_parser(Parser, *args):
    """Arugment parser for the main script
    """
    scriptname = os.path.basename(sys.argv[0])

    formatter = argparse.RawDescriptionHelpFormatter(scriptname)
    width = formatter._width

    title = "extract_parameters"
    copyright = "Copyright (c) 2022 JHU/APL"
    shortdesc = "Analyze Incoherent Scatter Radar (ISR) data."
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
        "%s [-s sami_input]" % scriptname
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

    return parser


def create_axes(ylabel=None, xlabel=None, title=None, grid=True):
    fig = plt.figure()
    ax = fig.add_subplot()
    if title is not None:
        ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if grid:
        ax.grid()
    return ax


def parse_plasmaline_h5data(h5data):
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
        fof2s = utils.denm3_to_freqhz(nmfs)*1e-6
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


def parse_mainh5group(mainh5group):
    timestamps = np.asarray(mainh5group['timestamps'])
    timestamps_datetime = np.asarray([datetime.datetime.utcfromtimestamp(t) for t in timestamps])
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
        az = np.zeros(timestamps_datetime.shape)
    if 'elm' in mainh5group['1D Parameters'].keys():
        el = mainh5group['1D Parameters']['elm']
    else:
        el = np.ones(timestamps_datetime.shape) * 90
    nmf = np.asarray(
        [np.nanmax(dene[:, i]) if np.any(np.isfinite(dene[:, i])) else np.nan for i in range(dene.shape[1])])
    dnmf = np.asarray([dene_err[np.nanargmax(dene[:, i]), i] if np.any(np.isfinite(dene[:, i])) else np.nan for i in
                       range(dene.shape[1])])
    hmf = np.asarray(
        [range_km[np.nanargmax(dene[:, i])] if np.any(np.isfinite(dene[:, i])) else np.nan for i in range(dene.shape[1])])
    # Compute fof2 assuming nmf = fof2
    fof2 = utils.denm3_to_freqhz(nmf)*1e-6
    array_vals_dict = {'datetime': timestamps_datetime,
                       'nmf': nmf,
                       'dnmf': dnmf,
                       'hmf': hmf,
                       'fof2': fof2,
                       'az': az,
                       'el': el}
    return pd.DataFrame.from_dict(array_vals_dict)


def hdf5_to_dataframe(filename):
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


def plot_all_data(filenames_and_descriptions, description_filter=None, val='nmf', ylabel='Electron Density [m$^{-3}$]',
                  xlabel='UTC Time', title='nmF2', semilogy=True, ax=None, xlim=None, make_legend=True,
                  use_error_bars=False, show_errors=True):
    if ax is None:
        ax = create_axes(ylabel=ylabel, xlabel=xlabel, title=title)
    for filename_and_desc in filenames_and_descriptions:
        filename = filename_and_desc[0]
        description = filename_and_desc[1]
        if description_filter is None or description_filter in description:
            print('Loading {}'.format(filename))
            data = hdf5_to_dataframe(filename)
            # Make sure the data has val in the key
            if val not in data.keys():
                continue
            # Get the maximum elevation
            max_el = data['el'].max()
            data_max_el = data[data['el'] == max_el]
            val_err = 'd'+val
            if show_errors and val_err in data.keys():
                if use_error_bars:
                    ax.errorbar(data_max_el.sort_values(by=['datetime'])['datetime'],
                                data_max_el.sort_values(by=['datetime'])[val],
                                yerr=data_max_el.sort_values(by=['datetime'])[val_err],
                                label=description)
                else:
                    ax.fill_between(data_max_el.sort_values(by=['datetime'])['datetime'],
                                    data_max_el.sort_values(by=['datetime'])[val]-data_max_el.sort_values(by=['datetime'])[val_err],
                                    data_max_el.sort_values(by=['datetime'])[val]+data_max_el.sort_values(by=['datetime'])[val_err],
                                    alpha=0.75, label=description)
            else:
                ax.plot(data_max_el.sort_values(by=['datetime'])['datetime'],
                        data_max_el.sort_values(by=['datetime'])[val], linewidth=0.5, label=description)
            if semilogy:
                ax.set_yscale('log')
    if make_legend:
        ax.legend()
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    return ax


def get_filenames_for_day(day, main_dir=os.path.join('C:\\', 'Users', 'sugargf1', 'Box', 'Data'), pfisr_dir=None,
                     mhisr_dir=None):
    """Sets the file locations for each experiment.
    Output: [pfisr_filenames, mhisr_filenames, ionogram_data_filename, pfisr_filename, mhisr_filename]
          pfisr_filenames: List of 2 element lists containing filename and description of the file
          mhisr_filenames: List of 2 element lists containing filename and description of the file
          ionogram_data_filename: String containing the filename of the ionogram_data
          pfisr_filename: The best pfisr file to use for analysis
          mhisr_filename: The best mhisr file to use for analysis
    """
    if pfisr_dir is None:
        pfisr_dir = os.path.join(main_dir, 'PFISR')
    if mhisr_dir is None:
        mhisr_dir = os.path.join(main_dir, 'MillstoneHill')
    if day == datetime.datetime(2012, 1, 14):
        ionogram_data_filename = 'ionogram_data/ionogram_parameters20120114.csv'
        pfisr_filenames = [[None, None]]
        mhisr_filenames = [[os.path.join(mhisr_dir, 'mlh120114m.005.hdf5'), 'Gridded data'],
                           [os.path.join(mhisr_dir, 'mlh120114g.003.hdf5'), 'Basic Parameters'],
                           [os.path.join(mhisr_dir, 'mlh120114i.003.hdf5'), 'Zenith single-pulse'],
                           [os.path.join(mhisr_dir, 'mlh120114j.003.hdf5'), 'Zenith alternating-code']]
        pfisr_filename = None
        mhisr_filename = mhisr_filenames[0][0]
    elif day == datetime.datetime(2012, 1, 15):
        ionogram_data_filename = 'ionogram_data/ionogram_parameters20120115.csv'
        pfisr_filenames = [[None, None]]
        mhisr_filenames = [[os.path.join(mhisr_dir, 'mlh120115m.004.hdf5'), 'Gridded data'],
                           [os.path.join(mhisr_dir, 'mlh120115g.003.hdf5'), 'Basic Parameters'],
                           [os.path.join(mhisr_dir, 'mlh120115i.003.hdf5'), 'Zenith single-pulse'],
                           [os.path.join(mhisr_dir, 'mlh120115j.003.hdf5'), 'Zenith alternating-code']]
        pfisr_filename = None
        mhisr_filename = mhisr_filenames[0][0]
    elif day == datetime.datetime(2012, 8, 2):
        # 2012-08-02 Normal day with Eielson turned on for first time in 2012, but no MHISR data
        ionogram_data_filename = 'ionogram_data/ionogram_parameters20120802.csv'
        pfisr_filenames = [[os.path.join(pfisr_dir, 'pfa120801.001.hdf5'), 'Long Pulse Uncorrected Ne'],
                           [os.path.join(pfisr_dir, 'pfa120801.002.hdf5'), 'Long Pulse (480)'],
                           [os.path.join(pfisr_dir, 'pfa120801.003.hdf5'), 'Alternating Code Uncorrected Ne'],
                           [os.path.join(pfisr_dir, 'pfa120801.004.hdf5'), 'Alternating Code (AC16-30)']]
        mhisr_filenames = [[None, None]]
        pfisr_filename = pfisr_filenames[1][0]
        mhisr_filename = None
    elif day == datetime.datetime(2012, 9, 5):
        # 2012-09-05 Normal day with Millstone Hill ISR data and good ionograms
        ionogram_data_filename = 'ionogram_data/ionogram_parameters20120905.csv'
        pfisr_filenames = [[os.path.join(pfisr_dir, 'pfa120904.001.hdf5'), 'Long Pulse Uncorrected Ne'],
                           [os.path.join(pfisr_dir, 'pfa120904.002.hdf5'), 'Long Pulse (480)'],
                           [os.path.join(pfisr_dir, 'pfa120904.003.hdf5'), 'Alternating Code Uncorrected Ne'],
                           [os.path.join(pfisr_dir, 'pfa120904.004.hdf5'), 'Alternating Code (AC16-30)']]
        mhisr_filenames = [[os.path.join(mhisr_dir, 'mlh120904m.004.hdf5'), 'Gridded data'],
                           [os.path.join(mhisr_dir, 'mlh120904g.005.hdf5'), 'Basic Parameters'],
                           [os.path.join(mhisr_dir, 'mlh120904i.006.hdf5'), 'Zenith single-pulse'],
                           [os.path.join(mhisr_dir, 'mlh120904j.006.hdf5'), 'Zenith alternating-code']]
        pfisr_filename = pfisr_filenames[1][0]
        mhisr_filename = mhisr_filenames[0][0]
    elif day == datetime.datetime(2012, 11, 28):
        # 2012-11-28 Partial day with PFISR, MHISR< and ionosonde data. Calm day
        ionogram_data_filename = 'ionogram_data/ionogram_parameters20121128.csv'
        pfisr_filenames = [[os.path.join(pfisr_dir, 'pfa121125.001.hdf5'), 'Long Pulse Uncorrected Ne'],
                           [os.path.join(pfisr_dir, 'pfa121125.002.hdf5'), 'Long Pulse (480)'],
                           [os.path.join(pfisr_dir, 'pfa121125.003.hdf5'), 'Alternating Code Uncorrected Ne'],
                           [os.path.join(pfisr_dir, 'pfa121125.004.hdf5'), 'Alternating Code (AC16-30)']]
        mhisr_filenames = [[os.path.join(mhisr_dir, 'mlh121128m.004.hdf5'), 'Gridded data'],
                           [os.path.join(mhisr_dir, 'mlh121128g.002.hdf5'), 'Basic Parameters'],
                           [os.path.join(mhisr_dir, 'mlh121128i.002.hdf5'), 'Zenith single-pulse'],
                           [os.path.join(mhisr_dir, 'mlh121128j.002.hdf5'), 'Zenith alternating-code']]
        pfisr_filename = pfisr_filenames[1][0]
        mhisr_filename = mhisr_filenames[0][0]
    elif day == datetime.datetime(2016, 10, 13):
        # 2016-10-13 Storm
        ionogram_data_filename = 'ionogram_data/ionogram_parameters20161013.csv'
        pfisr_filenames = [[os.path.join(pfisr_dir, 'pfa161012.001.hdf5'), 'Long Pulse Uncorrected Ne'],
                           [os.path.join(pfisr_dir, 'pfa161012.002.hdf5'), 'Long Pulse (480)'],
                           [os.path.join(pfisr_dir, 'pfa161012.003.hdf5'), 'Alternating Code Uncorrected Ne'],
                           [os.path.join(pfisr_dir, 'pfa161012.004.hdf5'), 'Alternating Code (AC16-30)']]
        mhisr_filenames = [[os.path.join(mhisr_dir, 'mlh161013m.006.hdf5'), 'Gridded data'],
                           [os.path.join(mhisr_dir, 'mlh161013g.005.hdf5'), 'Basic Parameters'],
                           [os.path.join(mhisr_dir, 'mlh161013i.006.hdf5'), 'Zenith single-pulse'],
                           [os.path.join(mhisr_dir, 'mlh161013j.006.hdf5'), 'Zenith alternating-code'],
                           [os.path.join(mhisr_dir, 'mlh_pl_20161013.002.hdf5'), 'Plasma Line']]
        pfisr_filename = pfisr_filenames[1][0]
        mhisr_filename = mhisr_filenames[0][0]
    elif day == datetime.datetime(2017, 9, 8):
        # 2017-09-08 Storm
        ionogram_data_filename = 'ionogram_data/ionogram_parameters20170908.csv'
        pfisr_filenames = [[os.path.join(pfisr_dir, 'pfa170905.001.hdf5'), 'Long Pulse Uncorrected Ne'],
                           [os.path.join(pfisr_dir, 'pfa170905.002.hdf5'), 'Long Pulse (480)'],
                           [os.path.join(pfisr_dir, 'pfa170905.003.hdf5'), 'Alternating Code Uncorrected Ne'],
                           [os.path.join(pfisr_dir, 'pfa170905.004.hdf5'), 'Alternating Code (AC16-30)'],
                           [os.path.join(pfisr_dir, 'pfa170905.005.hdf5'), 'Alternating Code Uncorrected Ne'],
                           [os.path.join(pfisr_dir, 'pfa170905.006.hdf5'), 'Alternating Code (AC16-30)'],
                           [os.path.join(pfisr_dir, 'pfa170905.007.hdf5'), 'Long Pulse Uncorrected Ne'],
                           [os.path.join(pfisr_dir, 'pfa170905.008.hdf5'), 'Long Pulse (480)']]
        mhisr_filenames = [[os.path.join(mhisr_dir, 'mlh170908m.004.hdf5'), 'Gridded data'],
                           [os.path.join(mhisr_dir, 'mlh170908g.003.hdf5'), 'Basic Parameters'],
                           [os.path.join(mhisr_dir, 'mlh170908i.004.hdf5'), 'Zenith single-pulse'],
                           [os.path.join(mhisr_dir, 'mlh170908j.004.hdf5'), 'Zenith alternating-code'],
                           [os.path.join(mhisr_dir, 'mlh_pl_20170908.002.hdf5'), 'Plasma Line']]
        pfisr_filename = pfisr_filenames[7][0]
        mhisr_filename = mhisr_filenames[0][0]
    else:
        raise ValueError('No data available for date {}'.format(day))
    return [pfisr_filenames, mhisr_filenames, ionogram_data_filename, pfisr_filename, mhisr_filename]


if __name__ == '__main__':
    # TODO: Implement the parser
    # Possible Implementation:
    # date argument that gets parsed as datetime and then appropriate files are read based on date
    # parser = _build_arg_parser(argparse.ArgumentParser)
    # args = parser.parse_args()

    # Set the day we want to make plots for
    #day_to_analyze = datetime.datetime(2012, 1, 14)
    day_to_analyze = datetime.datetime(2012, 1, 15)
    #day_to_analyze = datetime.datetime(2012, 9, 5)
    #day_to_analyze = datetime.datetime(2012, 11, 28)
    #day_to_analyze = datetime.datetime(2016, 10, 13)
    #day_to_analyze = datetime.datetime(2017, 9, 8)

    # Set ISR data directories and filenames
    main_dir = os.path.join('C:\\', 'Users', 'sugargf1', 'Box', 'Data')
    pfisr_dir = os.path.join(main_dir, 'PFISR')
    mhisr_dir = os.path.join(main_dir, 'MillstoneHill')
    start_time = day_to_analyze
    stop_time = day_to_analyze + datetime.timedelta(days=1)
    mh_station_id = 'MHJ45'  # 'HAJ43'
    pf_station_id = 'EI764'  # 'CO764' 'PF765'

    [pfisr_filenames, mhisr_filenames, ionogram_data_filename, pfisr_filename, mhisr_filename] = get_filenames_for_day(
        day_to_analyze, pfisr_dir=pfisr_dir, mhisr_dir=mhisr_dir)

    # Load the ionogram data
    ionogram_data = pd.read_csv(ionogram_data_filename, parse_dates=['datetime'])
    # Filter by station id
    pf_ig_data = ionogram_data[ionogram_data['station_id'] == pf_station_id]
    mh_ig_data = ionogram_data[ionogram_data['station_id'] == mh_station_id]

    # Load the ISR data
    if pfisr_filename is not None:
        pf_isr_data = hdf5_to_dataframe(pfisr_filename)
        pf_isr_data = pf_isr_data[pf_isr_data['el'] == 90]
        ax = utils.plot_isr_and_ionosonde_data(pf_isr_data, pf_ig_data, window_size=7,
                                               title='Poker Flat',
                                               ylabel='Maximum Density [m$^{-3}$]', val='nmf')
        ax = utils.plot_isr_and_ionosonde_data(pf_isr_data, pf_ig_data, window_size=7,
                                               title='Poker Flat',
                                               ylabel='Critical Frequency [MHz]', val='fof2')
    if mhisr_filename is not None:
        mh_isr_data = hdf5_to_dataframe(mhisr_filename)
        mh_isr_data = mh_isr_data[mh_isr_data['el'] == 90]
        ax = utils.plot_isr_and_ionosonde_data(mh_isr_data, mh_ig_data, window_size=7,
                                               title='Millstone Hill',
                                               ylabel='Maximum Density [m$^{-3}$]', val='nmf', plot_filter=False,
                                               classifier_filename=os.path.join('classifiers',
                                                                                'scaler_and_classifiers_and_betas_27_5_10.sav'))
        ax = utils.plot_isr_and_ionosonde_data(mh_isr_data, mh_ig_data, window_size=7,
                                               title='Millstone Hill',
                                               ylabel='Critical Frequency [MHz]', val='fof2', plot_filter=False)

    # Plot all the isr data
    ax = plot_all_data(pfisr_filenames, title='PFISR', show_errors=False)
    ax.set_xlim(start_time, stop_time)
    ax = plot_all_data(mhisr_filenames, title='Millstone Hill ISR')
    ax.set_xlim(start_time, stop_time)
    ax = plot_all_data(mhisr_filenames, description_filter=None, val='fof2', ylabel='fOF2 [MHz]',
                  xlabel='UTC Time', title='Millstone Hill ISR', semilogy=False)
    ax.set_xlim(start_time, stop_time)
    # ax = plot_all_data([mhisr_filenames[0]], val='hmf', title='hmf', ylabel='Range [km]', semilogy=False)
    # ax = plot_all_data([mhisr_filenames[0]], val='nmf', title='nmf', ylabel='Electron Density [m^-3]', semilogy=True,
    #                    xlim=[start_time, stop_time])
    # ax = plot_all_data([pfisr_filenames[7]], val='hmf', title='hmf', ylabel='Range [km]', semilogy=False)
    # ax = plot_all_data([pfisr_filenames[7]], val='nmf', title='nmf', ylabel='Electron Density [m^-3]', semilogy=True,
    #                    xlim=[start_time, stop_time])
    print('Done')
