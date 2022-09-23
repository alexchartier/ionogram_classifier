import requests
from bs4 import BeautifulSoup as bs
import os
from PIL import Image, ImageOps
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
import glob
import sys
import argparse
import scipy.signal

""" 
Convert UMass/Lowell ionogram images into data using Python
Authors: Glenn Sugar & Alex Chartier, JHU/APL 2021

Objectives:
    #1 Digitize the image data for analysis
    #2 Grab the images automatically
    #3 Query the database for the autoscaled parameters (not implemented yet)
"""

# Set the ionospheric parameters to extract
iono_params = ['fof2', 'fof1', 'fof1p', 'foe', 'foep', 'fxi', 'foes', 'fmin', 'mufd', 'md', 'd', 'hf', 'hf2', 'he',
               'hes', 'hmf2', 'hmf1', 'hme', 'yf2', 'yf1', 'ye', 'b0', 'b1', 'clevel']

def plot_multi(imgdir, ext='.png'):
    print('plotting contents of %s' % imgdir)
    # Loop through the files in img directory
    flist = glob.glob(os.path.join(imgdir, '*%s' % ext))
    for img_fn in flist:    
        produce_plot(img_fn, img_fn)

    #TODO presumably save out the files as either .pkl or .png or something


def produce_plot(img_fn, title):
    img = Image.open(img_fn)

    # Process the images
    params = get_params()
    colors = get_colors()
    ionogram = analyse_image(img, params, colors)

    # Plot the ionograms
    plot_ionogram(ionogram, colors, title=title)


def dl_data(station_id_list, img_dir, img_fn_fmt, day,
            base_url='https://lgdc.uml.edu/common/DIDBDayStationStatistic?ursiCode=%s&',
            entire_day=True, entire_year=False):
    os.makedirs(img_dir, exist_ok=True)
    # See if we want to download an entire year
    if entire_year and len(station_id_list) == 1:
        station_id = station_id_list[0]
        day_to_look = day
        while day_to_look.year == day.year:
            # Try to download for the current day
            station_url = base_url % station_id + day_to_look.strftime('year=%Y&month=%m&day=%d')
            # Get ionogram URLs
            ionogram_urls = read_web_links(station_url)
            for url in ionogram_urls:
                # Pull time out of the URL
                try:
                    time = dt.datetime.strptime(url.split('=')[-1], '%Y.%m.%d (%j) %H:%M:%S.%f')
                except:
                    print('Could not parse %s' % url)
                    continue
                # Download the images
                img_fn = '%s_%s' % (os.path.join(img_dir, station_id), time.strftime(img_fn_fmt))
                try:
                    dl_ionogram(url, img_fn)
                    print("Downloaded to %s" % img_fn)
                except Exception as e:
                    print("Could not download {}".format(url))
            # Look at the next day
            day_to_look += dt.timedelta(1)
        return

    # We are not downloading an entire year for a single station
    for station_id in station_id_list:
        station_url = base_url % station_id + day.strftime('year=%Y&month=%m&day=%d')

        # Get ionogram URLs
        ionogram_urls = read_web_links(station_url)
        for url in ionogram_urls:
            # Pull time out of the URL
            try:
                time = dt.datetime.strptime(url.split('=')[-1], '%Y.%m.%d (%j) %H:%M:%S.%f')
            except:
                print('Could not parse %s' % url)
                continue

            # Download the images
            if entire_day or time == day:
                img_fn = '%s_%s' % (os.path.join(img_dir, station_id), time.strftime(img_fn_fmt))
                try:
                    dl_ionogram(url, img_fn)
                    print("Downloaded to %s" % img_fn)
                except Exception as e:
                    print("Could not download {}".format(url))


def get_ionogram_times(station_id, day, servlet_url):
    """ 
    Read the times of the ionograms from DIDBase 
    https://lgdc.uml.edu/common/DIDBGetValues?ursiCode=DB049&charName=foF2&fromDate=2007.06.25&toDate=2007.06.26
    """
    timestr1 = day.strftime('%Y.%m.%d')
    timestr2 = (day + dt.timedelta(days=1)).strftime('%Y.%m.%d')
    url = '%s?ursiCode=%s&charName=foF2&fromDate=%s&toDate=%s' % (servlet_url, station_id, timestr1, timestr2)
    text = read_web_text(url)
    times = []
    for entry in text.split('\n'):
        try:
            times.append(dt.datetime.strptime(entry, '%Y-%m-%dT%H:%M:%S.%fZ'))
        except:
            None
    return times


def gen_iono_list(
        day,
        iono_list_fn='lowell_iono_list.txt',
        latlim=[-90, 90],
        lonlim=[0, 360]):

    # Load the list
    iono_list, creation_date = read_ionosonde_list(iono_list_fn)
    if creation_date < day:
        print('Iono list out of date - time to download a new one')
        # TODO: Read list from https://lgdc.uml.edu/common/DIDBStationList

    # Get a list of site IDs within the requested area
    idx = np.logical_and.reduce([
        iono_list['lat'] > latlim[0], iono_list['lat'] < latlim[1],
        iono_list['lon'] > lonlim[0], iono_list['lon'] < lonlim[1],
        iono_list['starttime'] < day,
        np.logical_or(iono_list['endtime'] > day, iono_list['endtime'] == creation_date),
    ])

    out = {}
    for k, v in iono_list.items():
        out[k] = v[idx]

    return out
    

def read_ionosonde_list(fn, timestr='%b%d%Y'):
    # Read the list of Lowell Digisondes
    with open(fn, 'r') as f:
        txt = f.readlines()

    out = {
        'code': [],
        'name': [],
        'lat': [],
        'lon': [],
        'starttime': [],
        'endtime': [],
    }

    for line in txt:
        entries = line.split()
        if len(entries) == 0:
            continue 
        if entries[0] == 'Created:':
            creation_date = dt.datetime.strptime(''.join(entries[1:4]), '%b%d,%Y') 
        try:
            float(entries[0])
        except:
            continue
        out['code'].append(entries[1])
        out['name'].append(' '.join(entries[2:-10]))
        out['lat'].append(float(entries[-10]))
        out['lon'].append(float(entries[-9]))
        out['starttime'].append(dt.datetime.strptime(entries[-8] + entries[-7] + entries[-5], timestr))
        out['endtime'].append(dt.datetime.strptime(entries[-4] + entries[-3] + entries[-1], timestr))

    for k, v in out.items():
        out[k] = np.array(v)

    assert len(out) > 0, 'Failed to read any ionograms from the iono_list %s. Check formatting and re-download' % iono_list_fn

    return out, creation_date
    

def plot_ionogram(
        ionogram, colors,
        rg=[0, 1000],
        frq=[0, 20],
        title='',
):
    for key, color in colors.items():
        freqs, ranges = ionogram[key]
        plt.scatter(freqs, ranges, 0.1, matplotlib.colors.to_hex(np.array(colors[key])/255))
    plt.grid('on')
    #plt.xlabel('Frequency (MHz)')
    plt.ylabel('Range (km)')
    plt.title(title)
    plt.show()


def get_params():
    # Pixel coordinates for ionogram image components
    params = {
        # ytick labels
        'left_ylabel' : 150,
        'top_ylabel': 45,
        'right_ylabel' : 187,
        'bottom_ylabel' : 550,
        # xtick labels
        'left_xlabel' : 217,
        'top_xlabel' : 530,
        'right_xlabel' : 735,
        'bottom_xlabel' : 543,
        # plot pixels
        'left_plot' : 189,
        'top_plot' : 45,
        'right_plot' : 747,
        'bottom_plot' : 522,
        # biteout pixels
        'left_freqbo' : 189,
        'top_freqbo' : 521,
        'right_freqbo' : 747,
        'bottom_freqbo' : 524,
    }
    return params


def get_colors():
    # Colors for each different key
    colors = {
        'NNE': [16, 222, 255],
        'E': [66, 173, 255],
        'W': [181, 0, 156],
        'Vo-': [255, 66, 156],
        'Vo+': [255, 0, 49],
        'SSW': [247, 181, 173],
        'X-': [0, 128, 0],
        'X+': [129, 184, 96],
        'SSE': [214, 216, 6],
        'NNW': [41, 0, 231],
    }
    return colors


def read_web_links(url):
    # Get the links from a webpage
    page = requests.get(url)    
    data = page.text
    soup = bs(data, 'html.parser')

    urls = []
    for url in soup.find_all('a'):
        urls.append(url.get('href'))
    return urls


def read_web_text(url):
    # Get the text from a webpage
    page = requests.get(url)
    soup = bs(page.content, "html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text

    
def dl_ionogram(url, filename):
    page = requests.get(url)
    soup = bs(page.content, "html.parser")
    img_url = soup.find("img").attrs.get("src")
    response = requests.get(img_url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
    # Return the image
    return Image.open(filename)


def get_tickvals_and_ticklocs(
        img, crop_coords, 
        axis='x',
):
    digit_thresholds = [25, 19, 22, 22, 25, 22, 25, 22, 25, 25]
    digit_col_push = [0, -1, 0, 0, 0, 0, 0, 0, 0, 0]
    img_cropped = ImageOps.grayscale(img.crop(crop_coords))
    img_cropped = img_cropped.convert('1')
    row_dict = {}
    # Get the indices of all digits
    for digit in range(10):
        kernel = Image.open(os.path.join('resources', '{digit:d}.png'.format(digit=digit))).convert('1')
        conv_result = scipy.signal.correlate2d(np.array(img_cropped) - 0.5, np.array(kernel) - 0.5, mode='valid')
        coords = np.where(conv_result > digit_thresholds[digit])
        # Append the digit and column location to the row
        for ind, row in enumerate(coords[0]):
            if row not in row_dict.keys():
                row_dict[row] = [[coords[1][ind], digit]]
            else:
                row_dict[row].append([coords[1][ind], digit])

    tick_vals = []
    tick_locs = []
    if axis == 'y':
        # Loop through each row and create the number
        rows = row_dict.keys()
        for row in rows:
            # Sort the columns
            cols = [dat_pair[0] for dat_pair in row_dict[row]]
            digits = [row_dict[row][ind][1] for ind in np.argsort(cols)]
            tick_vals.append(float(''.join([str(d) for d in digits])))
            tick_locs.append(row + crop_coords[1] + 6 + digit_col_push[digit])
    else:
        single_digit_shift = 3
        double_digit_shift = 7
        rows = row_dict.keys()
        if len(rows) > 1:
            print("Error parsing x axis. Too many rows")
        for row in rows:
            cols = [dat_pair[0] for dat_pair in row_dict[row]]
            cols_sorted = np.sort(cols)
            digits = [row_dict[row][ind][1] for ind in np.argsort(cols)]
            i = 0
            while i < len(cols_sorted):
                # See if this is the last digit
                if i == len(cols_sorted)-1:
                    tick_vals.append(float(digits[i]))
                    tick_locs.append(cols_sorted[i] + crop_coords[0] + single_digit_shift)
                    i += 1
                else:
                    # See if this is a 2 digit number
                    if cols_sorted[i+1] - cols_sorted[i] < 20:
                        tick_vals.append(float(digits[i]*10 + digits[i+1]))
                        tick_locs.append(cols_sorted[i] + crop_coords[0]+double_digit_shift)
                        i += 2
                    else:
                        tick_vals.append(float(digits[i]))
                        tick_locs.append(cols_sorted[i] + crop_coords[0] + single_digit_shift)
                        i += 1
    return [tick_vals, tick_locs]


def analyse_image(img, params, colors):
    # Get the x/y label cropping out
    ylabels_crop = (params['left_ylabel'], params['top_ylabel'], params['right_ylabel'], params['bottom_ylabel'])

    [ytick_vals, ytick_locs] = get_tickvals_and_ticklocs(img, ylabels_crop, axis='y')
    xlabels_crop = (params['left_xlabel'], params['top_xlabel'], params['right_xlabel'], params['bottom_xlabel'])
    [xtick_vals, xtick_locs] = get_tickvals_and_ticklocs(img, xlabels_crop, axis='x')
    linfit_freqs = np.polyfit(xtick_locs, xtick_vals, 1)
    linfit_ranges = np.polyfit(ytick_locs, ytick_vals, 1)

    # Get the frequency bite-outs
    biteout_crop = (params['left_freqbo'], params['top_freqbo'], params['right_freqbo'], params['bottom_freqbo'])
    freq_biteouts_pix = get_frequency_biteouts(img, biteout_crop)
    freq_biteouts = linfit_freqs[0] * (freq_biteouts_pix + params['left_freqbo']) + linfit_freqs[1]

    # Extract ionosphere parameters
    iono_params = get_ionospheric_parameters(img)

    # Crop all non-plot pixels
    plot_crop = (params['left_plot'], params['top_plot'], params['right_plot'], params['bottom_plot'])
    img_plot = img.crop(plot_crop)
    values = {'freq_biteouts': freq_biteouts}
    img_plot_data = np.array(img_plot.getdata())
    plt.figure()
    for key in colors.keys():
        indices = np.where((img_plot_data == colors[key]).all(axis=1))[0]
        x = indices % img_plot.size[0]
        y = np.floor(indices / img_plot.size[0])
        # Convert x,y pixel to freq,range
        linfit_freqs = np.polyfit(xtick_locs, xtick_vals, 1)
        freqs = linfit_freqs[0] * (x + params['left_plot']) + linfit_freqs[1]
        linfit_ranges = np.polyfit(ytick_locs, ytick_vals, 1)
        ranges = linfit_ranges[0] * (y + params['top_plot']) + linfit_ranges[1]
        # Put into the values dictionary
        values[key] = np.vstack([freqs, ranges])

    return values


def get_frequency_biteouts(img, crop_coords):
    # Crop the image
    img_cropped = np.asarray(img.crop(crop_coords).convert('1'))
    # Extract the top, middle, and bottom rows
    top_row = img_cropped[0, :]
    mid_row = img_cropped[1, :]
    valid_pixels = np.where(np.logical_xor(top_row, mid_row))[0]
    # Look to left and right of all valid_pixels to see if there was a tick there that was missed
    missed_pixel_left = ~top_row[valid_pixels-1]
    valid_pixels = np.sort(np.concatenate([valid_pixels, valid_pixels[missed_pixel_left]-1]))
    missed_pixel_right = ~top_row[valid_pixels + 1]
    # Add the missing pixels
    for missed_pix in valid_pixels[missed_pixel_right]-1:
        if missed_pix not in valid_pixels:
            valid_pixels = np.append(valid_pixels, missed_pix)
    valid_pixels = np.sort(valid_pixels)
    return valid_pixels


def extract_iono_param_value(img):
    img_bw = ImageOps.grayscale(img).convert('1')
    col_list = []
    # Get the indices of all digits
    for digit in range(10):
        kernel = Image.open(os.path.join('resources', '{digit:d}sm.png'.format(digit=digit))).convert('1')
        conv_result = scipy.signal.correlate2d(np.array(img_bw) - 0.5, np.array(kernel) - 0.5, mode='valid')
        # 5's are hard
        if digit == 5:
            coords = np.where(conv_result >= kernel.size[0]*kernel.size[1]/4-1)
        else:
            coords = np.where(conv_result >= kernel.size[0] * kernel.size[1] / 4 - 0.5)
        # Append the digit and column location to the row
        for ind, row in enumerate(coords[0]):
            col_list.append([coords[1][ind], digit])
    # Find any decimal
    kernel = Image.open(os.path.join('resources', 'dotsm.png')).convert('1')
    conv_result = scipy.signal.correlate2d(np.array(img_bw) - 0.5, np.array(kernel) - 0.5, mode='valid')
    coords = np.where(conv_result >= kernel.size[0] * kernel.size[1] / 4 - 0.5)
    # Append the digit and column location to the row
    for ind, row in enumerate(coords[0]):
        col_list.append([coords[1][ind], '.'])
    # Sort the columns
    cols = [dat_pair[0] for dat_pair in col_list]
    digits = [col_list[ind][1] for ind in np.argsort(cols)]
    if len(digits) == 0:
        value = 'N/A'
    else:
        value = float(''.join([str(d) for d in digits]))
    return value


def get_ionospheric_parameters(img, aggressive_crop=True):
    img_orig = img.convert('1')
    if aggressive_crop:
        crop_start_coords = [0, 50]
        crop_width_height = [img_orig.size[0] / 4 - 50, img_orig.size[1] - 130]
    else:
        crop_start_coords = [0, 0]
        crop_width_height = [img_orig.size[0] / 4, img_orig.size[1]]
    img_orig_crop = img_orig.crop([crop_start_coords[0], crop_start_coords[1], crop_width_height[0], crop_width_height[1]])
    iono_param_dict = {}
    for iono_param in iono_params:
        img_text = Image.open(os.path.join('resources', iono_param+'.png')).convert('1')
        # Find where the image text is in the original image
        conv_result = scipy.signal.correlate2d(np.array(img_orig_crop) - 0.5, np.array(img_text) - 0.5, mode='valid')
        conv_coords = np.where(conv_result >= img_text.size[0]*img_text.size[1]/4-2)
        # If there is a match, continue
        if len(conv_coords[0]) > 0:
            # Extract the number (or N/A) for the parameter
            value = extract_iono_param_value(img.crop((conv_coords[1][0] + img_text.size[0] + crop_start_coords[0],
                                                       conv_coords[0][0] + crop_start_coords[1],
                                                       conv_coords[1][0] + crop_start_coords[0] + img_text.size[0] + 100,
                                                       conv_coords[0][0] + crop_start_coords[1] + 14)))
            iono_param_dict[iono_param] = value
    return iono_param_dict


def _build_arg_parser(Parser, *args):
    scriptname = os.path.basename(sys.argv[0])

    formatter = argparse.RawDescriptionHelpFormatter(scriptname)
    width = formatter._width

    title = "ionogram_scraper"
    copyright = "Copyright (c) 2021 JHU/APL"
    shortdesc = "Scrape Lowell's ionograms into Python."
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
        "%s [-f imgfile] [-t yyyy,mm,dd(,HH,MM)] [-d img_dir] [--download] [-p]" % scriptname
    )

    # parse options
    parser = Parser(
        description=desc,
        usage=usage,
        prefix_chars="-+",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-f",
        "--img_file",
        dest="img_fn",
        default=None,
        help="""Ionogram image file to plot""",
    )

    parser.add_argument(
        "-t",
        "--time",
        dest="daystr",
        default=None,
        help="""Day to download/plot (format yyyy,mm,dd (optional: ,hh,mm,ss)""",
    )

    parser.add_argument(
        "-i",
        "--station_id",
        dest="station_id",
        default=None,
        help="""Station ID to download (leave blank if you want to download all stations)""",
    )

    parser.add_argument(
        "-d",
        "--img_dir",
        dest="imgdir",
        default=None,
        help="""Where to store the files (can include time wildcards)""",
    )

    parser.add_argument(
        "-dl",
        "--download",
        dest="download",
        action="store_true",
        help="""Optionally download the files from UMass/Lowell""",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        dest="verbose",
        action="store_false",
        help="""Reduce text output to the screen. (default: False)""",
    )

    parser.add_argument(
        "-p",
        "--plot",
        dest="make_plots",
        action="store_true",
        help="""Generate plots. (default: False)""",
    )

    return parser


if __name__ == '__main__':
    parser = _build_arg_parser(argparse.ArgumentParser)
    args = parser.parse_args()

    if args.img_fn:
        print('Image file provided - ignoring other args')
        produce_plot(args.img_fn, args.img_fn)
    elif args.imgdir:
        if len(args.daystr.split(',')) == 1:
            day = dt.datetime.strptime(args.daystr, '%Y')
            entire_day = True
            entire_year = True
        elif len(args.daystr.split(',')) == 3:
            day = dt.datetime.strptime(args.daystr, '%Y,%m,%d')
            entire_day = True
            entire_year = False
        elif len(args.daystr.split(',')) == 5:
            day = dt.datetime.strptime(args.daystr, '%Y,%m,%d,%H,%M')
            entire_day = False
            entire_year = False

        imgdir = day.strftime(args.imgdir)
        if args.download:
            img_fn_fmt = '%Y%b%d_%H%M%S.png'
            # See if we want to download just a single station
            if args.station_id is None:
                iono_list = gen_iono_list(day)
                dl_data(iono_list['code'], imgdir, img_fn_fmt, day, entire_day=entire_day, entire_year=entire_year)
            else:
                print('Downloading all data for station: {}'.format(args.station_id))
                dl_data([args.station_id], imgdir, img_fn_fmt, day, entire_day=entire_day, entire_year=entire_year)
        if args.make_plots:
            plot_multi(imgdir)
    else:
        parser.print_usage()
        print('Either image filename or directory must be provided')
