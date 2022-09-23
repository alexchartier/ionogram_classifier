# This will look at the themens data and classify whether or not the ARTIST autoscaled values are good or bad
import h5py
import os
import datetime
import julian
from ionosondeparser.scrapeImages import dl_data


main_dir = 'themens_data'
all_files = os.listdir(main_dir)

for file in all_files:
    station_id = file.split('.')[0]
    h5data = h5py.File(os.path.join(main_dir, file), 'r')
    timestamps = [julian.from_jd(i) for i in h5data['Julian_dates'][:]]
    img_fn_fmt = '%Y%b%d_%H%M%S.png'
    imgdir = os.path.join('ionosondeparser', 'themens')
    unique_days = []
    for timestamp in timestamps:
        day = timestamp.strftime('%Y,%m,%d')
        if day not in unique_days:
            unique_days.append(day)
            # Download the station day
            dl_data([station_id], imgdir, img_fn_fmt, timestamp, entire_day=True)
