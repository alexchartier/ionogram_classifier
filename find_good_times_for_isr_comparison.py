import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# This script will help find times to look for ISR data that overlap with EIELSON and MILLSTONE HILL ionosondes

# Load in the data
#ionogram_data = pd.read_csv('ionogram_data/ionogram_parametersEI764.csv', parse_dates=['datetime'])
ionogram_data = pd.read_csv('ionogram_data/ionogram_parametersMHJ45.csv', parse_dates=['datetime'])

plt.figure()
sort_inds = ionogram_data['datetime'].values.argsort()
plt.plot(ionogram_data['datetime'][sort_inds], ionogram_data['fof2'][sort_inds])
plt.grid('on')
plt.xlabel('Time (UTC)')
plt.ylabel('fOF2 (MHz)')
plt.title('Millstone Hill')
