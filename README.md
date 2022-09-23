# Authorship:

This code was written by Glenn Sugar, now at Space-X. 

# sami3-validation

This project will run validation tests on sami3 ionospheric models. It also contains code
that will create classifiers to determine whether ARTIST foF2 values are accurate or 
inaccurate (defined as being within 0.5 MHz of a manually analyzed ionogram). There are training
data that have been manually analyzed by David Themens.

# Installing

Once you clone the git repo, you will need to initialize the ionosondeparser submodule:
```
git submodule init
git submodule update
```
You also need to install Cartopy, which might not work with a simple `pip install cartopy`.
If you have trouble and are on Windows, you can install by downloading the correct wheel for your python version here:
https://www.lfd.uci.edu/~gohlke/pythonlibs/#cartopy and then run `pip install cartopy_file_you_downloaded.whl`.
You will also might need to download statsmodels from https://pypi.org/project/statsmodels/0.12.1/#files in order to install sktime.
 
# Classifier
You can generate interesting plots by running the `make_paper_plots.py` file.
You can create new classifiers by running the `analyze_themens_data.py` file.

# Running the SAMI validation

When this code was first written, I did not know about a web interface to automatically get the ARTIST foF2 values.
The method described below will download ionogram images, use character recognition to extract the ionospheric 
parameters from the images, and then use the extracted values for analysis. This code should be re-written so that
the ionosondeparser submodule is not needed anymore, and instead just use the `utils.get_iono_params` function. 

## Non-optimal way using the `ionosondeparser` submodule
You will need to download the ionogram images for the time you want to analyze the SAMI3 output:
```
cd ionosondeparser
python scrapeImages.py -dl -t "2020,12,13,01,00" -d "images%Y%m%d"
```
This will save images for 2020-12-13 01:00 UTC to the images20201213 directory (it will be created if it does not exist).
 
Extract parameters from the ionograms:
```
python extract_parameters.py -d images20201213
``` 
This will save a file called `ionogram_parameters.json` that will contain all the ionogram parameters extracted by ARTIST
from the images20201213 directory.

Validate SAMI3:
```
cd ..
python validate_sami3.py -i "ionosondeparser/ionogram_parameters.json" -s "~\Box\Aug_21\prior_reg\2021213-0100_prior_reg.nc"
``` 
Here `-i` is the input .json file and `-s` is the sami3 output netcdf file.

## Better way using `utils.get_iono_params`
The `utils.get_iono_params` function is an easy way to get ARTIST-extracted ionosphere parameters
for different sites. You can call this function with:
```
utils.get_iono_params(station_id, params, start_datetime, end_datetime)
```
where `station_id` is a string containing the ionosonde station id, 
`params` is a list of strings and acceptable strings are:
```
['foF2', 'foF1', 'foE', 'foEs', 'fbEs', 'foEa', 'foP', 'fxI', 'MUFD', 'MD', 'hF2', 'hF', 'hE', 'hEs',
 'hEa', 'hP', 'TypeEs', 'hmF2', 'hmF1', 'zhalfNm', 'yF2', 'yF1', 'yE', 'scaleF2', 'B0', 'B1', 'D1',
 'TEC', 'FF', 'FE', 'QF', 'QE', 'fmin', 'fminF', 'fminE', 'fminEs', 'foF2p']
```
This function still needs to be tied into the overall `validate_sami3.py` script.
