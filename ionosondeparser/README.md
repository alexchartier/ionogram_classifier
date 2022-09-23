# IonosondeParser

An ionosonde image parser.

# Instructions

To download ionograms for a particular day (e.g. 2021-11-06) run

```
python scrapeImages -dl -t "2021,11,06" -d "images%Y%m%d"
```

To download ionograms for a particular day and time (e.g. 2021-11-06 04:55 UTC)run
```
python scrapeImages -dl -t "2021,11,06,04,55" -d "images%Y%m%d"
```
