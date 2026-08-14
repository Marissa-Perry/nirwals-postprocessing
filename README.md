# NIRWALS postprocessing

Additional reduction steps for NIRWALS data from SALT:
- atmospheric absorption (telluric) correction
- flux calibration
- combining dithered observations (if any)
- co-addition of source fibers by computing the half-light radius

## Structure

- `YYYYMMDD/` - data directories (not tracked in git) containing data products from the NIRWALS DRP to be postprocessed
- `data/` - information on the NIRWALS IFU and spectrophotometric standard data for flux calibration
- `functions/` - helper functions organized by each postprocessing step 
- `outputs/` - diagnostic plots, flux calibration throughput curves (necessary?), and data products
- `wrappers/` - scripts to run postprocessing

## Setup

[eventually add a list of dependencies in .txt file]

## Usage
```bash
python -m post_reduction_processing.wrappers.postprocessing YYYYMMDD
```
where YYYYMMDD is the date of the observation to be reduced. An optional argument for a spectrophotometric standard observation can be passed in the same format (i.e., the first YYYYMMDD corresponds to the science observation and the second corresponds to a spectrophotometric standard observation).
