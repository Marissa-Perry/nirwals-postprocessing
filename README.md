# NIRWALS postprocessing

Additional reduction steps for NIRWALS data from SALT:
- atmospheric absorption (telluric) correction
- flux calibration
- combining dithered observations (if any) into a data cube 
    - Reconstruction uses a Gaussian kernel/modified Shepard's-method following the approach of [Law+2016](https://ui.adsabs.harvard.edu/abs/2016AJ....152...83L/abstract) and was originally developed by Antoine Mahoro.
- extracting a 1D spectrum for the science target

## Structure

- `YYYYMMDD/` - data directories (not tracked in git) containing data products from the NIRWALS DRP to be postprocessed
- `data/` - on the NIRWALS IFU, dither pattern, and standard spectrum for flux calibration
- `functions/` - helper functions organized by each postprocessing step 
- `outputs/` - diagnostic plots, throughput curves, and data products
- `wrappers/postprocessing.py` - the script to run the postprocessing

## Setup

[eventually add a list of dependencies in .txt file]

## Usage
```bash
python -m post_reduction_processing.wrappers.postprocessing YYYYMMDD
```
where YYYYMMDD is the date of the observation to postprocess. Two optional arguments can be added:
- `--telluric-standard-date YYYYMMDD`: a reduced telluric-standard observation used to derive the telluric correction with a PypeIt star-model fit. If omitted, the telluric correction is instead fit directly to the science target's own spectrum with PypeIt's poly-model fit.
- `--specphot-date YYYYMMDD`: a reduced spectrophotometric-standard observation, used for flux calibration. If omitted, flux calibration is skipped.

Multiple science exposures found for YYYYMMDD are treated as dithers. The dither offset pattern used for combination is read from `data/IFU/User_offset.csv`. This file must be edited directly to setup a different dither pattern.