# NIRWALS postprocessing

Postprocessing steps for reduced NIRWALS observations from SALT:
- atmospheric absorption (telluric) correction
- flux calibration
- 2D fiber data product
    - dithered observations (if any) are stacked as additional rows.
- 3D data cube product
    - dithered observations (if any) are combined in this data product. 
    - originally developed by Antoine Mahoro and uses a Gaussian kernel/modified Shepard's-method.
    - Note: there is correlation among neighboring spaxels, so the spaxel errors should be treated as lower bounds.
- 1D spectra product
    - selects on approximated half-light radius of IFU for each dithered exposure (if any) and averages across them.

## Structure

- `YYYYMMDD/` - data directories (not tracked in git) containing data products from the NIRWALS DRP to be postprocessed
- `data/` - NIRWALS IFU layout, dither-pattern setup, and standard spectrum for flux calibration
- `functions/` - helper functions organized by each postprocessing step 
- `outputs/` - diagnostic plots, throughput curves, and final data products
- `wrappers/postprocessing.py` - script for running the postprocessing

## Setup

[eventually add a list of dependencies in .txt file]

## Usage
```bash
python -m post_reduction_processing.wrappers.postprocessing YYYYMMDD --telluric-standard-date YYYYMMDD --specphot-date YYYYMMDD
```
where the required YYYYMMDD argument is the date of the science target observation to postprocess. Some optional arguments are included in this example, as they're highly encouraged.

Description of optional arguments:
- `--telluric-standard-date YYYYMMDD`: a reduced telluric-standard observation used to derive the telluric correction with a PypeIt star-model fit. If omitted, the telluric correction is instead fit directly to the science target's own spectrum with PypeIt's poly-model fit.
- `--specphot-date YYYYMMDD`: a reduced spectrophotometric-standard observation used for flux calibration. If omitted, flux calibration is skipped.
- `--no-dithers`: Skips the dither combining step. Science target is auto-identified based on the highest exposure time. If there is more than one of these exposures, only the first is used.
- `--telluric-standard-polyorder`: Continuum polynomial order for the star-model telluric fit.
- `--specphot-polyorder`: Continuum polynomial order for the spec-phot standard\'s self telluric correction with poly-model.
- `--target-polyorder`: Continuum polynomial order for the poly-model telluric fit, used when no telluric standard is given.

Unless `--no-dithers` is passed, multiple science target exposures found for YYYYMMDD are treated as dithers. The dither offset pattern used for combination is read from `data/IFU/dither_pattern.csv`. This file must be edited directly to setup a different dither pattern.