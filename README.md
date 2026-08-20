# NIRWALS postprocessing

Postprocessing steps for reduced NIRWALS observations from SALT:
- atmospheric absorption (telluric) correction
- flux calibration
- 1D spectra product
    - selects on approximated half-light radius of IFU for each dithered exposure (if any) and averages across them.
- 2D fiber data product
    - dithered observations (if any) are stacked as additional rows.
- 3D data cube product
    - dithered observations (if any) are combined in this data product. 
    - originally developed by Antoine Mahoro and uses a Gaussian kernel/modified Shepard's-method.
    - Note: there is correlation among neighboring spaxels, so the spaxel errors should be treated as lower bounds.

## Structure

- `data/` - NIRWALS IFU layout, dither-pattern setup, and standard spectra for flux calibration
- `scripts/` - helper functions organized by each postprocessing step 
- `outputs/` - diagnostic plots, throughput curves, and final data products
- `wrappers/postprocessing.py` - script for running the postprocessing

## Setup

Clone this repository, then create and activate the conda environment with all dependencies:

```bash
git clone https://github.com/Marissa-Perry/nirwals-postprocessing.git
cd nirwals-postprocessing
conda env create -f nirwals_postprocessing_env.yml
conda activate nirwals_postprocessing
```

Make sure your NIRWALS DRP directory (`nirwals-pipeline`) is on the same level as this cloned repository (`nirwals-postprocessing`).

Be sure to download your data for flux calibration into `data/Rayner_standard_data/`.

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