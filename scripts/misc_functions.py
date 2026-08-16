import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
from pathlib import Path
import glob
import os

# -------- filepath params ----------
this_dir = Path(__file__).resolve().parent     # .../post_reduction_processing/scripts
proj_dir = this_dir.parent                     # .../post_reduction_processing
repo_root = proj_dir.parent                    # .../NIRWALS_reduction

data_dir = proj_dir / 'data'
output_dir = proj_dir / 'outputs'
plot_dir = output_dir / 'plots'
fiber_map_file = data_dir / 'IFU' / 'fiber_map_20221018.csv'
pipeline_root = repo_root / 'nirwals_pipeline'


os.makedirs(plot_dir, exist_ok=True)
plot_ext_spectra_dir = os.path.join(plot_dir, 'NIRWALS_DRP_reduced_spectra')
os.makedirs(plot_ext_spectra_dir, exist_ok=True)
# -----------------------------------

def load_fiber_map_dict(map_file=fiber_map_file, verbose=False):
    '''
    Load the NIRWALS fiber map.

    Returns a dict with:
      'full'        : full DataFrame (object + sky fibers, combined pseudo-slit)
      'object'      : object-only DataFrame, reindexed 0..N-1
      'sky_indices' : list of sky-fiber indices in the COMBINED pseudo-slit (pass this to process_a_fits_reduced_data)
      'obj_to_full' : array mapping object-slit index -> combined-slit index
    '''
    full = pd.read_csv(map_file)
    obj = full[full['TYPE'] == 'object'].reset_index(drop=True)
    sky_indices = full[full['TYPE'] == 'sky'].index.tolist()

    # map each object-slit index to its combined-slit index, via SLIT_ID
    obj_to_full = np.array([full[full['SLIT_ID'] == sid].index[0] for sid in obj['SLIT_ID']])

    if verbose:
        print(f'fibers, combined pseudo-slit: {len(full)}')
        print(f'fibers, object pseudo-slit:   {len(obj)}')
        print(f'sky fibers:                   {len(sky_indices)}')

    fiber_map_dict = {'full': full, 'object': obj, 'sky_indices': sky_indices, 'obj_to_full': obj_to_full}

    return fiber_map_dict


def pixel_to_wavelength(sci_header, sci_data):
    '''
    x-pixel to wavelength conversion (equation given in comments of set_header_items() in nirwalsreduce.py)

    sci_header (astropy Header obj): SCI header
    sci_data (1D arr): SCI data
    '''
    CRVAL1 = sci_header['CRVAL1']
    CDELT1 = sci_header['CDELT1']
    CRPIX1 = sci_header['CRPIX1']
    n = sci_data.shape[-1]
    return CRVAL1 - (CRPIX1 - 1.) * CDELT1 + np.arange(n) * CDELT1

def get_reduction_products(obs_date, suffix, root=pipeline_root):
    '''
    obs_date (str): YYYYMMDD
    suffix (str): the type of reduction product (e.g., ssc, ss, cf, a)
    root (str): path to pipeline dir
    '''
    primary_reduced_pattern = os.path.join(root, obs_date, 'nirwals', 'product', f'*reduced.fits')
    primary_reduced_files = sorted(glob.glob(primary_reduced_pattern))

    if not primary_reduced_files:
        raise FileNotFoundError(f'Unable to find products with the following path: {primary_reduced_pattern}')

    science_reduced_pattern = os.path.join(root, obs_date, '*', 'product', f'*reduced*{suffix}.fits')
    science_reduced_files = sorted(glob.glob(science_reduced_pattern))

    if not science_reduced_files:
        raise FileNotFoundError(f'Unable to find products with the following path: {science_reduced_pattern}')
    
    return primary_reduced_files, science_reduced_files

def exposure_dict(filepath, product_type='science'):
    '''
    filepath (str): filepath of reduction product
    product_type (str): 'science' for extracted, wavelength-calibrated products;
                        'primary' for raw detector frames
    '''
    with fits.open(filepath) as hdul:
        h0 = hdul['PRIMARY'].header
        ext_names = [x.name for x in hdul]

        def hdr(key, cast=float):
            val = h0.get(key)
            if val is None:
                return None
            try:
                return cast(val)
            except (TypeError, ValueError):
                return None

        # primary products may not carry a named SCI extension
        sci = hdul['SCI'] if 'SCI' in ext_names else hdul[0]
        wave = hdul['WAVE'] if 'WAVE' in ext_names else None
        specres = hdul['SPECRES'] if 'SPECRES' in ext_names else None
        specresc = hdul['SPECRES_COEFF_PER_FIBER'] if 'SPECRES_COEFF_PER_FIBER' in ext_names else None

        # airmass value from metadata is invalid, computing using telescope altitude
        tel_alt = float(h0['TELALT'])
        airmass = 1.0 / np.sin(np.radians(tel_alt))

        d = {
            'file': filepath,
            'filename': Path(filepath).name,
            'product_type': product_type,
            'object': h0['OBJECT'],
            'exp_type': h0['EXPTYPE'],
            'exptime': hdr('EXPTIME'),
            'airmass': airmass, 
            'gain': hdr('GAIN'),
            'grating_angle': hdr('GRRANGLE'),
            'pupil_start': hdr('PUPSTA'),
            'pupil_end': hdr('PUPEND'),
            'ngroups': hdr('NGROUPS'),
            'cfw': str(h0.get('CFWCURZ', ''))[-12:],

            'flux': sci.data,
            'spec_res': np.asarray(specres.data, dtype=float) if specres is not None else None,
            'specres_coeffs': np.asarray(specresc.data, dtype=float) if specresc is not None else None,
            
            'primary_header': h0,
            'sci_header': sci.header,
            'specres_header': specres.header if specres is not None else None,
        }

        # set data depending on product type
        if product_type == 'primary':
            # raw detector frame: (rows, cols), no dispersion solution
            d['wave'] = None
            d['shape'] = None if d['flux'] is None else d['flux'].shape
        else:
            if wave is not None:
                d['wave'] = np.asarray(wave.data, dtype=float)      # explicit product grid
            else:
                d['wave'] = pixel_to_wavelength(sci.header, sci.data)  # legacy fallback

        # coordinates
        ra_str = h0.get('RA', '')
        dec_str = h0.get('DEC', '')
        if isinstance(ra_str, str) and isinstance(dec_str, str) and ra_str.strip() and dec_str.strip():
            try:
                c = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
                d['ra'], d['dec'] = float(c.ra.deg), float(c.dec.deg)
            except ValueError:
                d['ra'], d['dec'] = None, None
        else:
            d['ra'], d['dec'] = None, None

        # other extension data and headers
        for ext in ('ERR', 'GPCNT'):
            if ext in ext_names:
                d[ext.lower()] = hdul[ext].data
                d[ext.lower() + '_header'] = hdul[ext].header

    return d


def get_reduced_exposures(obs_date, suffixes=('ssc', 'cf', 'a'), root=pipeline_root):
    '''
    Return a dictionary of dictionaries, one per exposure, for a given obs date.
    '''
    exposures = {}
    primary_loaded = False

    shared_keys = ('object', 'exptime', 'airmass', 'gain', 'grating_angle',
                   'pupil_start', 'pupil_end', 'ra', 'dec', 'ngroups', 'cfw')

    def seed(exp_id, product):
        '''Record shared metadata the first time we see this exposure.'''
        if exp_id not in exposures:
            exposures[exp_id] = {'exposure_id': exp_id}
            exposures[exp_id].update({k: product[k] for k in shared_keys})

    for suffix in suffixes:
        try:
            primary_reduced_files, science_reduced_files = get_reduction_products(obs_date, suffix, root)
        except FileNotFoundError:
            print(f'  (no *{suffix}.fits found, skipping)')
            continue

        # the primary glob ignores suffix, so ingest these only once
        if not primary_loaded:
            for f in primary_reduced_files:
                exp_id = Path(f).name.split('.reduced')[0]
                product = exposure_dict(f, product_type='primary')
                seed(exp_id, product)
                exposures[exp_id]['primary'] = product
            primary_loaded = True

        for f in science_reduced_files:
            exp_id = Path(f).name.split('.reduced')[0]
            product = exposure_dict(f, product_type='science')
            seed(exp_id, product)
            # keep top-level flux/wave pointing at extracted science data,
            # not the raw detector frame
            if 'flux' not in exposures[exp_id]:
                exposures[exp_id]['flux'] = product['flux']
                exposures[exp_id]['wave'] = product['wave']
            exposures[exp_id][suffix] = product

    # sky exposures: 'object' names the corresponding target and needs updating
    for exp_id, exp in exposures.items():
        intermediate_data_products = [ext for ext in ('ssc', 'cf', 'a') if ext in exp]
        # rename sky frame exposures from OBJECT to SKY
        if ('ssc' not in intermediate_data_products) and (exp['object'] != 'ARC'):
            exp['object'] = 'SKY'

    return exposures


def sci_to_gpcnt_spectrum(data, gpm, sci_wave, gpcnt_wave):
    """
    Align the SCI data array to the GPCNT wavelength grid by trimming
    the blue-end extension added during rectification.
    
    Returns data_aligned with same shape as gpm.
    """
    sci_start = np.argmin(np.abs(sci_wave - gpcnt_wave[0]))
    sci_end = sci_start + gpm.shape[1]

    data_aligned = data[:, sci_start:sci_end]
    wavelength_aligned = sci_wave[sci_start:sci_end]

    return data_aligned, wavelength_aligned


def process_a_fits_reduced_data(sci_header, sci_data, gpcnt_header, gpcnt_data, exp_time, gain, full_map_sky_indices):
    '''
    steps for extracting wavelength and flux data for .a.fits reduced files:
        - remove extra (sky) fibers from science data (only for .a.fits files)
        - convert x-pixel to wavelength
        - align the science spectrum with its GPCNT array
        - compute Poisson noise error on spectra
    '''
    # REMOVE SKY FIBERS
    sci_data = np.delete(sci_data, full_map_sky_indices, axis=0)
    gpcnt_data = np.delete(gpcnt_data, full_map_sky_indices, axis=0)

    # X-PIXEL to WAVELENGTH
    sci_wave = pixel_to_wavelength(sci_header, sci_data)
    gpcnt_wave = pixel_to_wavelength(gpcnt_header, gpcnt_data)

    # Align SCI to GPCNT wavelength grid
    sci_data_aligned, sci_wave_aligned = sci_to_gpcnt_spectrum(sci_data, gpcnt_data, sci_wave, gpcnt_wave)

    # ERROR
    # Reconstruct the renormalization factor that was applied per fiber
    # (gpmarr.mean() / gpmarr), so we can get the pre-renormalization (per-pixel-averaged) flux, then redo the error propagation consistently
    gpm_mean_per_fiber = np.nanmean(gpcnt_data, axis=1, keepdims=True)
    renorm = gpm_mean_per_fiber.repeat(sci_data_aligned.shape[1], axis=1)
    mask = gpcnt_data > 0

    # un-normalized counts per pixel
    counts_per_pixel = np.full(sci_data_aligned.shape, np.nan, dtype=np.float32)
    counts_per_pixel[mask] = (sci_data_aligned[mask] / renorm[mask]) * exp_time

    # Poisson error
    # N_e = gain * N_ADU --> N_ADU = N_e / gain
    sigma_per_pixel = np.sqrt(np.abs(counts_per_pixel) / gain)   # ADU

    # divide by exp time to convert ADU into counts/s
    sci_data_sigma = np.full(sci_data_aligned.shape, np.nan, dtype=np.float32)
    sci_data_sigma[mask] = (sigma_per_pixel[mask] / exp_time) * renorm[mask]

    return sci_wave_aligned, sci_data_aligned, sci_data_sigma, gpcnt_wave, gpcnt_data

def find_matching_sky(target_exp, exposures, tol=1.0):
    '''
    Find the SKY exposure whose exptime matches target_exp's, within tol seconds.
    Returns the sky exposure dict, or None.
    '''
    for exp_id, exp in exposures.items():
        if (exp['object'] == 'SKY') and (abs(exp['exptime'] - target_exp['exptime']) <= tol) and (round(exp['grating_angle'],1) == round(target_exp['grating_angle'],1)):
            return exp
    return None

def process_reduced_data(sci_header, sci_data, a_wave=None, a_data_sigma=None, sky_a_wave=None, sky_a_sigma=None):
    '''
    Wavelength solution + error for the sky-subtracted reduced (ssc) data.
    '''
    sci_wave = pixel_to_wavelength(sci_header, sci_data)
    if a_wave is None or a_data_sigma is None:
        return sci_wave, sci_data, None

    var_obj = interp1d(a_wave, a_data_sigma.astype(np.float64)**2, axis=1, bounds_error=False, fill_value=np.nan)(sci_wave)

    if sky_a_wave is not None and sky_a_sigma is not None:
        var_sky = interp1d(sky_a_wave, sky_a_sigma.astype(np.float64)**2, axis=1, bounds_error=False, fill_value=np.nan)(sci_wave)
        total_var = var_obj + var_sky   # sky subtraction adds sky Poisson noise
    else:
        total_var = var_obj             # if no sky data passed, approximate variance with a lower bound: (pre-sky-sub) Poisson variance, interpolated onto the ssc grid. 

    return sci_wave, sci_data, np.sqrt(total_var)

def get_reduced_spectra(sci_exp, all_exp):
    '''
    Reduced flux + sigma (a-file variance + sky a-file variance) for one reduced science exposure.
    '''
    fmap = load_fiber_map_dict()

    a = sci_exp['a']
    a_wave, _, a_sigma, gpcnt_wave, gpcnt_data = process_a_fits_reduced_data(a['sci_header'], a['flux'], a['gpcnt_header'], a['gpcnt'], a['exptime'], a['gain'], fmap['sky_indices'])

    # find matching sky exposures
    sky_exp = find_matching_sky(sci_exp, all_exp)
    if sky_exp is not None:
        sa = sky_exp['a']
        sky_a_wave, _, sky_a_sigma, _, _ = process_a_fits_reduced_data(sa['sci_header'], sa['flux'], sa['gpcnt_header'], sa['gpcnt'],sa['exptime'], sa['gain'], fmap['sky_indices'])

    wave, flux_all, sigma_all = process_reduced_data(sci_exp['ssc']['sci_header'], sci_exp['ssc']['flux'], a_wave, a_sigma, sky_a_wave, sky_a_sigma) 
    return wave, flux_all, sigma_all, gpcnt_wave, gpcnt_data


def plot_science_reduction_results(obs_date, outdir=plot_ext_spectra_dir, smooth=10, show=False, root=pipeline_root):
    """
    Plot raw (a), continuum fit (cf), and reduced (ssc) spectra for every fiber, for every exposure on a given date. 

    obs_date (str): YYYYMMDD
    outdir (str): directory to save plots into
    smooth (int): over what number of data points to compute the median for smoothing
    """
    fmap = load_fiber_map_dict()
    sky_indices = fmap['sky_indices']

    exposures = get_reduced_exposures(obs_date, suffixes=('a', 'cf', 'ssc'), root=root)

    for exp_id, e in exposures.items():
        missing = [s for s in ('a', 'cf', 'ssc') if s not in e]
        if missing:
            print(f'{exp_id}: skipping, missing {missing}')
            continue

        save_dir = os.path.join(outdir, obs_date, exp_id)
        os.makedirs(save_dir, exist_ok=True)

        a = e['a']
        a_wave, a_data, a_data_sigma, gpcnt_wave, gpcnt_data = process_a_fits_reduced_data(a['sci_header'], a['flux'], a['gpcnt_header'], a['gpcnt'], a['exptime'], a['gain'], sky_indices)

        ssc_wave, ssc_data, ssc_data_sigma = process_reduced_data(e['ssc']['sci_header'], e['ssc']['flux'], a_wave, a_data_sigma)
        cf_wave, cf_data = e['cf']['wave'], e['cf']['flux']

        # fiber axes must line up after sky removal
        assert a_data.shape[0] == ssc_data.shape[0] == cf_data.shape[0], \
            f'fiber mismatch: a={a_data.shape[0]}, ssc={ssc_data.shape[0]}, cf={cf_data.shape[0]}'

        nfib = ssc_data.shape[0]
        print(f'{exp_id} ({e["object"]}): plotting {nfib} fibers -> {save_dir}')

        for fiber in range(nfib):
            flux_a = a_data[fiber, :]
            err_a = a_data_sigma[fiber, :]
            flux_cf = cf_data[fiber, :]
            flux_ssc = ssc_data[fiber, :]
            err_ssc = ssc_data_sigma[fiber, :]
            gpm_arr = gpcnt_data[fiber, :]
            mean_gp = np.nanmean(gpm_arr)

            med_a = median_filter(np.nan_to_num(flux_a).astype(np.float32), size=smooth, mode='reflect')
            med_ssc = median_filter(np.nan_to_num(flux_ssc).astype(np.float32), size=smooth, mode='reflect')

            finite = flux_a[np.isfinite(flux_a)]
            if finite.size < 10:
                continue
            lower_lim = np.percentile(finite, 1)
            upper_lim = np.percentile(finite, 95)

            fig, (ax_flux, ax_gpm) = plt.subplots(2, 1, figsize=(10, 4), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0})
            fig.suptitle(f'{e["object"]} --- fiber #{fiber}', fontsize=15)

            # --- top: flux ---
            ax_flux.fill_between(a_wave, flux_a - err_a, flux_a + err_a, step='mid', color='grey', alpha=0.3, linewidth=0.6, zorder=0)
            ax_flux.step(a_wave, flux_a, where='mid', color='grey', alpha=0.5, lw=0.6, zorder=1)
            ax_flux.step(a_wave, med_a, where='mid', color='black', lw=1.5, label='raw', zorder=3)
            ax_flux.step(cf_wave, flux_cf, where='mid', color='red', ls='dashed', lw=1.5, label='continuum fit', zorder=4)
            ax_flux.fill_between(ssc_wave, flux_ssc - err_ssc, flux_ssc + err_ssc, step='mid', color='blue', alpha=0.2, linewidth=0.6)
            ax_flux.step(ssc_wave, flux_ssc, where='mid', color='blue', alpha=0.4, lw=0.5, zorder=2)
            ax_flux.step(ssc_wave, med_ssc, where='mid', color='blue', lw=1.5, label='reduced', zorder=5)
            ax_flux.set_ylabel('counts / s', fontsize=15, labelpad=15)
            ax_flux.set_ylim(lower_lim, upper_lim)
            ax_flux.legend(fontsize=12, loc='upper left')

            # --- bottom: number of good pixels ---
            ax_gpm.step(gpcnt_wave, gpm_arr, where='mid', color='grey', alpha=0.8, linewidth=0.6)
            ax_gpm.fill_between(gpcnt_wave, 0, gpm_arr, step='mid', color='black', alpha=0.15)
            ax_gpm.axhline(mean_gp, color='black', lw=1, linestyle='dashed')
            # ax_gpm.text(x=np.median(gpcnt_wave), y=mean_gp, s='mean # of good pixels',ha='center', va='bottom', color='black', fontsize=10)
            ax_gpm.set_ylabel('# good pixels', fontsize=12, labelpad=15)
            ax_gpm.set_xlabel(r'Observed Wavelength [$\AA$]', fontsize=15, labelpad=15)
            ax_gpm.set_xlim(np.min(a_wave), np.max(a_wave))

            plt.savefig(os.path.join(save_dir, f'fiber_{fiber:03d}_reduction.png'), dpi=200, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close()


def compute_resolution_percentiles(coeffs, hdr, wave):
    coeffs = np.atleast_2d(np.asarray(coeffs, float))  # (n_fibres, deg+1)
    wave = np.asarray(wave, float)
    # lo, hi = hdr['SPRESWLO'], hdr['SPRESWHI']

    fwhm = np.vstack([np.polyval(c, wave) for c in coeffs])
    R = wave[None, :] / fwhm
    # R[:, (wave < lo) | (wave > hi)] = np.nan
    R[fwhm <= 0] = np.nan

    R_med = np.nanmedian(R, axis=0)
    R_16 = np.nanpercentile(R, 16, axis=0)
    R_84 = np.nanpercentile(R, 84, axis=0)
    for row in (R_med, R_16, R_84):
        g = np.isfinite(row)
        if g.any() and not g.all():
            row[~g] = np.interp(wave[~g], wave[g], row[g])
    return R_med, R_16, R_84