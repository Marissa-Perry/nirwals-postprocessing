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
        ext_names = [x.name for x in hdul]
        # observation metadata now lives in OBSINFO (fall back to PRIMARY for older files)
        meta = hdul['OBSINFO'].header if 'OBSINFO' in ext_names else hdul['PRIMARY'].header

        def hdr(key, cast=float):
            val = meta.get(key)
            if val is None:
                return None
            try:
                return cast(val)
            except (TypeError, ValueError):
                return None

        # flux extension: FLUX (new) -> SCI (old) -> primary
        if 'FLUX' in ext_names:
            sci = hdul['FLUX']
        elif 'SCI' in ext_names:
            sci = hdul['SCI']
        else:
            sci = hdul[0]
        wave = hdul['WAVE'] if 'WAVE' in ext_names else None
        specres = hdul['SPECRES'] if 'SPECRES' in ext_names else None
        specresd = hdul['SPECRESD'] if 'SPECRESD' in ext_names else None
        specresc = hdul['SPECRES_COEFF_PER_FIBER'] if 'SPECRES_COEFF_PER_FIBER' in ext_names else None

        # airmass value from metadata is invalid, computing using telescope altitude
        tel_alt = float(meta['TELALT'])
        airmass = 1.0 / np.sin(np.radians(tel_alt))

        d = {
            'file': filepath,
            'filename': Path(filepath).name,
            'product_type': product_type,
            'object': meta['OBJECT'],
            'exp_type': meta['EXPTYPE'],
            'exptime': hdr('EXPTIME'),
            'airmass': airmass, 
            'gain': hdr('GAIN'),
            'grating_angle': hdr('GRRANGLE'),
            'pupil_start': hdr('PUPSTA'),
            'pupil_end': hdr('PUPEND'),
            'ngroups': hdr('NGROUPS'),
            'cfw': str(meta.get('CFWCURZ', ''))[-12:],

            'flux': sci.data,
            'spec_res': np.asarray(specres.data, dtype=float) if specres is not None else None,
            'specresd': np.asarray(specresd.data, dtype=float) if specresd is not None else None,
            'specres_coeffs': np.asarray(specresc.data, dtype=float) if specresc is not None else None,

            'primary_header': meta,
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
        ra_str = meta.get('RA', '')
        dec_str = meta.get('DEC', '')
        if isinstance(ra_str, str) and isinstance(dec_str, str) and ra_str.strip() and dec_str.strip():
            try:
                c = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
                d['ra'], d['dec'] = float(c.ra.deg), float(c.dec.deg)
            except ValueError:
                d['ra'], d['dec'] = None, None
        else:
            d['ra'], d['dec'] = None, None

        # per-pixel arrays: IVAR/MASK/SKYCORR (new) plus legacy ERR/GPCNT
        for ext in ('IVAR', 'MASK', 'SKYCORR', 'ERR', 'GPCNT'):
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

    # subset of keys in the outer-most dict
    # these are invariant across every reduction product (i.e., ssc, ss, cf, a) of a given exposure
    shared_keys = (
        'object',
        'exp_type',
        'exptime',
        'airmass',
        'gain',
        'grating_angle',
        'pupil_start',
        'pupil_end',
        'ra',
        'dec',
        'ngroups',
        'cfw'
    )

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
        # identifying sky exposures based on whether they have a ssc intermediate data product
        if ('ssc' not in intermediate_data_products) and (exp['object'] != 'ARC'):
            exp['object'] = 'SKY'

    return exposures


def variance_from_ivar(ivar, sky_indices=None):
    """
    Per-pixel variance (1/IVAR) from a reduced product's IVAR array. Optionally
    drops sky fibers so the result matches the object-fiber astrometry.
    """
    ivar = np.asarray(ivar, dtype=float)
    if sky_indices is not None and ivar.ndim == 2:
        ivar = np.delete(ivar, sky_indices, axis=0)
    var = np.full(ivar.shape, np.nan)
    good = ivar > 0
    var[good] = 1.0 / ivar[good]
    return var


def find_matching_sky(target_exp, exposures, tol=1.0):
    '''
    Find the SKY exposure whose exptime matches target_exp's, within tol seconds.
    Returns the sky exposure dict, or None.
    '''
    for exp_id, exp in exposures.items():
        if (exp['object'] == 'SKY') and (abs(exp['exptime'] - target_exp['exptime']) <= tol) and (round(exp['grating_angle'],1) == round(target_exp['grating_angle'],1)):
            return exp
    return None


def get_reduced_spectra(sci_exp, all_exp):
    '''
    Flux + sigma for one reduced science exposure, built from the reduced IVAR:
    the object variance (from the 'a' product) plus, when a matching sky exposure
    exists, the sky variance -- both interpolated onto the ssc grid. Sky fibers
    are removed so the arrays match the object-fiber astrometry.
    '''
    fmap = load_fiber_map_dict()
    sky_idx = fmap['sky_indices']

    ssc = sci_exp['ssc']
    wave = ssc['wave']
    flux_all = ssc['flux']

    # object variance from the 'a' product, interpolated onto the ssc grid
    a = sci_exp['a']
    var_obj = interp1d(np.asarray(a['wave'], float), variance_from_ivar(a['ivar'], sky_idx), axis=1, bounds_error=False, fill_value=np.nan)(wave)

    # keep flux on the same (object) fibers as the variance
    if flux_all.shape[0] != var_obj.shape[0]:
        flux_all = np.delete(flux_all, sky_idx, axis=0)

    total_var = var_obj
    sky_exp = find_matching_sky(sci_exp, all_exp)
    if sky_exp is not None:
        sa = sky_exp['a']
        var_sky = interp1d(np.asarray(sa['wave'], float), variance_from_ivar(sa['ivar'], sky_idx), axis=1, bounds_error=False, fill_value=np.nan)(wave)
        total_var = var_obj + var_sky   # sky subtraction adds sky Poisson noise

    return wave, flux_all, np.sqrt(total_var)

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
        a_wave = np.asarray(a['wave'], dtype=float)
        a_data = np.delete(np.asarray(a['flux'], dtype=float), sky_indices, axis=0)  # remove sky fibers (248 --> 212 fibers)
        a_data_sigma = np.sqrt(variance_from_ivar(a['ivar'], sky_indices))
        a_mask = (np.delete(np.asarray(a['mask'], dtype=float), sky_indices, axis=0)
                  if a.get('mask') is not None else np.zeros_like(a_data))

        ssc_wave, ssc_data, ssc_data_sigma = get_reduced_spectra(e, exposures)
        cf_wave = np.asarray(e['cf']['wave'], dtype=float)
        cf_data = np.asarray(e['cf']['flux'], dtype=float)

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
            mask_arr = a_mask[fiber, :]

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
            ax_gpm.step(a_wave, mask_arr, where='mid', color='grey', alpha=0.8, linewidth=0.6)
            ax_gpm.fill_between(a_wave, 0, mask_arr, step='mid', color='black', alpha=0.15)
            ax_gpm.set_ylabel('bad-pixel mask', fontsize=12, labelpad=15)
            ax_gpm.set_xlabel(r'Observed Wavelength [$\AA$]', fontsize=15, labelpad=15)
            ax_gpm.set_xlim(np.min(a_wave), np.max(a_wave))

            plt.savefig(os.path.join(save_dir, f'fiber_{fiber:03d}_reduction.png'), dpi=200, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close()