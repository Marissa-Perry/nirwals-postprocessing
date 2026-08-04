import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import os

from pypeit.core.telluric import read_telluric_pca
from pypeit.core import telluric as _tell
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
from pypeit.core.telluric import Telluric, init_star_model, eval_star_model, init_poly_model, eval_poly_model
from pypeit.core import standard, flux_calib
from pathlib import Path

# -------- filepath params ----------
this_dir = Path(__file__).resolve().parent     # .../post_reduction_processing/scripts
proj_dir = this_dir.parent                     # .../post_reduction_processing

data_dir = proj_dir / 'data'
output_dir = proj_dir / 'outputs'
plot_dir = output_dir / 'plots'

os.makedirs(plot_dir, exist_ok=True)
plot_tell_corr_dir = os.path.join(plot_dir, 'telluric_correction')
os.makedirs(plot_tell_corr_dir, exist_ok=True)
# -----------------------------------

# PypeIt PCA decomposition of the telluric models across all observatories
TELLPCA_FILE = 'TellPCA_3000_26000_R10000.fits'

# spectral types available in PypeIt's Schmidt-Kaler (1982) table
SCHMIDT_KALER_TYPES = ['B0','B1','B2','B3','B5','B6','B7','B8','B9',
                       'A0','A1','A2','A3','A5','A7','A8',
                       'F0','F2','F5','F8','G0','G2','G5','G8']

def load_model_transmission(wave_min, wave_max, telgridfile=TELLPCA_FILE):
    '''
    Mean model atmospheric transmission from PypeIt's telluric PCA file.

    Returns (wave_grid, transmission) over [wave_min, wave_max].
    '''
    tell_dict = read_telluric_pca(telgridfile, wave_min=wave_min, wave_max=wave_max)
    wave_grid = tell_dict['wave_grid']
    # mean PCA component -> transmission (transform used in Telluric.sort_telluric)
    transmission = np.exp(-np.sinh(tell_dict['tell_pca'][0]))
    return wave_grid, transmission

def telluric_windows(wave, threshold=0.95, telgridfile=TELLPCA_FILE, min_width=5):
    '''
    Define telluric windows from the model atmosphere, clipped to this
    spectrum's coverage. Windows are where model transmission < threshold.

    wave (1D arr): your data's wavelength grid [A]
    threshold (float): transmission below this counts as telluric
    min_width (int): drop windows narrower than this many Angstrom
    '''
    lo, hi = float(np.nanmin(wave)), float(np.nanmax(wave))
    wave_grid, transmission_curve = load_model_transmission(lo, hi, telgridfile)

    mask = transmission_curve < threshold
    change = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = wave_grid[np.where(change == 1)[0]]
    ends = wave_grid[np.where(change == -1)[0] - 1]
    windows = [(float(s), float(e)) for s, e in zip(starts, ends) if (e - s) >= min_width]

    return windows

def query_star_properties(star_name):
    '''
    Get spectral type, V mag, and coordinates from SIMBAD.

    star_name (str): e.g. 'HIP 105164'
    Returns dict with 'sp_type', 'v_mag', 'ra', 'dec'
    '''
    s = Simbad()
    s.add_votable_fields('sp_type', 'V', 'ra', 'dec')
    t = s.query_object(star_name)
    if t is None or len(t) == 0:
        raise ValueError(f'SIMBAD returned nothing for {star_name!r}')

    cols = {c.lower(): c for c in t.colnames}
    def get(key):
        return t[cols[key]][0] if key in cols else None
    
    star_properties_dict = {'name': star_name,
                            'sp_type': str(get('sp_type')),
                            'v_mag': float(get('v')),
                            'ra': float(get('ra')),
                            'dec': float(get('dec')),
                            'colnames': t.colnames}

    return star_properties_dict


def to_schmidt_kaler_type(sp_type, valid=SCHMIDT_KALER_TYPES):
    '''
    Map a SIMBAD spectral type ('B7III') to the nearest entry in PypeIt's
    Schmidt-Kaler table ('B7'). Raises if the class isn't covered (K, M).
    '''
    letter = sp_type[0].upper()
    if letter not in 'OBAFG':
        raise ValueError(f'{sp_type}: Schmidt-Kaler table only covers B-G (available: {valid})')
    # take letter + leading digits, drop luminosity class
    digits = ''
    for ch in sp_type[1:]:
        if ch.isdigit():
            digits += ch
        else:
            break
    if not digits:
        raise ValueError(f'{sp_type}: no subclass digit found')

    base = letter + digits
    if base in valid:
        return base
    # round to the nearest available subclass within the same letter
    same = [v for v in valid if v[0] == letter]
    if not same:
        raise ValueError(f'{sp_type}: no {letter} types in table')
    nearest = min(same, key=lambda v: abs(int(v[1:]) - int(digits)))
    print(f'  ({sp_type} -> {nearest}, nearest available)')
    return nearest

def fit_telluric_star_model(star_flux, star_wave, star_ivar, star_gpm, star_props, airmass, exptime,
                            telgridfile=TELLPCA_FILE, resln_guess=4000., polyorder=5, sn_clip=30.0, maxiter=3,
                            pix_shift_bounds=(-10.0, 10.0), pix_stretch_bounds=(0.95, 1.05), hydrogen_mask_wid=20., disp=True):
    '''
    Fit PypeIt's star object model + telluric transmission to a standard star.
    Mirrors pypeit.core.telluric.star_telluric, but on arrays (no spec1dfile).

    star_props: dict from query_star_properties (uses sp_type, v_mag, ra, dec)

    Returns dict with 'transmission', 'star_model', 'TelObj', 'mask_recomb', fit params
    '''
    star_type = to_schmidt_kaler_type(star_props['sp_type'])

    # model standard-star SED (Kurucz, or Vega for A0)
    std_spec = standard.get_standard_spectrum(
        spectral_type=star_type, V_mag=star_props['v_mag'],
        ra=star_props['ra'], dec=star_props['dec'])

    polyorder_vec = np.full(1, polyorder)
    obj_params = dict(
        std_spec=std_spec,
        airmass=float(airmass),
        delta_coeff_bounds=(-20.0, 20.0),
        minmax_coeff_bounds=(-5.0, 5.0),
        polyorder_vec=polyorder_vec,
        exptime=float(exptime),
        func='legendre',
        model='exp',
        sigrej=3.0,
        std_ra=std_spec.meta['ra_deg'],
        std_dec=std_spec.meta['dec_deg'],
        std_name=std_spec.meta['Name'],
        std_cal=std_spec.meta['File'],
        output_meta_keys=('airmass', 'polyorder_vec', 'exptime', 'func',
                          'std_ra', 'std_dec', 'std_cal'),
        debug=False,
    )

    # mask bad pixels + stellar hydrogen recombination lines
    mask_bad, mask_recomb, mask_tell = flux_calib.get_mask(star_wave, star_flux, star_ivar, star_gpm,
                                                           mask_hydrogen_lines=True, mask_helium_lines=False,
                                                           mask_telluric=False, hydrogen_mask_wid=hydrogen_mask_wid)
    mask_tot = mask_bad & mask_recomb & mask_tell

    TelObj = Telluric(
        star_wave.astype(float), star_flux.astype(float),
        star_ivar.astype(float), mask_tot,
        telgridfile, obj_params, init_star_model, eval_star_model,
        teltype='pca', resln_guess=resln_guess, sn_clip=sn_clip, maxiter=maxiter,
        pix_shift_bounds=pix_shift_bounds, pix_stretch_bounds=pix_stretch_bounds,
        debug=False, disp=disp,
    )
    TelObj.run(only_orders=None)

    m = TelObj.model[0]
    out = {
        'transmission': m['TELLURIC'],
        'star_model': m['OBJ_MODEL'],
        'wave': m['WAVE'],
        'success': bool(m['SUCCESS']),
        'chi2': float(m['CHI2']),
        'resln': float(m['TELL_RESLN']),
        'shift': float(m['TELL_SHIFT']),
        'stretch': float(m['TELL_STRETCH']),
        'mask_recomb': mask_recomb,
        'mask_tot': mask_tot,
        'star_type': star_type,
        'TelObj': TelObj
    }
    print(f"  SUCCESS={out['success']}  CHI2={out['chi2']:.1f}  "
          f"R={out['resln']:.0f}  shift={out['shift']:.2f}  stretch={out['stretch']:.4f}")
    return out


def fit_telluric_poly_model(star_flux, star_wave, star_ivar, star_gpm, airmass, exptime,
                            telgridfile=TELLPCA_FILE, resln_guess=4000., polyorder=3, sn_clip=30.0, maxiter=3,
                            pix_shift_bounds=(-10.0, 10.0), pix_stretch_bounds=(0.95, 1.05), func='legendre', 
                            model='exp', z_obj=0.0, mask_lyman_a=False, mask_hydrogen=True, hydrogen_mask_wid=20., disp=True):
    '''
    Fit PypeIt's polynomial object model + telluric transmission to a spectrum

    Fits the object continuum as (polynomial * telluric)

    Returns dict with 'transmission', 'poly_model', 'TelObj', 'mask_tot', fit params
    '''
    polyorder_vec = np.full(1, polyorder)
    obj_params = dict(
        z_obj=float(z_obj),
        mask_lyman_a=mask_lyman_a,
        airmass=float(airmass),
        delta_coeff_bounds=(-20.0, 20.0),
        minmax_coeff_bounds=(-5.0, 5.0),
        polyorder_vec=polyorder_vec,
        exptime=float(exptime),
        func=func,
        model=model,
        sigrej=3.0,
        output_meta_keys=('airmass', 'polyorder_vec', 'exptime', 'func'),
        debug=False,
    )

    # mask bad pixels (+ optionally stellar hydrogen recombination lines)
    mask_bad, mask_recomb, mask_tell = flux_calib.get_mask(
        star_wave, star_flux, star_ivar, star_gpm,
        mask_hydrogen_lines=mask_hydrogen, mask_helium_lines=False,
        mask_telluric=False, hydrogen_mask_wid=hydrogen_mask_wid)
    mask_tot = mask_bad & mask_recomb & mask_tell

    # poly_telluric restricts the fit to redward of Lyman-alpha
    if mask_lyman_a:
        mask_tot = mask_tot & (star_wave > 1216.15 * (1 + z_obj))

    TelObj = Telluric(
        star_wave.astype(float), star_flux.astype(float),
        star_ivar.astype(float), mask_tot,
        telgridfile, obj_params, init_poly_model, eval_poly_model,
        teltype='pca', resln_guess=resln_guess, sn_clip=sn_clip, maxiter=maxiter,
        pix_shift_bounds=pix_shift_bounds, pix_stretch_bounds=pix_stretch_bounds,
        debug=False, disp=disp,
    )
    TelObj.run(only_orders=None)

    m = TelObj.model[0]
    out = {
        'transmission': m['TELLURIC'],
        'poly_model': m['OBJ_MODEL'],
        'wave': m['WAVE'],
        'success': bool(m['SUCCESS']),
        'chi2': float(m['CHI2']),
        'resln': float(m['TELL_RESLN']),
        'shift': float(m['TELL_SHIFT']),
        'stretch': float(m['TELL_STRETCH']),
        'mask_recomb': mask_recomb,
        'mask_tot': mask_tot,
        'TelObj': TelObj
    }
    print(f"  SUCCESS={out['success']}  CHI2={out['chi2']:.1f}  "
          f"R={out['resln']:.0f}  shift={out['shift']:.2f}  stretch={out['stretch']:.4f}")
    return out


def apply_telluric_model(flux, sigma, wave, transmission, tell_wave, t_floor=0.15):
    '''
    Divide a galaxy spectrum by a model telluric transmission curve.
    Interpolates the transmission onto gal_wave if the grids differ.

    Returns dict with 'flux', 'sigma', 'transmission', 'good'
    '''
    flux = np.asarray(flux,  dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if len(tell_wave) == len(wave) and np.allclose(tell_wave, wave):
        T = transmission
    else:
        T = np.interp(wave, tell_wave, transmission, left=np.nan, right=np.nan)

    good = np.isfinite(T) & (T > t_floor)         
    flux_corr = np.full_like(flux,  np.nan)
    sigma_corr = np.full_like(sigma, np.nan)
    flux_corr[...,  good] = flux[...,  good] / T[good]  
    sigma_corr[..., good] = sigma[..., good] / T[good]

    tell_corr_dict = {'flux': flux_corr, 'sigma': sigma_corr, 'transmission': T, 'good': good}

    return tell_corr_dict


def windows_from_transmission(transmission, wave, threshold=0.95, min_width=5):
    '''
    Telluric windows where fitted transmission drops below threshold.
    transmission, wave: from fit_telluric_star_model (fit['transmission'], fit['wave'])
    '''
    mask = transmission < threshold
    change = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = wave[np.where(change == 1)[0]]
    ends = wave[np.where(change == -1)[0] - 1]
    return [(float(s), float(e)) for s, e in zip(starts, ends) if (e - s) >= min_width]


def plot_star_telluric_model_fit(star_wave, star_flux, fit, obj_dict, savepath=plot_tell_corr_dir, ylim=None, show=True):
    '''
    Star model fit diagnostic.
      top:    obs, full model (star x telluric), intrinsic star model
      middle: residual (obs - model)
      bottom: fitted transmission
    Shaded = hydrogen-masked regions excluded from the fit.
    '''
    wave_m = fit['wave']
    T = fit['transmission']
    star_model = fit['star_model']
    full_model = star_model * T                    # <- what the fit compares to the data

    # hydrogen-masked windows
    masked = ~fit['mask_recomb']
    change = np.diff(np.concatenate([[0], masked.astype(int), [0]]))
    starts = star_wave[np.where(change == 1)[0]]
    ends = star_wave[np.where(change == -1)[0] - 1]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 5), sharex=True, gridspec_kw={'hspace': 0, 'height_ratios': [3, 1, 1]})
    ax1.set_title(obj_dict['object'], fontsize=14, pad=15)

    for ax in (ax1, ax2):
        for s, e in zip(starts, ends):
            ax.axvspan(s, e, color='grey', alpha=0.35, zorder=0)

    # --- top: obs + models ---
    ax1.step(star_wave, star_flux, where='mid', color='black', alpha=0.5, lw=0.8, label='observed', zorder=1)
    ax1.step(wave_m, star_model, where='mid', color='black', lw=1.2, ls='dashed', label=r'star model', zorder=2)
    ax1.step(wave_m, full_model, where='mid', color='red', lw=1.2, label=r'star $\times$ tellurics (full model)', zorder=3)
    ax1.set_ylabel('counts / s', fontsize=13, labelpad=15)
    ax1.set_xlim(np.nanmin(wave_m), np.nanmax(wave_m))
    if ylim is not None:
        ax1.set_ylim(*ylim)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.text(0.02, 0.92, 'masked H recombination lines', transform=ax1.transAxes, fontsize=9, color='0.3')

    # --- middle: residuals ---
    resid = (star_flux - full_model) / np.nanmedian(full_model)
    ax2.set_ylabel('res', fontsize=10, labelpad=10)
    ax2.set_ylim(-0.2, 0.2)
    ax2.step(star_wave, resid, where='mid', color='black', alpha=0.5, lw=0.8, zorder=1)
    ax2.axhline(0, color='red', lw=1.0)

    # --- bottom: fitted transmission ---
    ax3.fill_between(wave_m, T, step='mid', color='grey', alpha=0.7, zorder=1)
    ax3.step(wave_m, T, where='mid', color='black', lw=0.6, zorder=2)
    ax3.set_ylim(-0.01, 1.1)
    ax3.set_ylabel('transmission', fontsize=10, labelpad=28)
    ax3.set_xlabel(r'Observed Wavelength [$\AA$]', fontsize=13, labelpad=15)

    plt.tight_layout()
    if savepath:
        save_dir = os.path.join(savepath, obj_dict['exposure_id'])
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.join(save_dir, 'telluric_model.png')
        plt.savefig(filename, dpi=500, bbox_inches='tight')
    plt.show() if show else plt.close()


def plot_poly_telluric_model_fit(star_wave, star_flux, fit, obj_dict, savepath=plot_tell_corr_dir, ylim=None, show=True):
    '''
    Polynomial model fit diagnostic.
      top:    obs, full model (poly x telluric), polynomial continuum model
      middle: residual (obs - model)
      bottom: fitted transmission
    Shaded = hydrogen-masked regions excluded from the fit (if mask_hydrogen was on).
    '''
    wave_m = fit['wave']
    T = fit['transmission']
    poly_model = fit['poly_model']
    full_model = poly_model * T                    # <- what the fit compares to the data

    # hydrogen-masked windows (empty if mask_hydrogen=False)
    masked = ~fit['mask_recomb']
    change = np.diff(np.concatenate([[0], masked.astype(int), [0]]))
    starts = star_wave[np.where(change == 1)[0]]
    ends = star_wave[np.where(change == -1)[0] - 1]
    has_masked = len(starts) > 0

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 5), sharex=True, gridspec_kw={'hspace': 0, 'height_ratios': [3, 1, 1]})
    ax1.set_title(obj_dict['object'], fontsize=14, pad=15)

    for ax in (ax1, ax2):
        for s, e in zip(starts, ends):
            ax.axvspan(s, e, color='grey', alpha=0.35, zorder=0)

    # --- top: obs + models ---
    ax1.step(star_wave, star_flux, where='mid', color='black', alpha=0.5, lw=0.8, label='observed', zorder=1)
    ax1.step(wave_m, poly_model, where='mid', color='black', lw=1.2, ls='dashed', label=r'polynomial model', zorder=2)
    ax1.step(wave_m, full_model, where='mid', color='red', lw=1.2, label=r'poly $\times$ tellurics (full model)', zorder=3)
    ax1.set_ylabel('counts / s', fontsize=13, labelpad=15)
    ax1.set_xlim(np.nanmin(wave_m), np.nanmax(wave_m))
    if ylim is not None:
        ax1.set_ylim(*ylim)
    ax1.legend(fontsize=10, loc='upper right')
    if has_masked:
        ax1.text(0.02, 0.92, 'masked H recombination lines', transform=ax1.transAxes, fontsize=9, color='0.3')

    # --- middle: residuals ---
    resid = (star_flux - full_model) / np.nanmedian(full_model)
    ax2.set_ylabel('res', fontsize=10, labelpad=10)
    ax2.set_ylim(-0.3, 0.3)
    ax2.step(star_wave, resid, where='mid', color='black', alpha=0.5, lw=0.8, zorder=1)
    ax2.axhline(0, color='red', lw=1.0)

    # --- bottom: fitted transmission ---
    ax3.fill_between(wave_m, T, step='mid', color='grey', alpha=0.7, zorder=1)
    ax3.step(wave_m, T, where='mid', color='black', lw=0.6, zorder=2)
    ax3.set_ylim(-0.01, 1.1)
    ax3.set_ylabel('transmission', fontsize=10, labelpad=22)
    ax3.set_xlabel(r'Observed Wavelength [$\AA$]', fontsize=13, labelpad=15)

    plt.tight_layout()
    if savepath:
        save_dir = os.path.join(savepath, obj_dict['exposure_id'])
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.join(save_dir, 'telluric_model_poly.png')
        plt.savefig(filename, dpi=500, bbox_inches='tight')
    plt.show() if show else plt.close()
    

def plot_telluric_correction(wave, raw_flux, corr_flux, obj_dict, savepath=plot_tell_corr_dir, smooth=10, show=True):
    '''
    Corrected vs uncorrected coadd, with the corr/raw ratio below.
    '''
    sm_raw = median_filter(np.nan_to_num(raw_flux).astype(np.float32), size=smooth, mode='reflect')
    sm_corr = median_filter(np.nan_to_num(corr_flux).astype(np.float32), size=smooth, mode='reflect')
    transmission = raw_flux / corr_flux

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={'hspace': 0, 'height_ratios': [3, 1]})
    ax1.set_title(obj_dict['object'], fontsize=14, pad=15)
    ax1.step(wave, raw_flux, where='mid', color='blue', alpha=0.2, lw=0.6)
    ax1.step(wave, sm_raw, where='mid', color='blue', alpha=0.8, lw=1.3, label='uncorrected')
    ax1.step(wave, corr_flux, where='mid', color='black', alpha=0.4, lw=0.6)
    ax1.step(wave, sm_corr, where='mid', color='black', lw=1.3, label='telluric corrected')
    ax1.set_ylim(np.nanpercentile(corr_flux, 0.1), np.nanpercentile(corr_flux, 99.9))
    ax1.set_ylabel('counts / s', fontsize=13, labelpad=15)
    ax1.legend(fontsize=12)

    wave_grid, model_transmission = load_model_transmission(wave.min(), wave.max())
    ax2.step(wave_grid, model_transmission, where='mid', color='grey', lw=0.8, zorder=1, label='model')
    ax2.step(wave, transmission, where='mid', color='black', lw=1.0, zorder=2, label='measured')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlim(wave.min(), wave.max())
    ax2.set_ylabel('transmission', fontsize=10, labelpad=22)
    ax2.set_xlabel(r'Observed Wavelength [$\AA$]', fontsize=13, labelpad=15)
    ax2.legend(fontsize=10)


    plt.tight_layout()
    if savepath:
        save_dir = os.path.join(savepath, obj_dict['exposure_id'])
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.join(save_dir, 'telluric_correction.png')
        plt.savefig(filename, dpi=500, bbox_inches='tight')
    plt.show() if show else plt.close()