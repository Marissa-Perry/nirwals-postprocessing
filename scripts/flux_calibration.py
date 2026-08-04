import numpy as np
from astropy.io import fits
from scipy.interpolate import make_smoothing_spline
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os

from .functions import select_bright_fibers

# -------- filepath params ----------
this_dir = Path(__file__).resolve().parent     # .../post_reduction_processing/scripts
proj_dir = this_dir.parent                     # .../post_reduction_processing

data_dir = proj_dir / 'data'
output_dir = proj_dir / 'outputs'
plot_dir = output_dir / 'plots'

os.makedirs(plot_dir, exist_ok=True)
plot_flux_cal_dir = os.path.join(plot_dir, 'flux_calibration')
os.makedirs(plot_flux_cal_dir, exist_ok=True)
rayner_dir = data_dir / 'Rayner_standard_data'
throughput_curve_dir = output_dir / 'throughput_curves'
os.makedirs(throughput_curve_dir, exist_ok=True)
processed_data_dir = output_dir / 'processed_data'
os.makedirs(processed_data_dir, exist_ok=True)
# -----------------------------------

# physical constants
h_plank_const = 6.63e-34  # [J * s]
lightspeed = 3e8         # [m / s]

# SALT pupil geometry
mirror_collecting_area = ((11 / 2)**2 * np.pi) - ((3 / 2)**2 * np.pi)   # [m^2]
# grating dispersion [nm/pixel]
dispersion_map = {37.0: 0.06869, 29.5: 0.07449, 25.0: 0.07795, 50.0: 0.05529}
# rounded grating angle and associated tags
GA_tag = {25.0: 'GA25', 29.5: 'GA295', 37.0: 'GA37', 50.0: 'GA50'}


def read_spec_phot_standard(obj_name, std_star_directory=rayner_dir):
    '''
    Read spec-phot standard FITS (from Rayner+2009): wave [A], flux [W/m^2/um], err
    '''
    obj_name_txt = obj_name.replace(' ', '')
    path = os.path.join(std_star_directory, f'*{obj_name_txt}.fits')
    file = glob.glob(path)[0]

    with fits.open(file) as hdul:
        h = hdul[0].header
        d = hdul[0].data

    spec_phot_dict = {'wave': d[0] * 1e4, 
                      'flux': d[1], 
                      'err': d[2],
                      'object': h['OBJECT'].replace('HD ', 'HD-')
                      }
    
    return spec_phot_dict


def compute_throughput(spec_phot_obs, spec_phot_name, tell_windows=None,
                       perf=0.01, cut=5000.0, smooth=15, lam=800000,
                       cutw=81, cutw2=None, fibfil=0.62):
    '''
    Compute system throughput = observed_star / spec-phot standard,
    fit with a smoothing spline.
    '''
    # --- read both spectra via the helper readers ---
    std = read_spec_phot_standard(spec_phot_name)  # reference (Rayner) spectrum

    data = spec_phot_obs['flux'].astype(float)
    wave = spec_phot_obs['wave']
    GA = spec_phot_obs['grating_angle']
    ga_r = round(GA, 1)
    # texp = spec_phot_obs['ngroups'] * ft if spec_phot_obs['ngroups'] is not None else np.nan
    texp = spec_phot_obs['exptime']

    # mask extreme pixels
    data[(data > cut) | (data < -cut)] = np.nan

    # select + sum bright (star) fibers
    ws = select_bright_fibers(data, frac=perf)
    sumf = np.nansum(data[ws, :], axis=0)

    # --- reference standard -> J/pixel on obs grid ---
    telarea = mirror_collecting_area * (spec_phot_obs['pupil_start'] + spec_phot_obs['pupil_end']) / 2.0
    sdisp = dispersion_map[ga_r] / 1000.0
    rflux2 = std['flux'] * telarea * sdisp * texp
    rflux3 = np.interp(wave, std['wave'], rflux2)

    # --- observed star -> J/pixel ---
    Jenergy_w = h_plank_const * lightspeed / (wave * 1e-10)
    oflux = sumf * Jenergy_w * texp * spec_phot_obs['gain'] / fibfil

    # trim edges, smooth
    if cutw2 is None:
        cutw2 = 113 if ga_r == 50.0 else 80
    wave_cut = wave[cutw:-cutw2]
    cflux = oflux / rflux3
    cflux_cut = cflux[cutw:-cutw2]

    # build spline weights: downweight telluric windows + drop non-finite
    w = np.ones_like(wave_cut)
    if tell_windows is not None:
        for lo, hi in tell_windows:
            w[(wave_cut >= lo) & (wave_cut <= hi)] = 1e-10   # near-zero weight

    finite = np.isfinite(cflux_cut) & np.isfinite(wave_cut)
    spl = make_smoothing_spline(wave_cut[finite], cflux_cut[finite], w=w[finite], lam=lam)
    throughput = spl(wave_cut)

    kernel = np.ones(smooth) / smooth
    soflux = np.convolve(oflux[cutw:-cutw2], kernel, mode='same')

    throughput_dict = {
        'wave': wave_cut, 
        'throughput': throughput,
        'GA': GA, 
        'ga_tag': GA_tag.get(ga_r, f'GA{ga_r:g}'),
        'std_name': std['object'], 
        'obs_name': spec_phot_obs['object'],
        'cfw': spec_phot_obs['cfw'],
        'wave_full': wave, 
        'ratio': cflux, 
        'cutw': cutw, 
        'cutw2': cutw2,
        'rflux3': rflux3, 
        'soflux': soflux, 
        'oflux': oflux,
        'tell_windows': tell_windows
    }

    return throughput_dict


def save_throughput(result, outdir=throughput_curve_dir):
    '''
    Write the throughput curve to a CSV in post_reduction_processing/throughput_curves.
    '''
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fname = outdir / f"{result['std_name']}_{result['ga_tag']}_throughput.csv"
    np.savetxt(fname, np.column_stack((result['wave'], result['throughput'])), fmt='%f', delimiter=',')
    print(f'wrote throughput curve: {fname}')
    return fname


def plot_throughput_fit(result, obj_dict, savepath=plot_flux_cal_dir, show=True):
    r = result
    cutw, cutw2 = r['cutw'], r['cutw2']
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(r['wave'], r['ratio'][cutw:-cutw2], lw=0.8, label='ratio (obs / Rayner)')
    ax.plot(r['wave'], r['throughput'], color='black', ls='dashed', lw=1.5, label='spline throughput')

    # shade downweighted telluric windows
    if r['tell_windows'] is not None:
        for j, (lo, hi) in enumerate(r['tell_windows']):
            ax.axvspan(lo, hi, color='grey', alpha=0.25,
                    label='downweighted (telluric)' if j == 0 else None)

    ax.set_title(f"Rayner: {r['std_name']}  GA={r['GA']:.1f}", fontsize=13)
    ax.set_xlabel(r'Wavelength [$\AA$]', fontsize=13)
    ax.set_ylabel('throughput', fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    if savepath:
        save_dir = os.path.join(savepath, obj_dict['exposure_id'])
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.join(save_dir, 'throughput_computation.png')
        plt.savefig(filename, dpi=500, bbox_inches='tight')
    plt.show() if show else plt.close()


def plot_throughput_validation(result, obj_dict, savepath=plot_flux_cal_dir, show=True):
    '''
    Apply throughput back to the standard as validation.
    '''
    r = result
    std_cal = r['soflux'] / r['throughput']
    rflux_cut = np.interp(r['wave'], r['wave_full'], r['rflux3'])
    ratio = std_cal / rflux_cut

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})
    ax1.step(r['wave'], std_cal, where='mid', lw=1.2, label='throughput-corrected NIRWALS')
    ax1.step(r['wave'], rflux_cut, where='mid', color='black', alpha=0.8, lw=1.2, label='standard (Rayner+2009)')
    
    # for j, (lo, hi) in enumerate(r['tell_windows']):
    #     ax1.axvspan(lo, hi, color='grey', alpha=0.25,
    #                 label='downweighted (telluric)' if j == 0 else None)
    #     ax2.axvspan(lo, hi, color='grey', alpha=0.25)

    ax1.set_ylabel('Flux [J/pixel]', fontsize=13)
    ax1.set_ylim(np.nanpercentile(std_cal, 0.5), np.nanpercentile(std_cal, 99.9))
    ax1.legend(fontsize=11, loc='lower left')
    ax1.set_title(r['std_name'], fontsize=14, pad=15)
    ax2.scatter(r['wave'], ratio, marker='o', s=0.5, color='black')
    ax2.axhline(1, color='grey', ls='dashed', lw=1)
    ax2.set_ylim(0.9, 1.1)
    ax2.set_ylabel('ratio', fontsize=12)
    ax2.set_xlabel(r'Wavelength [$\AA$]', fontsize=13)
    plt.tight_layout()
    if savepath:
        save_dir = os.path.join(savepath, obj_dict['exposure_id'])
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.join(save_dir, 'throughput_validation.png')
        plt.savefig(filename, dpi=500, bbox_inches='tight')
    plt.show() if show else plt.close()


def load_throughput(throughput_file):
    curve = np.genfromtxt(throughput_file, delimiter=',')
    return curve[:, 0], curve[:, 1] 


def telescope_area(pupil_start, pupil_end):
    '''
    Effective collecting area [m^2] from pupil start to end.
    '''
    obsap = (pupil_start + pupil_end) / 2.0
    telarea = mirror_collecting_area * obsap
    return telarea


def flux_calibrate(wave, flux, sigma, throughput_wave, throughput_values,
                   gain, texp, pupil_start, pupil_end, cdelt1, fibfil=0.62):
    '''
    Convert counts/s spectrum to flux density F_lambda.

    wave (1D arr): wavelength grid [A]
    flux, sigma (arr): counts/s and its error; 1D or 2D (fiber, wave)
    throughput_wave/_values: system throughput curve (dimensionless)
    gain [e-/count], texp [s], pupil_start/end, cdelt1 [A/pixel]

    Returns (F_lambda, F_lambda_err), same shape as flux.
    '''
    # photon energy per wavelength bin
    Jenergy_w = h_plank_const * lightspeed / (wave * 1e-10)  # [J / photon]
    scale = Jenergy_w * texp * gain / fibfil                 # [J * s / count]
    F_jy = flux * scale       # [J/pixel]
    F_err_jy = sigma * scale  # ''

    # divide by throughput
    thru = np.interp(wave, throughput_wave, throughput_values)
    good = thru > 0
    F_jy_corr = np.full_like(F_jy, np.nan, dtype=np.float64)
    F_err_jy_corr= np.full_like(F_err_jy, np.nan, dtype=np.float64)
    F_jy_corr[good] = F_jy[good] / thru[good]
    F_err_jy_corr[good] = F_err_jy[good] / thru[good]

    # J/pixel -> F_lambda: divide out time, area, dispersion
    telarea = telescope_area(pupil_start, pupil_end)
    denom = telarea * texp * cdelt1
    F_si = F_jy_corr / denom          # [W/m^2/A]
    F_err_si = F_err_jy_corr / denom   # ''

    # convert to cgs units
    F_cgs = F_si * 1e3           # [erg/s/cm^2/A]
    F_err_cgs = F_err_si * 1e3   # ''

    return F_cgs, F_err_cgs
    

def flux_calibrate_exposure(wave, flux, sigma, exp, throughput_file, fibfil=0.62):
    '''
    Flux-calibrate using metadata from a get_reduced_exposures() dict.

    exp: dict with 'gain','exptime','pupil_start','pupil_end','grating_angle',
         and 'ssc'/'sci_header' for CDELT1.
    '''
    tw, tv = load_throughput(throughput_file)
    cdelt1 = float(exp['ssc']['sci_header']['CDELT1'])

    F, Ferr = flux_calibrate(wave, flux, sigma, tw, tv,
                             gain=exp['gain'], texp=exp['exptime'], 
                             pupil_start=exp['pupil_start'], pupil_end=exp['pupil_end'],
                             cdelt1=cdelt1, fibfil=fibfil)
    return F, Ferr

def write_fluxcal_fits(template_file, wave, flux, sigma, out_file, throughput_file):
    '''
    Write a flux-calibrated spectrum (1D or 2D) to FITS, copying headers from
    template_file, replacing SCI with flux and adding/replacing ERR.
    '''
    def update_hdu(hdul, hdu):
        '''Replace an extension of the same name if present, else append.'''
        names = [h.name for h in hdul]
        if hdu.name in names:
            hdul[names.index(hdu.name)] = hdu
        else:
            hdul.append(hdu)

    with fits.open(template_file) as hdul:
        sci = hdul['SCI']
        sci.data = flux.astype(np.float32)
        sci.header.add_history('Telluric corrected and flux calibrated')
        sci.header.add_history(f'Flux calibration throughput: {os.path.basename(throughput_file)}')

        # --- ERR: sigma, carrying the SCI spectral WCS ---
        err_hdu = fits.ImageHDU(data=sigma.astype(np.float32), name='ERR')
        for k in ('CRVAL1', 'CDELT1', 'CRPIX1', 'CTYPE1', 'CUNIT1'):
            if k in sci.header:
                err_hdu.header[k] = sci.header[k]
        update_hdu(hdul, err_hdu)

        # --- WAVE: explicit wavelength array (float64), exact round-trip ---
        wave_hdu = fits.ImageHDU(data=np.asarray(wave, dtype=np.float64), name='WAVE')
        if 'CUNIT1' in sci.header:
            wave_hdu.header['BUNIT'] = (sci.header['CUNIT1'], 'wavelength unit')
        update_hdu(hdul, wave_hdu)

        hdul.writeto(out_file, overwrite=True)


def plot_throughput(throughput_wave, throughput_values, tell_windows=None, wave_range=None, title='', show=True):
    '''
    Plot the system throughput curve, optionally restricted to a data range.
    '''
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(throughput_wave, throughput_values, color='black', lw=1.2)
    if tell_windows is not None:
        for j, (lo, hi) in enumerate(tell_windows):
            ax.axvspan(lo, hi, color='grey', alpha=0.25,
                       label='telluric (downweighted)' if j == 0 else None)
    ax.set_xlabel(r'Wavelength [$\AA$]', fontsize=13)
    ax.set_ylabel('throughput', fontsize=13)
    ax.set_title(title, fontsize=13, pad=15)
    if wave_range is not None:
        ax.axvspan(*wave_range, color='grey', alpha=0.15, label='data coverage')
        ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show() if show else plt.close()


def plot_flux_calibration(wave, counts, counts_err, flux_cal, flux_cal_err, gpcnt_wave, gpcnt_arr, obj_dict, title='', savepath=plot_flux_cal_dir, smooth=10, show=True):
    '''
    Two-panel flux-calibration diagnostic for a single spectrum.
      top:    input counts/s (+ 1-sigma band, smoothed overlay)
      bottom: flux-calibrated F_lambda (+ 1-sigma band, smoothed overlay)

    wave (1D arr): wavelength [A]
    counts, counts_err (1D arr): input spectrum, counts/s
    flux_cal, flux_cal_err (1D arr): calibrated spectrum
    '''
    med_counts = median_filter(np.nan_to_num(counts).astype(np.float32), size=smooth, mode='reflect')
    med_flux = median_filter(np.nan_to_num(flux_cal).astype(np.float32), size=smooth, mode='reflect')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [3, 3, 1], 'hspace': 0})
    ax1.set_title(title, fontsize=15, pad=15)

    # top: counts/s
    ax1.fill_between(wave, counts - counts_err, counts + counts_err, step='mid', color='blue', alpha=0.1, linewidth=0.6, zorder=0)
    ax1.step(wave, counts, where='mid', color='blue', alpha=0.35, lw=0.6, zorder=1)
    ax1.step(wave, med_counts, where='mid', color='blue', lw=1.5, label='reduced, telluric-corrected', zorder=2)
    ax1.set_ylabel('counts / s', fontsize=13, labelpad=15)
    finite_c = counts[np.isfinite(counts)]
    if finite_c.size:
        ax1.set_ylim(np.percentile(finite_c, 0.1), np.percentile(finite_c, 99.9))
    ax1.legend(fontsize=11, loc='upper right')

    # middle: flux calibrated
    ax2.fill_between(wave, flux_cal - flux_cal_err, flux_cal + flux_cal_err, step='mid', color='brown', alpha=0.1, linewidth=0.6, zorder=0)
    ax2.step(wave, flux_cal, where='mid', color='brown', alpha=0.35, lw=0.6, zorder=1)
    ax2.step(wave, med_flux, where='mid', color='brown', lw=1.5, label='and flux calibrated', zorder=2)
    ax2.set_ylabel(r'$F_\lambda$ [erg/s/cm$^2$/$\AA$]', fontsize=13, labelpad=15)
    ax2.set_xlabel(r'Observed Wavelength [$\AA$]', fontsize=13, labelpad=15)
    finite_f = flux_cal[np.isfinite(flux_cal)]
    if finite_f.size:
        ax2.set_ylim(np.percentile(finite_f, 0.1), np.percentile(finite_f, 99.9))
    ax2.legend(fontsize=11, loc='upper right')

    # bottom: good pixel counts
    mean_gp = np.mean(gpcnt_arr)
    ax3.step(gpcnt_wave, gpcnt_arr, where='mid', color='grey', alpha=0.8, linewidth=0.6)
    ax3.fill_between(gpcnt_wave, 0, gpcnt_arr, step='mid', color='black', alpha=0.15)
    # ax3.axhline(mean_gp, color='black', lw=1, linestyle='dashed')
    # ax3.text(x=np.median(gpcnt_wave), y=mean_gp, s='mean # of good pixels',ha='center', va='bottom', color='black', fontsize=10)
    ax3.set_ylabel('# good pix', fontsize=8, labelpad=15)
    ax3.set_xlabel(r'Observed Wavelength [$\AA$]', fontsize=15, labelpad=15)
    ax3.set_xlim(np.min(gpcnt_wave), np.max(gpcnt_wave))

    plt.tight_layout()
    if savepath:
        save_dir = os.path.join(savepath, obj_dict['exposure_id'])
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.join(save_dir, 'telluric_corr_and_flux_cal.png')
        plt.savefig(filename, dpi=500, bbox_inches='tight')
    plt.show() if show else plt.close()