import argparse
import contextlib
import numpy as np
import os
import sys
from datetime import datetime

from ..scripts.get_NIRWALS_DRP_products import (get_reduced_exposures, get_reduced_spectra, plot_science_reduction_results, plot_ext_spectra_dir, load_fiber_map_dict)
from ..scripts.telluric_correction import (query_star_properties, fit_telluric_star_model, fit_telluric_poly_model, apply_telluric_model, plot_star_telluric_model_fit, plot_poly_telluric_model_fit, plot_telluric_correction)
from ..scripts.flux_calibration import (compute_throughput, save_throughput, flux_calibrate_exposure, write_fluxcal_fits, plot_throughput_fit, plot_throughput_validation, plot_flux_calibration, processed_data_dir)
from ..scripts.dither_combine import (combine_dithers, write_combined_fits, plot_fiber_map, plot_cube_image, plot_coadded_spectrum, clip_edges_mask)
from ..scripts.fiber_coadd import (select_bright_fibers, coadd_fibers, half_light_radius_rough_estimate, coadd_fibers_averaged)
from ..scripts.bitmask import (MASK_BADPIX, MASK_SKYLINE, MASK_LOWTELL, MASK_DONOTUSE, DONOTUSE_BITS, TELL_MIN, SKY_NSIG)

# ----------------------------------------------------------------------
# run logging (mirror stdout to a per-night log file)
# ----------------------------------------------------------------------
class _Tee:
    """Write to several streams at once, so prints reach the terminal and the log file."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


@contextlib.contextmanager
def tee_stdout_to_log(log_path):
    """
    Mirror everything printed to stdout into log_path (creating parent dirs)
    while still printing to the terminal; restore stdout on exit. Note: this
    captures stdout only -- uncaught tracebacks go to stderr and won't appear
    in the log.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logfile = open(log_path, 'w')
    original = sys.stdout
    sys.stdout = _Tee(original, logfile)
    try:
        yield
    finally:
        sys.stdout = original
        logfile.close()


# ----------------------------------------------------------------------
# raw extracted-spectra diagnostics
# ----------------------------------------------------------------------
def plot_reduced_1D_spectra(obs_date):
    """
    Save per-fiber diagnostic plots (raw 'a', continuum fit 'cf', reduced
    'ssc', plus good-pixel count) for every exposure on obs_date -- the
    notebook's plot_science_reduction_results() step, run at the start of
    the reduction.

    Skipped if outputs/plots/NIRWALS_DRP_reduced_spectra/<obs_date>/ already
    exists, so re-running the reduction doesn't overwrite existing plots.
    """
    date_dir = os.path.join(plot_ext_spectra_dir, obs_date)
    if os.path.isdir(date_dir):
        print(f'Reduced 1D spectra already plotted for {obs_date} -- skipping ({date_dir})')
        return
    print(f'Plotting raw extracted spectra for {obs_date} ...')
    plot_science_reduction_results(obs_date)


# ----------------------------------------------------------------------
# target / standard identification 
# ----------------------------------------------------------------------
def identify_target_and_standard(exposures, exclude=('SKY', 'ARC')):
    """
    Auto-identify which object(s) on a night are the science target vs. a
    standard star, purely from exposure counts: the object with the most 
    exposures is assumed to be the (possibly dithered) science target. 
    If a second object is present, it's assumed to be a standard.

    Returns (target_name, target_exps, standard_name (or None), standard_exps).
    """
    by_object = {}
    for e in exposures.values():

        object_name = e['object']
        exp_type = e['exp_type']

        # First exclude ARC exposures
        if object_name in exclude:
            continue

        # Exclude associated Sky frames (they use the same 'object' name as the target)
        if exp_type != 'Science':
            continue
        by_object.setdefault(object_name, []).append(e)

    if len(by_object) == 0:
        raise ValueError('No non-ARC Science exposures found on this night.')

    # only one science observation
    if len(by_object) == 1:
        name = next(iter(by_object))

        return name, by_object[name], None, []

    # science + standard star observation
    if len(by_object) == 2:
        (n1, e1), (n2, e2) = sorted(by_object.items(), key=lambda kv: -len(kv[1]))  # science target should have longest exposure time

        if len(e1) == len(e2):
            raise ValueError(
                f'Found two objects ({n1}, {n2}) with the same number of '
                'Science exposures on this night -- cannot auto-identify '
                'which is the science target from exposure counts alone.'
            )

        return n1, e1, n2, e2

    raise ValueError(
        f'Found {len(by_object)} Science objects on this night '
        f'({sorted(by_object)}) -- auto-identification only supports one or two.'
    )


def pick_matching_grating_angle(candidates, target_GA):
    """
    Restrict a list of exposures to those sharing the science target's grating angle.
    """
    return [e for e in candidates if round(e['grating_angle'], 1) == round(target_GA, 1)]


def coadd_bright_fibers_with_ivar(flux_all, sigma_all, frac=0.2):
    """
    Co-adding based on flux threshold.

    NOTE: this is the fiber-selection step used only to build the 1D input
    to the telluric-fitting routines below. Thus, it is separate from the 
    final science 1D output.
    """
    fibers = select_bright_fibers(flux_all, frac=frac)
    flux, sigma = coadd_fibers(flux_all, sigma_all, fibers)

    gpm = np.isfinite(flux) & np.isfinite(sigma) & (sigma > 0)
    ivar = np.zeros_like(flux)
    ivar[gpm] = 1.0 / sigma[gpm] ** 2
    gpm &= np.isfinite(ivar) & (ivar < 1e40)  # drop inf ivar

    return flux, sigma, ivar, gpm


# ----------------------------------------------------------------------
# telluric correction
# ----------------------------------------------------------------------
def fit_telluric_star(standard_exp, standard_name, standard_exposures, polyorder=8, disp=False, plot=True, obs_date=None):
    """
    Star-model telluric fit to a co-added telluric-standard star spectrum.
    """
    star_props = query_star_properties(standard_name)
    wave, flux_all, sigma_all = get_reduced_spectra(standard_exp, standard_exposures)
    flux, sigma, ivar, gpm = coadd_bright_fibers_with_ivar(flux_all, sigma_all)
    gpm = gpm & clip_edges_mask(wave)   # trim detector edges (same clip applied to the final product for target)

    fit = fit_telluric_star_model(
        flux, wave, ivar, gpm, star_props,
        standard_exp['airmass'], standard_exp['exptime'], polyorder=polyorder, disp=disp
        )
    
    if plot:
        plot_star_telluric_model_fit(wave, flux, fit, standard_exp, obs_date=obs_date, show=False)

    return fit


def fit_telluric_poly(target_exp, target_exposures, polyorder=3, disp=False, plot=True, obs_date=None):
    """
    Poly-model telluric fit, fit directly to a co-added spectrum (target or spec-phot standard).
    """
    wave, flux_all, sigma_all = get_reduced_spectra(target_exp, target_exposures)
    flux, sigma, ivar, gpm = coadd_bright_fibers_with_ivar(flux_all, sigma_all)
    gpm = gpm & clip_edges_mask(wave)   # trim detector edges (same clip applied to the final product for target)

    fit = fit_telluric_poly_model(
        flux, wave, ivar, gpm,
        target_exp['airmass'], target_exp['exptime'], polyorder=polyorder, disp=disp
    )
    if plot:
        plot_poly_telluric_model_fit(wave, flux, fit, target_exp, obs_date=obs_date, show=False)
    return fit


def resolve_telluric_correction(args, obs_date_exposures, target_name, sci_exposures):
    """
    If --telluric-standard-date is passed, Identify a telluric-standard observation 
    and fit the telluric transmission curve; otherwise fall back to fitting the 
    science target's own spectrum.
    """
    if not args.telluric_standard_date:
        print(f"\n\nTelluric correction: no --telluric-standard-date given -- fitting {target_name}'s "
              'own spectrum with PypeIt poly model', end='\n\n')
        return fit_telluric_poly(sci_exposures[0], obs_date_exposures, polyorder=args.target_polyorder, obs_date=args.obs_date)

    if args.telluric_standard_date == args.obs_date:
        tell_exposures = obs_date_exposures
        _, _, standard_name, standard_exps = identify_target_and_standard(obs_date_exposures)
        if standard_name is None:
            raise ValueError(
                f'--telluric-standard-date matches obs_date, but only one object was found on '
                f'{args.obs_date} -- expected a science target and a separate standard.'
            )
    else:
        tell_exposures = get_reduced_exposures(args.telluric_standard_date)
        standard_name, standard_exps, extra_name, _ = identify_target_and_standard(tell_exposures)
        if extra_name is not None:
            raise ValueError(
                f'Expected a single telluric-standard object on {args.telluric_standard_date}, '
                f'found two: {standard_name}, {extra_name}.'
            )

    matches = pick_matching_grating_angle(standard_exps, sci_exposures[0]['grating_angle'])
    if not matches:
        raise ValueError(
            f'No {standard_name} exposure on {args.telluric_standard_date} matches the science '
            f'grating angle {round(sci_exposures[0]["grating_angle"], 1)}.'
        )
    standard_exp = matches[0]
    print()
    print('----------')
    print('TELLURIC STANDARD')
    print()
    for i, match in enumerate(matches):

        print(f'matches[{i}] file:', match['a']['filename'])
        print(f'matches[{i}] EXPTYPE:', match['a']['exp_type'])
        print(f'matches[{i}] EXPTIME:', match['a']['exptime'])
        print()
    print()
    print('----------')
    print()

    print(f'\n\nTelluric correction: fitting standard {standard_name} ({args.telluric_standard_date}) with PypeIt star model', end='\n\n')
    return fit_telluric_star(standard_exp, standard_name, tell_exposures, polyorder=args.telluric_standard_polyorder, obs_date=args.obs_date)


# ----------------------------------------------------------------------
# flux calibration
# ----------------------------------------------------------------------
def get_throughput_file(sci_exposures, args, plot=True):
    """
    If --specphot-date is passed, derive the throughput curve. Otherwise, returns None.
    """
    if not args.specphot_date:
        print('\n\nFlux calibration: no --specphot-date given -- skipping flux calibration. '
              'Output will be in counts/s rather than absolute flux units.', end='\n\n')
        return None

    specphot_exposures = get_reduced_exposures(args.specphot_date)
    specphot_name, specphot_exps, extra_name, _ = identify_target_and_standard(specphot_exposures)
    if extra_name is not None:
        raise ValueError(
            f'Expected a single spec-phot standard object on {args.specphot_date}, '
            f'found two: {specphot_name}, {extra_name}.'
        )

    matches = pick_matching_grating_angle(specphot_exps, sci_exposures[0]['grating_angle'])
    if not matches:
        raise ValueError(
            f'No {specphot_name} exposure on {args.specphot_date} matches the science grating angle '
            f'{round(sci_exposures[0]["grating_angle"], 1)}.'
        )
    specphot_exp = matches[-1]
    print()
    print('----------')
    print('SPEC-PHOT')
    for i, match in enumerate(matches):

        print(f'matches[{i}] file:', match['a']['filename'])
        print(f'matches[{i}] EXPTYPE:', match['a']['exp_type'])
        print(f'matches[{i}] EXPTIME:', match['a']['exptime'])
        print()
    print()
    print('----------')
    print()

    print(f'\n\nFlux calibration: self-correcting spec-phot standard {specphot_name} ({args.specphot_date}) '
          'with PypeIt poly model, then computing throughput', end='\n\n')
    specphot_fit = fit_telluric_poly(specphot_exp, specphot_exposures, polyorder=args.specphot_polyorder, obs_date=args.obs_date)

    wave, _, sigma_all = get_reduced_spectra(specphot_exp, specphot_exposures)
    flux_all = np.asarray(specphot_exp['flux'], dtype=float)
    res = apply_telluric_model(flux_all, sigma_all, wave, specphot_fit['transmission'], specphot_fit['wave'])
    specphot_exp['flux'][:] = res['flux'].astype(specphot_exp['flux'].dtype)

    throughput_result = compute_throughput(spec_phot_obs=specphot_exp, spec_phot_name=specphot_name, tell_windows=None)

    if plot:
        plot_throughput_fit(throughput_result, specphot_exp, obs_date=args.obs_date, show=False)
        plot_throughput_validation(throughput_result, specphot_exp, obs_date=args.obs_date, show=False)

    return save_throughput(throughput_result)


# ----------------------------------------------------------------------
# per-exposure processing
# ----------------------------------------------------------------------
def process_single_exposure(sci_exp, exposures, telluric_fit, throughput_file=None, fibfil=0.62, plot=True, obs_date=None):
    """
    Apply telluric correction (and flux calibration, if throughput_file is
    given) to one science exposure's full per-fiber 2D spectrum.
    """
    wave, flux_all, sigma_all = get_reduced_spectra(sci_exp, exposures)
 
    tcorr = apply_telluric_model(flux_all, sigma_all, wave, telluric_fit['transmission'], telluric_fit['wave'])
 
    tcorr_1d = tcorr_sigma_1d = None
    if plot:
        raw_1d, _, _, _ = coadd_bright_fibers_with_ivar(flux_all, sigma_all)
        tcorr_1d, tcorr_sigma_1d, _, _ = coadd_bright_fibers_with_ivar(tcorr['flux'], tcorr['sigma'])
        plot_telluric_correction(wave, raw_1d, tcorr_1d, sci_exp, obs_date=obs_date, show=False)
 
    if throughput_file is not None:
        flux, sigma = flux_calibrate_exposure(wave, tcorr['flux'], tcorr['sigma'], sci_exp, throughput_file)
        if plot:
            fluxcal_1d, fluxcal_sigma_1d, _, _ = coadd_bright_fibers_with_ivar(flux, sigma)
            # number of good (unmasked) bright fibers per wavelength, from the reduced MASK
            bright = select_bright_fibers(flux, frac=0.2)
            m = sci_exp['ssc'].get('mask')
            if m is not None:
                m = np.asarray(m)
                if m.shape[0] != flux.shape[0]:
                    m = np.delete(m, load_fiber_map_dict()['sky_indices'], axis=0)
                good_fibers_1d = np.nansum(m[bright] == 0, axis=0)
            else:
                good_fibers_1d = np.zeros_like(wave)
            plot_flux_calibration(wave, tcorr_1d, tcorr_sigma_1d, fluxcal_1d, fluxcal_sigma_1d,
                                   wave, good_fibers_1d, sci_exp, obs_date=obs_date, show=False)
    else:
        flux, sigma = tcorr['flux'], tcorr['sigma']
 
    # --- extra per-exposure products for the MaNGA-style 2D FITS (read from the reduced extensions) ---
    fmap = load_fiber_map_dict(); sky_idx = fmap['sky_indices']
    ssc = sci_exp['ssc']
    n_obj = flux_all.shape[0]

    def _obj_fibers(arr):                       # drop sky fibers if the array still carries them
        if arr is None:
            return None
        arr = np.asarray(arr)
        return np.delete(arr, sky_idx, axis=0) if arr.shape[0] != n_obj else arr

    mask = _obj_fibers(ssc.get('mask'))                                           # (n_fib, n_wave), 1 = bad
    skycorr = _obj_fibers(ssc.get('skycorr'))                                     # (n_fib, n_wave) sky model, or None
    tellcorr = np.interp(wave, telluric_fit['wave'], telluric_fit['transmission'], left=np.nan, right=np.nan)  # 1D transmission
    specres = np.interp(wave, ssc['wave'], ssc['spec_res']) if ssc.get('spec_res') is not None else None
    specresd = np.interp(wave, ssc['wave'], ssc['specresd']) if ssc.get('specresd') is not None else None

    # --- quality bitmask (NIRWALS_DRP_PIXELMASK) ---
    bitmask = np.zeros(flux.shape, dtype=np.int32)
    if mask is not None:
        bitmask[np.asarray(mask) != 0] |= MASK_BADPIX
    if skycorr is not None:
        sky1d = np.nanmedian(skycorr, axis=0)                       # sky brightness per wavelength
        base = np.nanmedian(sky1d)
        scat = 1.4826 * np.nanmedian(np.abs(sky1d - base))          # robust scatter (MAD)
        sky_line = np.isfinite(sky1d) & (scat > 0) & (sky1d > base + SKY_NSIG * scat)
        bitmask[:, sky_line] |= MASK_SKYLINE
    low_tell = ~np.isfinite(tellcorr) | (tellcorr < TELL_MIN)
    bitmask[:, low_tell] |= MASK_LOWTELL
    bitmask[(bitmask & DONOTUSE_BITS) != 0] |= MASK_DONOTUSE

    obsinfo = {'exposure_id': sci_exp['exposure_id'], 'exptime': ssc.get('exptime'),
               'airmass': ssc.get('airmass'), 'gain': ssc.get('gain'), 'grating_angle': ssc.get('grating_angle')}

    return {'wave': wave, 'flux': flux, 'sigma': sigma,
            'exp': sci_exp,
            'mask': bitmask, 'skycorr': skycorr, 'tellcorr': tellcorr,
            'specres': specres, 'specresd': specresd, 'obsinfo': obsinfo}


def output_paths(ssc_file, obs_date, suffixes):
    """
    outputs/processed_data/<obs_date>/<original basename with 'ssc.fits'
    swapped for each of `suffixes`>, creating the directory if needed.
    """
    outdir = processed_data_dir / obs_date
    outdir.mkdir(parents=True, exist_ok=True)
    basename = os.path.basename(ssc_file)
    return [outdir / basename.replace('ssc.fits', suffix) for suffix in suffixes]


def write_combined_outputs(combined, obs_date, throughput_file, plot=True):
    """
    Write the combined data product (per-fiber spectra as the primary product,
    cube as a secondary) plus a quick-look 1D.
    """
    x, y = combined['x_arcsec'], combined['y_arcsec']
    (cx, cy), reff = half_light_radius_rough_estimate(combined['flux_all'], x, y)
    inside = (x - cx) ** 2 + (y - cy) ** 2 <= reff ** 2
    # flux-conserving co-add: sum in-region fibers per exposure, then ivar-average the exposures
    flux_1d, sigma_1d = coadd_fibers_averaged(combined['flux_all'], combined['sigma_all'], inside, combined['n_dither'])
    extract_info = {'method': 'fiber co-add', 'center': (cx, cy), 'radius': reff,
                    'n_fibers': int(inside.sum()), 'fibers_used': inside}

    print(f"\n{extract_info['method']}: centre=({cx:.1f}, {cy:.1f}) arcsec, R_eff={reff:.1f}\", {combined['n_dither']} exposures, {extract_info['n_fibers']} fibers")

    if plot:
        plot_fiber_map(combined, obs_date, show=False)                              # raw IFU map with colorbar representing target flux
        plot_fiber_map(combined, obs_date, show=False, aperture_info=extract_info)  # same as above but with 1D co-added fibers highlighted
        plot_cube_image(combined, obs_date, show=False)                             # plotting resampled exposure (combined dithers if any) onto cube grid (more noisy)
        plot_coadded_spectrum(combined, flux_1d, sigma_1d, obs_date, show=False,    # quick-look 1D spectrum which is a co-add of fibers indicated in IFU diagnostic plot
                              bunit=(r'erg/s/cm$^2$/$\AA$' if throughput_file else 'counts / s'), extract_info=extract_info)

    ssc_file = combined['exp']['ssc']['file']
    out_dir = processed_data_dir / obs_date
    out_dir.mkdir(parents=True, exist_ok=True)
    out_2d = out_dir / f'{obs_date}.reduced.postprocessed.fits'
    out_1d = out_dir / f'{obs_date}.reduced.postprocessed_1Dspec.fits'

    # edge-clip the 1D to the same range as the 2D file before writing
    keep = clip_edges_mask(combined['wave'])
    write_combined_fits(combined, str(out_2d), throughput_file=throughput_file)

    # setting ivar=0 for DONOTUSE in 1D spec
    wl_bits = (np.bitwise_or.reduce(combined['mask_all'].astype(np.int32), axis=0)
               if combined.get('mask_all') is not None
               else np.zeros(combined['wave'].shape, dtype=np.int32))
    
    donotuse = (wl_bits & MASK_DONOTUSE) != 0

    flux_out = flux_1d       # unmasked
    sigma_out = sigma_1d.copy()
    sigma_out[donotuse] = np.inf  # ivar=0 for DONOTUSE pixels

    write_fluxcal_fits(template_file=ssc_file, wave=combined['wave'][keep],
                       flux=flux_out[keep], sigma=sigma_out[keep],
                       out_file=str(out_1d), throughput_file=throughput_file)

    print(f'\n\nwrote {out_2d}')
    print(f'wrote {out_1d}', end='\n\n')


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('obs_date', help='YYYYMMDD of the science observation to postprocess')

    p.add_argument('--telluric-standard-date', default=None,
                    help='YYYYMMDD a telluric-standard star was observed on (same night as obs_date if equal to it). Omit to fit the science target\'s own spectrum instead.')
    p.add_argument('--telluric-standard-polyorder', type=int, default=8,
                    help='Continuum polynomial order for the star-model telluric fit.')
    p.add_argument('--target-polyorder', type=int, default=3,
                    help='Continuum polynomial order for the poly-model telluric fit, used when no telluric standard is given.')

    p.add_argument('--specphot-date', default=None,
                    help='YYYYMMDD of a spec-phot standard observation. If omitted, flux calibration and dither combination are both skipped.')
    p.add_argument('--specphot-polyorder', type=int, default=3,
                    help='Continuum polynomial order for the spec-phot standard\'s self telluric correction with poly-model.')

    p.add_argument('--no-dithers', action='store_true',
                    help='Use only the first exposure instead of combining all dithered exposures. A single-exposure cube is still built and written as the usual combined outputs (cube + 1D + plots).')

    return p.parse_args()


def main():
    args = parse_args()
    log_path = processed_data_dir / args.obs_date / f'postprocessing_{datetime.now():%Y%m%d_%H%M%S}.log'
    with tee_stdout_to_log(log_path):
        run(args)


def run(args):
    plot_reduced_1D_spectra(args.obs_date)

    exposures = get_reduced_exposures(args.obs_date)

    target_name, sci_exposures, extra_name, _ = identify_target_and_standard(exposures)
    if extra_name is not None and args.telluric_standard_date != args.obs_date:
        raise ValueError(
            f'Found two objects on {args.obs_date} ({target_name}, {extra_name}) but '
            f'--telluric-standard-date does not match obs_date -- pass '
            f'--telluric-standard-date {args.obs_date} if {extra_name} is a telluric standard '
            'observed the same night, otherwise check the data.'
        )

    n_dithers = len(sci_exposures)
    print(f'\n\nscience target: {target_name}\nfound {n_dithers} exposure(s) on {args.obs_date}'
          + (' (dithered)' if n_dithers > 1 else ''), end='\n\n')

    if args.no_dithers and n_dithers > 1:
        print(f'--no-dithers is set -- using the first of {n_dithers} exposures only '
              '(still cubed and written as combined outputs).')
        sci_exposures = sci_exposures[:1]

    telluric_fit = resolve_telluric_correction(args, exposures, target_name, sci_exposures)
    throughput_file = get_throughput_file(sci_exposures, args)

    per_exposure_results = [
        process_single_exposure(exp, exposures, telluric_fit, throughput_file=throughput_file, obs_date=args.obs_date)
        for exp in sci_exposures
    ]

    combined = combine_dithers(per_exposure_results)
    write_combined_outputs(combined, args.obs_date, throughput_file)


if __name__ == '__main__':
    main()
