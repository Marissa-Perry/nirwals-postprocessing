"""
Originally written by Antoine Mahoro for combining dithered NIRWALS observations.

Combines one or more exposures into a single datacube. 
"""
import os
import warnings
import numpy as np
import pandas as pd

from ..scripts.bitmask import MASK_DONOTUSE
from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Circle
from astropy.visualization import ImageNormalize, ZScaleInterval
from scipy.ndimage import median_filter

from . import cube_ifu
from .get_NIRWALS_DRP_products import data_dir, plot_dir, PIPELINE_BLUE_CLIP_PX, PIPELINE_FIBER_FRAC

plot_dither_dir = plot_dir / 'combined_dithers'
os.makedirs(plot_dither_dir, exist_ok=True)

# ---- default cube-reconstruction parameters (from Version_3/parameters_file.txt) ----
ASTROMETRY_FILE = data_dir / 'IFU' / 'NIRWALS_obj_bundle_IFU_astrom.csv'
DITHER_OFFSETS_FILE = data_dir / 'IFU' / 'dither_pattern.csv'  # edit this file directly for a different dither pattern

INPUT_GRID_DIMS = (25, 35)     # on-sky extent (arcsec) the astrometry positions are defined against
OUTPUT_GRID_DIMS = (25, 35)    # output cube spatial extent (arcsec), before dividing by OUT_ARCSEC_PER_PIXEL
OUT_ARCSEC_PER_PIXEL = 1.0
DIAM_CORES_ARCSEC = 1.326
GAUSSIAN_SIGMA_PIXELS = 1.1 / OUT_ARCSEC_PER_PIXEL
RLIM_PIXELS = 1.5 / OUT_ARCSEC_PER_PIXEL

# edge flux-deviation check: how far past each edge to look for a stable interior plateau,
# the max fractional deviation from it allowed, how many consecutive columns must satisfy
# that before the scan stops, and the median-filter size so single noisy columns can't
# interrupt the scan or bias the plateau
EDGE_INTERIOR_MARGIN = 350
EDGE_DEV_FRAC = 0.20
EDGE_SUSTAIN_RUN = 30
EDGE_SMOOTH_SIZE = 9


def _edge_flux_trim(flux_smooth, start, direction, plateau, dev_frac=EDGE_DEV_FRAC, sustain=EDGE_SUSTAIN_RUN):
    """
    From column `start`, scanning inward by `direction` (+1 from the blue edge, -1 from
    the red edge), find the first run of `sustain` consecutive columns all within
    dev_frac of `plateau`, and return how many columns were trimmed to reach the start of
    that run. A single lucky column can't stop the scan early -- real edge artifacts seen
    in practice decay gradually over 100+ columns, not as a sharp step, so requiring a
    sustained run (rather than just the first in-band column) matters here. If no such run
    is ever found, trims everything scanned.
    """
    n = flux_smooth.size
    idx = start
    run_start = start
    run_len = 0
    while 0 <= idx < n:
        dev = abs(flux_smooth[idx] / plateau - 1.0)
        if dev < dev_frac:
            if run_len == 0:
                run_start = idx
            run_len += 1
            if run_len >= sustain:
                return abs(run_start - start)
        else:
            run_len = 0
        idx += direction
    return abs(idx - start)


def clip_edges_mask(wave, flux, ivar, blue_clip_px=PIPELINE_BLUE_CLIP_PX, fiber_frac=PIPELINE_FIBER_FRAC):
    """
    Used to trim the dead instrument edges off the saved 1D and 2D spectra.

    Blue: fixed blue_clip_px columns, matching the common-grid padding every exposure gets
    in nirwalsreduce.py -- no fibre has real data there, so this is exact and the same for
    every grating setting, unlike a fixed wavelength budget (confirmed: good-pixel
    fraction is identically zero for exactly this many columns in every exposure checked,
    across 4 different grating angles spanning Y/J/H).

    Red: the last column where at least fiber_frac of fibres have both ivar > 0 and a
    nonzero flux (ivar alone isn't enough -- it can stay positive past a fibre's native
    coverage even though its flux is identically zero there).

    Past each of those, both edges can still carry a real flux bias from telluric opacity
    rising toward the edge of each band's atmospheric window (confirmed against the
    fitted telluric transmission model: dense, deep absorption sets in right where the
    flux deviates) -- not a sharp spike, but a gradual decay over 100-300+ columns, so
    neither edge's own nearby data makes a reliable baseline to compare against on its
    own (checked and rejected: comparing to a small nearby window is unstable, since nearby
    columns can themselves still be somewhat biased). Both edges are instead trimmed
    forward to the first sustained run of EDGE_SUSTAIN_RUN columns within EDGE_DEV_FRAC of
    one shared plateau -- the median flux over the deep interior, EDGE_INTERIOR_MARGIN
    columns past each of the above edges, which is far enough in to be stable and
    unaffected by either edge's own bias.
    """
    flux2d = np.atleast_2d(flux)
    ivar2d = np.atleast_2d(ivar)
    coverage = np.mean((ivar2d > 0) & (flux2d != 0), axis=0)
    has_data = coverage >= fiber_frac

    n_red_coverage = 0
    for v in has_data[::-1]:
        if not v:
            n_red_coverage += 1
        else:
            break

    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)  # all-NaN columns are expected past the coverage edge
        flux_1d = np.nanmedian(np.where(flux2d != 0, flux2d, np.nan), axis=0)
    flux_smooth = median_filter(np.nan_to_num(flux_1d), size=EDGE_SMOOTH_SIZE, mode='nearest')

    n = flux_smooth.size
    interior_lo = min(blue_clip_px + EDGE_INTERIOR_MARGIN, n)
    interior_hi = max(n - n_red_coverage - EDGE_INTERIOR_MARGIN, interior_lo)
    interior = flux_smooth[interior_lo:interior_hi]
    plateau = np.median(interior) if interior.size else np.median(flux_smooth)

    n_blue_flux = _edge_flux_trim(flux_smooth, blue_clip_px, +1, plateau)
    n_red_flux = _edge_flux_trim(flux_smooth, n - 1, -1, plateau)

    n_blue = blue_clip_px + n_blue_flux
    n_red = max(n_red_coverage, n_red_flux)

    keep = np.ones(wave.shape, dtype=bool)
    keep[:n_blue] = False
    if n_red > 0:
        keep[-n_red:] = False

    return keep


def load_astrometry(astrometry_file=ASTROMETRY_FILE):
    """
    Base on-sky (X_arcsec, Y_arcsec) fiber positions, one row per object fiber.
    """
    return pd.read_csv(astrometry_file)


def load_dither_offsets(offsets_file=DITHER_OFFSETS_FILE):
    """
    Dither offsets + relative flux scale, keyed by exposure ID
    ('{YYYYMMDD}_{n}'). Returns an empty table if the file doesn't exist
    (e.g. no dithering has been configured yet) -- every exposure then
    falls back to zero offset / unit flux scale.
    """
    offsets_file = Path(offsets_file)
    if not offsets_file.exists():
        return pd.DataFrame(columns=['X_offset', 'Y_offset', 'flux_scale'])
    return pd.read_csv(offsets_file).set_index('ID')


def exposure_offset_id(result):
    """
    '{YYYYMMDD}_{n}' from a process_single_exposure() result, matching User_offset.csv's ID convention.
    """
    exp_id = result['exp']['exposure_id']   # e.g. 'dphN202407010002.2.1'
    obs_date = exp_id[4:12]                 # 'YYYYMMDD' following the 'dphN' prefix
    n = exp_id.split('.')[1]                # '2'
    return f'{obs_date}_{n}'


def combine_dithers(per_exposure_results, astrometry_file=ASTROMETRY_FILE, offsets_file=DITHER_OFFSETS_FILE,
                     input_grid_dims=INPUT_GRID_DIMS, output_grid_dims=OUTPUT_GRID_DIMS,
                     diam_cores_arcsec=DIAM_CORES_ARCSEC, gaussian_sigma_pixels=GAUSSIAN_SIGMA_PIXELS,
                     rlim_pixels=RLIM_PIXELS, fibfil=0.62):
    """
    Build one combined datacube from one or more per-exposure post-processing
    results (postprocessing.process_single_exposure), applying each
    exposure's dither offset and flux scale first.

    Returns {'wave', 'cube', 'ivar_cube', 'exp'}: cube/ivar_cube have shape
    (n_wave, size_y, size_x); 'exp' is the first exposure's metadata dict
    (used for header/filename purposes downstream).
    """
    astrometry = load_astrometry(astrometry_file)
    offsets = load_dither_offsets(offsets_file)

    wave = per_exposure_results[0]['wave']
    for r in per_exposure_results[1:]:
        if not np.array_equal(r['wave'], wave):
            raise ValueError('Exposures have different wavelength grids -- resample onto a common grid before combining.')

    all_flux, all_sigma, all_x, all_y, all_expidx = [], [], [], [], []
    all_mask, all_skycorr, exposure_ids, obsinfo = [], [], [], []
    for e, result in enumerate(per_exposure_results):
        oid = exposure_offset_id(result)
        if oid in offsets.index:
            x_off, y_off, flux_scale = offsets.loc[oid, ['X_offset', 'Y_offset', 'flux_scale']]
        else:
            print(f'  {oid}: not found in {os.path.basename(str(offsets_file))} -- using zero offset, unit flux scale')
            x_off, y_off, flux_scale = 0.0, 0.0, 1.0

        n_fibers = result['flux'].shape[0]
        if len(astrometry) != n_fibers:
            raise ValueError(
                f'Astrometry file has {len(astrometry)} fiber rows but this exposure has {n_fibers} '
                'object fibers -- check the astrometry-file/fiber-ordering assumption.'
            )

        all_flux.append(result['flux'] * flux_scale)
        all_sigma.append(result['sigma'] * flux_scale)
        all_x.append(astrometry['X_arcsec'].values + x_off)
        all_y.append(astrometry['Y_arcsec'].values + y_off)
        all_expidx.append(np.full(n_fibers, e, dtype=np.int16))   # which exposure each fiber row is from
        exposure_ids.append(result['exp']['exposure_id'])
        all_mask.append(result['mask'])                           # gpcnt bad-pixel mask (never flux-scaled)
        if result.get('skycorr') is not None:
            all_skycorr.append(result['skycorr'])                 # sky-correction term (cs - ss), unscaled
        obsinfo.append(result['obsinfo'])

    flux_all = np.concatenate(all_flux, axis=0)
    sigma_all = np.concatenate(all_sigma, axis=0)
    x_arcsec = np.concatenate(all_x)
    y_arcsec = np.concatenate(all_y)
    exposure_idx = np.concatenate(all_expidx)
    mask_all = np.concatenate(all_mask, axis=0)
    skycorr = np.concatenate(all_skycorr, axis=0) if len(all_skycorr) == len(per_exposure_results) else None

    core_x_pixels, core_y_pixels, core_diam_pixels = cube_ifu.change_coords(
        x_arcsec + input_grid_dims[0] / 2., y_arcsec + input_grid_dims[1] / 2., diam_cores_arcsec,
        input_grid_dims=input_grid_dims, output_grid_dims=output_grid_dims,
    )

    gpm = np.isfinite(sigma_all) & (sigma_all > 0)
    ivar_all = np.zeros_like(sigma_all)
    ivar_all[gpm] = 1.0 / sigma_all[gpm] ** 2
    # zero IVAR wherever DONOTUSE is set
    ivar_all[(mask_all & MASK_DONOTUSE) != 0] = 0.0 

    (cube, ivar_cube), _ = cube_ifu.ifu_to_grid(
        flux_all, core_x_pixels, core_y_pixels, core_diam_pixels,
        grid_dimensions_pixels=output_grid_dims, use_gaussian_weights=True,
        gaussian_sigma_pixels=gaussian_sigma_pixels, rlim_pixels=rlim_pixels,
        use_broadcasting=False, ivar_data=ivar_all,
    )

    # scale cube down by the fiber fill factor
    cube = cube * fibfil
    ivar_cube = ivar_cube / fibfil ** 2

    first = per_exposure_results[0]
    combined = {'wave': wave, 'cube': cube, 'ivar_cube': ivar_cube,
                'flux_all': flux_all, 'sigma_all': sigma_all,
                'mask_all': mask_all,
                'x_arcsec': x_arcsec, 'y_arcsec': y_arcsec,
                'exposure_idx': exposure_idx, 'exposure_ids': exposure_ids, 'obsinfo': obsinfo,
                'n_dither': len(per_exposure_results),   # number of exposures (dither pointings) combined
                'exp': first['exp']}
    # single-curve products (per night, same across exposures) -- add only if present
    if skycorr is not None:
        combined['skycorr'] = skycorr
    if first.get('tellcorr') is not None:
        combined['tellcorr'] = first['tellcorr']
    if first.get('specres') is not None:
        combined['specres'] = first['specres']
    return combined


def cube_region_mask(cube_shape, center, reff):
    """
    (px, py, r_pix, mask) for a circular region (centre + radius in arcsec) on
    the cube grid, using the exact arcsec -> pixel transform combine_dithers
    used (change_coords: shift by half the FOV, scale by output/input dims).
    Shared by coadd_cube_region and its diagnostic so the plot shows exactly
    what is summed.
    """
    size_y, size_x = cube_shape[1], cube_shape[2]
    cx, cy = center
    scale = OUTPUT_GRID_DIMS[0] / INPUT_GRID_DIMS[0]
    px = (cx + INPUT_GRID_DIMS[0] / 2.) * scale
    py = (cy + INPUT_GRID_DIMS[1] / 2.) * scale
    r_pix = reff * scale
    yy, xx = np.mgrid[0:size_y, 0:size_x]
    mask = (xx - px) ** 2 + (yy - py) ** 2 <= r_pix ** 2
    return px, py, r_pix, mask


def coadd_cube_region(combined, center, reff):
    """
    Sum the cube spaxels within a circular region. 

    Note: since no covariance correction is applied, spectra from the cube have correlated errors and therefore, the errors should be considered as lower bounds.

    Returns (flux_1d, sigma_1d, n_spaxels).
    """
    cube, ivar_cube = combined['cube'], combined['ivar_cube']
    _, _, _, mask = cube_region_mask(cube.shape, center, reff)

    flux = np.nansum(cube[:, mask], axis=1)
    var = np.zeros_like(ivar_cube)
    good = ivar_cube > 0
    var[good] = 1.0 / ivar_cube[good]
    sigma = np.sqrt(np.nansum(var[:, mask], axis=1))
    return flux, sigma, int(mask.sum())


def plot_fiber_map(combined, obs_date, savepath=plot_dither_dir, show=True, aperture_info=None):
    """
    IFU fiber map: each fiber at its on-sky (X, Y) coloured by its wavelength-summed flux.
    One panel per exposure (dither) so overlapping pointings stay legible.
    """
    x, y = combined['x_arcsec'], combined['y_arcsec']
    f = np.nansum(combined['flux_all'], axis=1)
    vmin, vmax = np.percentile(f[np.isfinite(f)], [2, 98])   # shared colour scale across all panels

    n_dither = combined.get('n_dither', 1)
    n_fibers = f.shape[0] // n_dither   # exposures are concatenated in equal blocks of n_fibers

    fig, axes = plt.subplots(1, n_dither, figsize=(5 * n_dither, 6), sharex=True, sharey=True, squeeze=False)

    for i, ax in enumerate(axes[0]):
        sl = slice(i * n_fibers, (i + 1) * n_fibers)   # this exposure's block of fibers
        # outline each fiber so it's clearly resolved
        sc = ax.scatter(x[sl], y[sl], c=f[sl], s=100, vmin=vmin, vmax=vmax, cmap='viridis', edgecolors='black', linewidths=0.4)

        if aperture_info is not None and aperture_info.get('fibers_used') is not None:
            used = aperture_info['fibers_used'][sl]   # just this exposure's fibers inside the shared region
            ax.scatter(x[sl][used], y[sl][used], s=180, facecolors='none', edgecolors='red', linewidths=1.2)
            if aperture_info.get('center') is not None and aperture_info.get('radius') is not None:
                ax.add_patch(Circle(aperture_info['center'], aperture_info['radius'], fill=False, edgecolor='red', linestyle='--', linewidth=1.5))
                # text position at the top of the half-light radius with 1" buffer
                ax.text(aperture_info['center'][0], aperture_info['center'][1] + aperture_info['radius'] + 1, 
                        s=f'reff={aperture_info['radius']:.1f}" {int(used.sum())} fibers', ha='center', 
                        weight='bold', color='red', path_effects=[path_effects.withStroke(linewidth=5.8, foreground='white')], fontsize=12)

        ax.set_title(f'exposure {i + 1}', fontsize=13, pad=15)
        ax.set_xlabel('X [arcsec]', fontsize=13, labelpad=10)
        ax.set_aspect('equal')
    axes[0][0].set_ylabel('Y [arcsec]', fontsize=13, labelpad=10)

    dither_txt = f'combined dithers -- ' if n_dither > 1 else ''
    fig.suptitle(f'{dither_txt}{obs_date}', fontsize=15)
    cb = fig.colorbar(sc, ax=list(axes.ravel()), location='right', pad=0.02)
    cb.set_label('summed flux', fontsize=12, labelpad=12)

    if savepath:
        save_dir = os.path.join(str(savepath), obs_date)
        os.makedirs(save_dir, exist_ok=True)
        name = 'fiber_map_coadd.png' if aperture_info is not None else 'fiber_map.png'
        fig.savefig(os.path.join(save_dir, name), dpi=300, bbox_inches='tight')
    plt.show() if show else plt.close()


def plot_cube_image(combined, obs_date, savepath=plot_dither_dir, show=True): #, region=None):
    """
    wavelength-collapsed image of the combined cube. 
    """
    image = np.nansum(combined['cube'], axis=0)
    norm = ImageNormalize(image, interval=ZScaleInterval())
 
    fig, ax = plt.subplots()
    img = ax.imshow(image, origin='lower', cmap='inferno', norm=norm)
    n_dither = combined.get('n_dither', 1)
    dither_txt = f'{n_dither} combined dithers -- ' if n_dither > 1 else ''
    ax.set_title(f'{dither_txt}{obs_date}', fontsize=14, pad=18)
    ax.set_xlabel('x-position [pixel]', fontsize=13, labelpad=15)
    ax.set_ylabel('y-position [pixel]', fontsize=13, labelpad=15)
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('summed flux', size=12, labelpad=15)

    # if region is not None:
    #     center, reff = region
    #     px, py, r_pix, mask = cube_region_mask(combined['cube'].shape, center, reff)
    #     ax.add_patch(Circle((px, py), r_pix, fill=False, edgecolor='cyan', linestyle='--', linewidth=1.5))  # the aperture
    #     ax.contour(mask.astype(float), levels=[0.5], colors='cyan', linewidths=0.8)                         # the spaxels actually summed
    #     ax.plot(px, py, '+', color='cyan', markersize=10)                                                   # region centre
    #     ax.text(0.03, 0.97, f'reff = {reff:.1f}", {int(mask.sum())} spaxels summed', transform=ax.transAxes,
    #             va='top', ha='left', color='cyan', fontsize=10,
    #             path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])
 
    if savepath:
        save_dir = os.path.join(str(savepath), obs_date)
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, 'cube_image.png'), dpi=500, bbox_inches='tight')
    plt.show() if show else plt.close()
 
 
def plot_coadded_spectrum(combined, flux_1d, sigma_1d, obs_date, savepath=plot_dither_dir, show=True,
                          bunit=r'erg/s/cm$^2$/$\AA$', extract_info=None):
    """
    1D co-added fiber spectrum.
    """
    n_dither = combined.get('n_dither', 1)
    wave = combined['wave']
    with np.errstate(divide='ignore'):
        ivar_1d = np.where(np.isfinite(sigma_1d) & (sigma_1d > 0), 1.0 / sigma_1d**2, 0.0)
    keep = clip_edges_mask(wave, flux_1d, ivar_1d)  # edges outside this are clipped from the FITS
    flux_keep = np.where(keep, flux_1d, np.nan)    # kept region only (clipped -> NaN so it's not drawn black)

    fig, ax = plt.subplots(figsize=(8, 3))
    if extract_info is not None:
        ax.set_title(f'{extract_info.get("method", "co-add")}: reff = {extract_info["radius"]:.1f}", {n_dither} exposures', fontsize=12, pad=15)
        # ax.set_title(f'fiber co-add: reff = {extract_info["radius"]:.1f}", {n_dither} exposures', fontsize=12, pad=15)
    # shade + draw the clipped edges in light red (excluded from the FITS); kept region over-plotted in black
    kept_wave = wave[keep]
    ax.axvspan(wave.min(), kept_wave.min() if kept_wave.size else wave.max(), color='lightcoral', alpha=0.12, zorder=0)
    ax.axvspan(kept_wave.max() if kept_wave.size else wave.min(), wave.max(), color='lightcoral', alpha=0.12, zorder=0)
    ax.step(wave, flux_1d, where='mid', color='lightcoral', linewidth=0.5, zorder=1)
    ax.step(wave, flux_keep, where='mid', color='black', linewidth=0.5, zorder=2)
    ax.fill_between(wave, flux_keep-sigma_1d, flux_keep+sigma_1d, step='mid', color='grey', alpha=0.6, zorder=0)
    ax.set_xlabel(r'Observed Wavelength [$\AA$]', fontsize=13, labelpad=15)
    ax.set_ylabel(fr'$f_\lambda$ [{bunit}]', fontsize=13, labelpad=15)

    finite = flux_1d[keep & np.isfinite(flux_1d)]   # scale y to the kept science range, not the dead edges
    if finite.size:
        lo, hi = np.percentile(finite, [1, 99])
        pad = 0.5 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)

    if savepath:
        save_dir = os.path.join(str(savepath), obs_date)
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, 'coadded_spectrum.png'), dpi=500, bbox_inches='tight')
    plt.show() if show else plt.close()
