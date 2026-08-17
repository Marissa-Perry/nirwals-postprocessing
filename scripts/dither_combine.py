"""
Originally written by Antoine Mahoro for combining dithered NIRWALS observations.

Combines one or more exposures into a single datacube. 
"""
import os
import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Circle
from astropy.visualization import ImageNormalize, ZScaleInterval

from . import cube_ifu
from .misc_functions import data_dir, plot_dir

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

# clipping edges of data
CLIP_WAVE_BLUE_RANGE = 50  # [A]
CLIP_WAVE_RED_RANGE = 50   # [A]


def clip_edges_mask(wave, blue_wav_clip=CLIP_WAVE_BLUE_RANGE, red_wav_clip=CLIP_WAVE_RED_RANGE):
    """
    Used to trim the dead instrument edges off the saved 1D and 2D spectra.
    """
    return (wave >= wave.min() + blue_wav_clip) & (wave <= wave.max() - red_wav_clip)


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
    if first.get('specresd') is not None:
        combined['specresd'] = first['specresd']
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


def write_combined_fits(combined, out_file, throughput_file=None):
    """
    Write the combined 2D fiber data product as a multi-extension FITS, MaNGA-style:

      0  PRIMARY   (empty; global header only)
      1  FLUX      stacked per-fiber spectra                 [NFIBER*NEXP, NWAVE]
      2  IVAR      inverse variance of FLUX                  [NFIBER*NEXP, NWAVE]
      3  MASK      bad-pixel mask (1 = bad, gpcnt == 0)      [NFIBER*NEXP, NWAVE]
      4  WAVE      wavelength vector                         [NWAVE]
      5  SPECRES   median spectral resolution R vs wave      [NWAVE]
      6  SPECRESD  1-sigma scatter of R across fibers        [NWAVE]
      7  OBSINFO   binary table, one row per exposure combined
      8  XPOS      fiber X positions (arcsec, IFU centre)    [NFIBER*NEXP]
      9  YPOS      fiber Y positions (arcsec, IFU centre)    [NFIBER*NEXP]
     10  SKYCORR   sky-correction term (cs - ss)             [NFIBER*NEXP, NWAVE]
     11  TELLCORR  telluric transmission applied             [NWAVE]
     12  FLUXCAL   flux-calibration throughput curve         [NWAVE]
     13  CUBE      Shepard-reconstructed datacube            [NWAVE, NY, NX]
     14  CUBE_IVAR inverse variance of CUBE                  [NWAVE, NY, NX]

    Each wavelength-axis array is edge-clipped to the useful range. Extensions
    whose data isn't present in `combined` are skipped (relative order preserved).
    """
    ref_header = combined['exp']['ssc']['sci_header']
    bunit = 'erg/s/cm2/A' if throughput_file else 'counts/s'

    keep = clip_edges_mask(combined['wave'])
    if not keep.any():
        raise ValueError(f'edge clip [{CLIP_WAVE_BLUE_RANGE}, {CLIP_WAVE_RED_RANGE}] A keeps no wavelengths')
    i0 = int(np.argmax(keep))
    wave = combined['wave'][keep]

    def stamp(hdu, desc, unit=None):
        if unit is not None:
            hdu.header['BUNIT'] = unit
        hdu.header['EXTDESC'] = (desc, 'extension contents')
        return hdu

    # --- 0 PRIMARY: empty, global header only ---
    pri = fits.PrimaryHDU()
    pri.header['OBJECT'] = (ref_header.get('OBJECT', combined['exp'].get('object', '')), 'science target')
    pri.header['NDITHER'] = (combined.get('n_dither', 1), 'number of combined dithered exposures')
    pri.header['NWAVE'] = (int(keep.sum()), 'number of wavelength pixels')
    pri.header['NFIBER'] = (int(combined['flux_all'].shape[0]), 'total fiber rows (NFIBER_PER_EXP * NEXP)')
    pri.header.add_history(f'Edge-clipped: {CLIP_WAVE_BLUE_RANGE:.0f} A blue, {CLIP_WAVE_RED_RANGE:.0f} A red')
    pri.header.add_history('Primary product: per-fiber spectra (telluric-corrected'
                           + (', flux-calibrated)' if throughput_file else ')'))
    hdus = [pri]

    # --- 1 FLUX (carries the spectral WCS on the wavelength axis) ---
    flux_hdu = fits.ImageHDU(combined['flux_all'][:, keep].astype(np.float32), name='FLUX')
    for k in ('CTYPE1', 'CRVAL1', 'CRPIX1', 'CDELT1', 'CUNIT1'):
        if k in ref_header:
            flux_hdu.header[k] = ref_header[k]
    if 'CRPIX1' in flux_hdu.header:
        flux_hdu.header['CRPIX1'] = flux_hdu.header['CRPIX1'] - i0
    hdus.append(stamp(flux_hdu, 'row-stacked fiber spectra in units of ergs/s/cm^2/A^-1', bunit))

    # --- 2 IVAR ---
    sigma = combined['sigma_all'][:, keep]
    fiber_ivar = np.zeros_like(sigma)
    gp = np.isfinite(sigma) & (sigma > 0)
    fiber_ivar[gp] = 1.0 / sigma[gp] ** 2
    hdus.append(stamp(fits.ImageHDU(fiber_ivar.astype(np.float32), name='IVAR'), 'inverse variance of FLUX', f'({bunit})^-2'))

    # --- 3 MASK (1 = bad; from gpcnt == 0) ---
    if combined.get('mask_all') is not None:
        hdus.append(stamp(fits.ImageHDU(combined['mask_all'][:, keep].astype(np.int16), name='MASK'), 'bad-pixel mask (1 = bad)'))

    # --- 4 WAVE ---
    wave_hdu = stamp(fits.ImageHDU(wave.astype(np.float64), name='WAVE'), 'wavelength vector in units of A', ref_header.get('CUNIT1', 'Angstrom'))
    hdus.append(wave_hdu)

    # --- 5 SPECRES / 6 SPECRESD ---
    if combined.get('specres') is not None:
        hdus.append(stamp(fits.ImageHDU(np.asarray(combined['specres'])[keep].astype(np.float32), name='SPECRES'), 'median spectral resolution (R)'))
    if combined.get('specresd') is not None:
        hdus.append(stamp(fits.ImageHDU(np.asarray(combined['specresd'])[keep].astype(np.float32), name='SPECRESD'), 'standard deviation (1-sigma) of SPECRES'))

    # --- 7 OBSINFO ---
    if combined.get('obsinfo'):
        rows = combined['obsinfo']; keys = list(rows[0].keys())
        def col(k):
            vals = [r[k] for r in rows]
            if all(isinstance(v, str) for v in vals):
                return fits.Column(name=k.upper(), format=f'{max(len(v) for v in vals)}A', array=np.array(vals))
            return fits.Column(name=k.upper(), format='E', array=np.array(vals, dtype=float))
        obs_hdu = fits.BinTableHDU.from_columns([col(k) for k in keys], name='OBSINFO')
        obs_hdu.header['EXTDESC'] = ('per-exposure metadata (one row per exposure)', 'extension contents')
        hdus.append(obs_hdu)

    # --- 8 XPOS / 9 YPOS (one value per fiber row) ---
    hdus.append(stamp(fits.ImageHDU(combined['x_arcsec'].astype(np.float32), name='XPOS'), 'fiber X-position relative to IFU center', 'arcsec'))
    hdus.append(stamp(fits.ImageHDU(combined['y_arcsec'].astype(np.float32), name='YPOS'), 'fiber Y-position relative to IFU center', 'arcsec'))

    # --- 10 SKYCORR ---
    if combined.get('skycorr') is not None:
        hdus.append(stamp(fits.ImageHDU(combined['skycorr'][:, keep].astype(np.float32), name='SKYCORR'), 'subtracted sky emission', bunit))

    # --- 11 TELLCORR ---
    if combined.get('tellcorr') is not None:
        hdus.append(stamp(fits.ImageHDU(np.asarray(combined['tellcorr'])[keep].astype(np.float32), name='TELLCORR'), 'telluric transmission correction'))

    # --- 12 FLUXCAL (throughput curve, interpolated onto WAVE) ---
    if throughput_file is not None:
        try:
            t = pd.read_csv(throughput_file)
            tw, tt = t.iloc[:, 0].values, t.iloc[:, 1].values
            thru = np.interp(wave, tw, tt, left=np.nan, right=np.nan).astype(np.float32)
            hdus.append(stamp(fits.ImageHDU(thru, name='FLUXCAL'), 'flux-calibration throughput curve'))
        except Exception as e:
            print(f'  FLUXCAL: could not read throughput {throughput_file} ({e})')

    # --- 13 CUBE / 14 CUBE_IVAR ---
    cube_hdu = fits.ImageHDU(combined['cube'][keep].astype(np.float32), name='CUBE')
    cube_hdu.header['EXTDESC'] = ('3d data cube (all exposures combined)', 'extension contents')
    cube_hdu.header['BUNIT'] = bunit
    hdus.append(cube_hdu)
    hdus.append(stamp(fits.ImageHDU(combined['ivar_cube'][keep].astype(np.float32), name='CUBE_IVAR'), 'inverse variance of CUBE', f'({bunit})^-2'))

    fits.HDUList(hdus).writeto(out_file, overwrite=True)


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
    keep = clip_edges_mask(wave)                   # edges outside this are clipped from the FITS
    flux_keep = np.where(keep, flux_1d, np.nan)    # kept region only (clipped -> NaN so it's not drawn black)

    fig, ax = plt.subplots(figsize=(8, 3))
    if extract_info is not None:
        ax.set_title(f'{extract_info.get("method", "co-add")}: reff = {extract_info["radius"]:.1f}", {n_dither} exposures', fontsize=12, pad=15)
        # ax.set_title(f'fiber co-add: reff = {extract_info["radius"]:.1f}", {n_dither} exposures', fontsize=12, pad=15)
    # shade + draw the clipped edges in light red (excluded from the FITS); kept region over-plotted in black
    ax.axvspan(wave.min(), wave.min() + CLIP_WAVE_BLUE_RANGE, color='lightcoral', alpha=0.12, zorder=0)
    ax.axvspan(wave.max() - CLIP_WAVE_RED_RANGE, wave.max(), color='lightcoral', alpha=0.12, zorder=0)
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
