import os
import numpy as np
import matplotlib.pyplot as plt
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.centroids import centroid_1dg

from .dither_combine import load_astrometry


def select_fibers_by_completeness(flux_all, completeness=0.98):
    '''
    Fibers added brightest-first until completeness * total flux is captured.
    '''
    mean_flux = np.nanmean(flux_all, axis=1)
    order = np.argsort(mean_flux)[::-1]
    cumf = np.cumsum(mean_flux[order])
    n = np.searchsorted(cumf, completeness * cumf[-1]) + 1
    return order[:n]


def select_fibers_by_peak_snr(flux_all, sigma_all):
    '''
    Fibers added brightest-first up to the fiber count that maximizes cumulative SNR.
    '''
    order = np.argsort(np.nanmean(flux_all, axis=1))[::-1]
    f = np.nansum(flux_all[order], axis=1)
    s = np.sqrt(np.nansum(sigma_all[order] ** 2, axis=1))
    cumf = np.cumsum(f)
    cum_s = np.sqrt(np.cumsum(s ** 2))
    with np.errstate(invalid='ignore', divide='ignore'):
        cum_snr = cumf / cum_s
    n = np.nanargmax(cum_snr) + 1
    return order[:n]


def plot_cumulative_flux_coadd(flux_all, sigma_all, obj_dict, savepath, filename, n_fib_coadd=None, obs_date=None, show=True):
    '''
    Top: cumulative flux, normalized to the total.
    Bottom: cumulative SNR.
    '''
    order = np.argsort(np.nanmean(flux_all, axis=1))[::-1]
    n_fibers = np.arange(1, len(order) + 1)

    f = np.nansum(flux_all[order], axis=1)
    s = np.sqrt(np.nansum(sigma_all[order] ** 2, axis=1))
    cumf = np.cumsum(f)
    cum_s = np.sqrt(np.cumsum(s ** 2))
    with np.errstate(invalid='ignore', divide='ignore'):
        cum_snr = cumf / cum_s

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, gridspec_kw={'hspace': 0})

    ax1.plot(n_fibers, cumf / cumf[-1], color='black', lw=1.5)
    ax2.plot(n_fibers, cum_snr, color='black', lw=1.5)

    ax1.axhline(1.0, color='grey', alpha=0.3, lw=1, ls='dashed', zorder=0)
    if n_fib_coadd is not None:
        ax1.axvline(n_fib_coadd, color='black', lw=1.2, ls='dotted', zorder=1, label=f'co-add ({n_fib_coadd} fibers)')
        ax2.axvline(n_fib_coadd, color='black', lw=1.2, ls='dotted', zorder=1)
    ax1.set_ylabel('fraction of total flux', fontsize=13, labelpad=15)
    ax1.set_title(f"{obj_dict['object']}", fontsize=15, pad=14)
    ax1.legend(fontsize=12, loc='lower right')

    ax2.set_ylabel('cumulative SNR', fontsize=13, labelpad=15)
    ax2.set_xlabel('# fibers (brightest to faintest)', fontsize=13, labelpad=15)

    plt.tight_layout()
    if savepath:
        save_dir = os.path.join(savepath, obs_date, obj_dict['exposure_id']) if obs_date else os.path.join(savepath, obj_dict['exposure_id'])
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename), dpi=500, bbox_inches='tight')
    plt.show() if show else plt.close()


def plot_fiber_selection(flux_all, fibers, obj_dict, label, savepath, filename, obs_date=None, show=True):
    '''
    IFU fiber map: each fiber at its on-sky (X, Y) coloured by mean flux, with
    fibers used for a bright-fiber sum/coadd circled in red.
    '''
    astrometry = load_astrometry()
    x, y = astrometry['X_arcsec'].values, astrometry['Y_arcsec'].values
    mean_flux = np.nanmean(flux_all, axis=1)
    vmin, vmax = np.nanpercentile(mean_flux[np.isfinite(mean_flux)], [2, 98])

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(x, y, c=mean_flux, s=100, vmin=vmin, vmax=vmax, cmap='viridis', edgecolors='black', linewidths=0.4)
    ax.scatter(x[fibers], y[fibers], s=180, facecolors='none', edgecolors='red', linewidths=1.2)

    ax.set_title(f"{obj_dict['object']} -- {len(fibers)} fibers ({label})", fontsize=13, pad=15)
    ax.set_xlabel('X [arcsec]', fontsize=13, labelpad=10)
    ax.set_ylabel('Y [arcsec]', fontsize=13, labelpad=10)
    ax.set_aspect('equal')
    cb = fig.colorbar(sc, ax=ax, location='right', pad=0.02)
    cb.set_label('mean flux', fontsize=12, labelpad=12)

    if savepath:
        save_dir = os.path.join(savepath, obs_date, obj_dict['exposure_id']) if obs_date else os.path.join(savepath, obj_dict['exposure_id'])
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename), dpi=500, bbox_inches='tight')
    plt.show() if show else plt.close()


def select_snr_fibers(flux_all, sigma_all, wave_mask, snr_thresh=3):
    '''
    Fibers whose median SNR within wave_mask exceeds snr_thresh.
    '''
    with np.errstate(invalid='ignore', divide='ignore'):
        snr = flux_all[:, wave_mask] / sigma_all[:, wave_mask]   # sigma_all is NaN at bad pixels, so those drop out of the median
    fiber_snr = np.nanmedian(snr, axis=1)
    return fiber_snr > snr_thresh


def coadd_fibers(flux_all, sigma_all, fibers):
    '''
    Co-add IFU source fibers. 
    '''

    flux = np.nansum(flux_all[fibers], axis=0)
    sigma = np.sqrt(np.nansum(sigma_all[fibers]**2, axis=0))
    return flux, sigma

def coadd_fibers_averaged(flux_all, sigma_all, fibers, n_dither):
    '''
    Flux-conserving co-add across dithered exposures.

    Within each exposure the in-region fibres are summed, then the per-exposure spectra are inverse-variance averaged. 
    Bad fibers do not contribute.

    Returns (flux, sigma, ngood)
    '''
    n_fib = flux_all.shape[0] // n_dither
    num = 0.0   # sum_e ivar_e * flux_e
    den = 0.0   # sum_e ivar_e
    ngood = np.zeros(flux_all.shape[1], dtype=int)   # good fibre-pixels per wavelength
    for e in range(n_dither):
        sl = slice(e * n_fib, (e + 1) * n_fib)
        m = fibers[sl]
        f_block = flux_all[sl][m]
        s_block = sigma_all[sl][m]
        # usable only where sigma is finite and positive (bad pixels have NaN sigma)
        gp_block = np.isfinite(s_block) & (s_block > 0)
        ngood += gp_block.sum(axis=0)
        # mask the flux too (not just sigma) so bad-pixel flux can't leak into the sum
        flux_e = np.nansum(np.where(gp_block, f_block, np.nan), axis=0)
        sig_e = np.sqrt(np.nansum(np.where(gp_block, s_block, np.nan) ** 2, axis=0))
        ivar_e = np.zeros_like(sig_e)
        gp = np.isfinite(sig_e) & (sig_e > 0)
        ivar_e[gp] = 1.0 / sig_e[gp] ** 2
        num = num + flux_e * ivar_e
        den = den + ivar_e

    flux = np.full_like(den, np.nan)
    sigma = np.full_like(den, np.nan)
    good = den > 0
    flux[good] = num[good] / den[good]
    sigma[good] = 1.0 / np.sqrt(den[good])
    return flux, sigma, ngood


def half_light_radius_rough_estimate(flux_all, x, y, fiber_spacing=1.636):
    """
    Discrete, fiber-sampled estimate on the half-light radius of the source in the IFU field-of-view.
    
    Returns (center, reff) in arcsec
    """
    f = np.clip(np.nan_to_num(np.nansum(flux_all, axis=1)), 0, None)  # sum up each fiber's total flux and clip any negative values to zero

    # if the sum of the flux across all fibers is zero, likely no source in field of view
    if f.sum() <= 0:
        raise ValueError('All fibres have flux <= 0 -- cannot locate a source.')

    # define the center of the source on the IFU using a brightness-weighted average of fiber positions (i.e., center can be between fibers... not on the brightest fiber)
    xc = float(np.sum(f * x) / f.sum())  # [arcsec]
    yc = float(np.sum(f * y) / f.sum())  # [arcsec]

    # compute the distance of each fiber from this center position
    r = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)  # [arcsec]
    # sort from closest to farthest from the center
    order = np.argsort(r)
    # compute the light contained within fibers
    #   at index 0, this is the flux of the 1st closest fiber from the center
    #   at index 1, this is the sum of the flux of the 1st and 2nd closest fibers from the center ... etc
    cumf = np.cumsum(f[order])
    # the total light captured in the field of view
    total_cummulative_flux = cumf[-1]
    # half of that ...
    half_light_flux = 0.5 * total_cummulative_flux
    # search for at what index in our sorted cummulative summed flux array we reach this flux value
    half_light_idx = np.searchsorted(cumf, half_light_flux)
    # retrieve the half-light radius using that index value
    reff = float(r[order][half_light_idx]) 
    # ensure that the radius cannot be smaller than the spacing between fibers
    reff = max(reff, fiber_spacing) 

    # returning the center position, and the computed half-light radius
    return (xc, yc), reff