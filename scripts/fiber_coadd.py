import numpy as np
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.centroids import centroid_1dg


def select_bright_fibers(flux_all, frac=0.2):
    '''
    Fibers whose mean flux exceeds frac * brightest fiber's mean flux.
    '''
    mean_flux = np.nanmean(flux_all, axis=1)
    return np.where(mean_flux > frac * np.nanmax(mean_flux))[0]


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

    Within each exposure the in-region fibres are summed, then the per-exposure
    spectra are inverse-variance averaged. Bad fibers do not contribute.

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