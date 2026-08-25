import os
import numpy as np
import pandas as pd
from astropy.io import fits

from .bitmask import MASK_DONOTUSE, write_maskbits_header
from .dither_combine import clip_edges_mask, CLIP_WAVE_BLUE_RANGE, CLIP_WAVE_RED_RANGE


# ----- shared HDU helpers --------------------------------------------------- #

def stamp(hdu, desc, unit=None):
    """
    Set EXTDESC (and BUNIT) on an HDU.
    """
    if unit is not None:
        hdu.header['BUNIT'] = unit
    hdu.header['EXTDESC'] = (desc, 'extension contents')
    return hdu


def copy_wcs(header, ref, keys=('CTYPE1', 'CRVAL1', 'CRPIX1', 'CDELT1', 'CUNIT1'), crpix_shift=0):
    """
    Copy spectral WCS keywords from ref into header.
    """
    for k in keys:
        if k in ref:
            header[k] = ref[k]
    if crpix_shift and 'CRPIX1' in header:
        header['CRPIX1'] = header['CRPIX1'] - crpix_shift
    return header


def ivar_from_sigma(sigma, mask=None):
    """
    set to zero where sigma is non-finite, negative, and at DONOTUSE.
    """
    sigma = np.asarray(sigma, float)
    ivar = np.zeros_like(sigma, np.float32)
    gp = np.isfinite(sigma) & (sigma > 0)
    ivar[gp] = 1.0 / sigma[gp] ** 2
    if mask is not None:
        ivar[(np.asarray(mask, np.int32) & MASK_DONOTUSE) != 0] = 0.0
    return ivar


def mask_hdu(mask):
    """
    MASK bitmask HDU (NIRWALS_DRP_PIXELMASK).
    """
    h = stamp(fits.ImageHDU(np.asarray(mask, np.int32), name='MASK'), 'quality bitmask (NIRWALS_DRP_PIXELMASK)')
    write_maskbits_header(h.header)
    return h


def obsinfo_hdu(rows):
    """
    OBSINFO binary table, one row per exposure.
    """
    keys = list(rows[0].keys())

    def col(k):
        vals = [r[k] for r in rows]
        if all(isinstance(v, str) for v in vals):
            return fits.Column(name=k.upper(), format=f'{max(len(v) for v in vals)}A', array=np.array(vals))
        return fits.Column(name=k.upper(), format='E', array=np.array(vals, dtype=float))

    h = fits.BinTableHDU.from_columns([col(k) for k in keys], name='OBSINFO')
    h.header['EXTDESC'] = ('per-exposure metadata (one row per exposure)', 'extension contents')
    return h


# ----- 2D post-processed product (row-stacked spectra + cube) --------------- #
def write_postprocessed_fits(combined, out_file, throughput_file=None):
    """
    Write the 2D product: PRIMARY, FLUX, IVAR, MASK, WAVE, SPECRES, SPECRESD,
    OBSINFO, XPOS, YPOS, SKYCORR, TELLCORR, FLUXCAL, CUBE, CUBE_IVAR.
    Wavelength-axis arrays are edge-clipped; extensions absent in `combined` are skipped.
    """
    ref = combined['exp']['ssc']['sci_header']
    bunit = 'erg/s/cm2/A' if throughput_file else 'counts/s'

    keep = clip_edges_mask(combined['wave'])
    if not keep.any():
        raise ValueError(f'edge clip [{CLIP_WAVE_BLUE_RANGE}, {CLIP_WAVE_RED_RANGE}] A keeps no wavelengths')
    i0 = int(np.argmax(keep))
    wave = combined['wave'][keep]
    has_mask = combined.get('mask_all') is not None

    # 0 PRIMARY
    pri = fits.PrimaryHDU()
    pri.header['OBJECT'] = (ref.get('OBJECT', combined['exp'].get('object', '')), 'science target')
    pri.header['NDITHER'] = (combined.get('n_dither', 1), 'number of combined dithered exposures')
    pri.header['NWAVE'] = (int(keep.sum()), 'number of wavelength pixels')
    pri.header['NFIBER'] = (int(combined['flux_all'].shape[0]), 'total fiber rows (NFIBER_PER_EXP * NEXP)')
    pri.header.add_history(f'Edge-clipped: {CLIP_WAVE_BLUE_RANGE:.0f} A blue, {CLIP_WAVE_RED_RANGE:.0f} A red')
    pri.header.add_history('Primary product: per-fiber spectra (telluric-corrected'
                           + (', flux-calibrated)' if throughput_file else ')'))
    hdus = [pri]

    # 1 FLUX (carries the spectral WCS)
    fx = fits.ImageHDU(combined['flux_all'][:, keep].astype(np.float32), name='FLUX')
    copy_wcs(fx.header, ref, crpix_shift=i0)
    hdus.append(stamp(fx, 'row-stacked fiber spectra in units of ergs/s/cm^2/A', bunit))

    # 2 IVAR
    ivar = ivar_from_sigma(combined['sigma_all'][:, keep], combined['mask_all'][:, keep] if has_mask else None)
    hdus.append(stamp(fits.ImageHDU(ivar, name='IVAR'), 'inverse variance of FLUX', f'({bunit})^-2'))

    # 3 MASK
    if has_mask:
        hdus.append(mask_hdu(combined['mask_all'][:, keep]))

    # 4 WAVE
    hdus.append(stamp(fits.ImageHDU(wave.astype(np.float64), name='WAVE'), 'wavelength vector in units of A', ref.get('CUNIT1', 'Angstrom')))

    # 5 SPECRES / 6 SPECRESD
    if combined.get('specres') is not None:
        hdus.append(stamp(fits.ImageHDU(np.asarray(combined['specres'])[keep].astype(np.float32), name='SPECRES'), 'median spectral resolution (R)'))
    if combined.get('specresd') is not None:
        hdus.append(stamp(fits.ImageHDU(np.asarray(combined['specresd'])[keep].astype(np.float32), name='SPECRESD'), 'standard deviation (1-sigma) of SPECRES'))

    # 7 OBSINFO
    if combined.get('obsinfo'):
        hdus.append(obsinfo_hdu(combined['obsinfo']))

    # 8 XPOS / 9 YPOS
    hdus.append(stamp(fits.ImageHDU(combined['x_arcsec'].astype(np.float32), name='XPOS'), 'fiber X-position relative to IFU center', 'arcsec'))
    hdus.append(stamp(fits.ImageHDU(combined['y_arcsec'].astype(np.float32), name='YPOS'), 'fiber Y-position relative to IFU center', 'arcsec'))

    # 10 SKYCORR
    if combined.get('skycorr') is not None:
        hdus.append(stamp(fits.ImageHDU(combined['skycorr'][:, keep].astype(np.float32), name='SKYCORR'), 'subtracted sky emission', bunit))

    # 11 TELLCORR
    if combined.get('tellcorr') is not None:
        hdus.append(stamp(fits.ImageHDU(np.asarray(combined['tellcorr'])[keep].astype(np.float32), name='TELLCORR'), 'telluric transmission correction'))

    # 12 FLUXCAL (throughput curve interpolated onto WAVE)
    if throughput_file is not None:
        try:
            t = pd.read_csv(throughput_file)
            tw, tt = t.iloc[:, 0].values, t.iloc[:, 1].values
            thru = np.interp(wave, tw, tt, left=np.nan, right=np.nan).astype(np.float32)
            hdus.append(stamp(fits.ImageHDU(thru, name='FLUXCAL'), 'flux-calibration throughput curve'))
        except Exception as e:
            print(f'  FLUXCAL: could not read throughput {throughput_file} ({e})')

    # 13 CUBE / 14 CUBE_IVAR
    cube = fits.ImageHDU(combined['cube'][keep].astype(np.float32), name='CUBE')
    cube.header['EXTDESC'] = ('3d data cube (all exposures combined)', 'extension contents')
    cube.header['BUNIT'] = bunit
    hdus.append(cube)
    hdus.append(stamp(fits.ImageHDU(combined['ivar_cube'][keep].astype(np.float32), name='CUBE_IVAR'), 'inverse variance of CUBE', f'({bunit})^-2'))

    fits.HDUList(hdus).writeto(out_file, overwrite=True)


# ----- 1D co-added product -------------------------------------------------- #

def write_1d_fits(template_file, wave, flux, sigma, out_file, throughput_file, mask=None):
    """
    Write the 1D co-add: PRIMARY, FLUX, IVAR, MASK, WAVE, SPECRES, SPECRESD --
    same layout/conventions as the 2D product. IVAR is zeroed at DONOTUSE; FLUX untouched.
    """
    wave = np.asarray(wave, np.float64)
    with fits.open(template_file) as tpl:
        names = [h.name for h in tpl]
        tpl_flux = tpl['FLUX'] if 'FLUX' in names else tpl['SCI']
        meta = tpl['OBSINFO'].header if 'OBSINFO' in names else tpl['PRIMARY'].header
        tpl_wave = np.asarray(tpl['WAVE'].data, float) if 'WAVE' in names else None

        # PRIMARY (carry obs metadata)
        pri = fits.PrimaryHDU()
        pri.header.extend(meta, strip=True, update=True)
        for k in ('EXTNAME', 'EXTVER', 'EXTDESC'):
            pri.header.remove(k, ignore_missing=True)

        # FLUX (linear WCS matching wave)
        fx = fits.ImageHDU(np.asarray(flux, np.float32), name='FLUX')
        copy_wcs(fx.header, tpl_flux.header, keys=('CTYPE1', 'CUNIT1'))
        fx.header['CRPIX1'] = 1.0
        fx.header['CRVAL1'] = float(wave[0])
        fx.header['CDELT1'] = float(wave[1] - wave[0])
        fx.header.add_history('Telluric corrected and flux calibrated' if throughput_file else 'Telluric corrected (no flux calibration -- no specphot standard given)')
        if throughput_file:
            fx.header.add_history(f'Flux calibration throughput: {os.path.basename(throughput_file)}')

        # IVAR
        iv = fits.ImageHDU(ivar_from_sigma(sigma, mask), name='IVAR')
        copy_wcs(iv.header, fx.header)
        if 'CUNIT1' in tpl_flux.header:
            iv.header['BUNIT'] = (f"({tpl_flux.header['CUNIT1']})^-2", 'inverse variance unit')

        # WAVE
        wv = fits.ImageHDU(wave, name='WAVE')
        if 'CUNIT1' in fx.header:
            wv.header['BUNIT'] = (fx.header['CUNIT1'], 'wavelength unit')

        # order: PRIMARY, FLUX, IVAR, MASK, WAVE, SPECRES, SPECRESD
        out = [pri, fx, iv]
        if mask is not None:
            mh = mask_hdu(mask)
            copy_wcs(mh.header, fx.header)
            out.append(mh)
        out.append(wv)

        if 'SPECRES' in names and tpl_wave is not None:
            R = np.interp(wave, tpl_wave, np.asarray(tpl['SPECRES'].data, float)).astype(np.float32)
            res = fits.ImageHDU(R, name='SPECRES')
            res.header['BUNIT'] = ('', 'Dimensionless R = lambda / FWHM')
            out.append(res)
        if 'SPECRESD' in names and tpl_wave is not None:
            Rd = np.interp(wave, tpl_wave, np.asarray(tpl['SPECRESD'].data, float)).astype(np.float32)
            out.append(fits.ImageHDU(Rd, name='SPECRESD'))

        fits.HDUList(out).writeto(out_file, overwrite=True)
