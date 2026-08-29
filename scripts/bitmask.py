MASKBITS = {
    'BADPIX':      (1, 'pixel flagged in the reduced bad-pixel mask'),
    'SKYLINE':     (2, 'strong OH sky line (informational; elevated sky shot noise)'),
    'TELLURIC': (4, 'telluric transmission < 0.6'),
    'DONOTUSE':    (8, 'do not use for analysis (= BADPIX or TELLURIC)'),
}

# get bit values
MASK_BADPIX = MASKBITS['BADPIX'][0]
MASK_SKYLINE = MASKBITS['SKYLINE'][0]
MASK_LOWTELL = MASKBITS['TELLURIC'][0]
MASK_DONOTUSE = MASKBITS['DONOTUSE'][0]

# bits that DONOTUSE is built from
DONOTUSE_BITS = MASK_BADPIX | MASK_LOWTELL

# thresholds for the positional flags
TELL_MIN = 0.8      # TELLURIC below this transmission
SKY_NSIG = 5.0      # SKYLINE where sky > median + SKY_NSIG * MAD


def write_maskbits_header(header):
    """
    Stamp MASKBIT<n> cards onto a FITS header so the MASK HDU self-describes.
    """
    for label, (value, desc) in MASKBITS.items():
        bit = value.bit_length() - 1            # 1->0, 2->1, 4->2, 8->3
        header[f'MASKBIT{bit}'] = (label, f'value {value}: {desc}')
    return header