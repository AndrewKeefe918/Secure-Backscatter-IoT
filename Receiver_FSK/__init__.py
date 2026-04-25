"""PlutoSDR backscatter receiver package — FSK version.

Binary FSK between 1.0 kHz ('1' bits) and 1.7 kHz ('0' bits) on the
backscatter subcarrier. The two frequencies are non-harmonic so their
respective spectral content does not overlap, giving the receiver a
clean two-frequency decision per chip.
"""
