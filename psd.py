#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import pyfftw, multiprocessing
from scipy.signal import windows

from preprocessing import Preprocessing
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def handle_windows(window_length, window=None, beta=None):
    '''
    handling various windows

    window_length:      length of the tapering window
    window:             to be chosen from ["bartlett", "blackman", "hamming", "hanning", "kaiser", etc.] (from scipy.signal.windows)
                        if None, a rectangular window is implied
                        if "kaiser" is given, an additional argument of beta is expected
    '''
    if window is None:
        window_sequence = np.ones(window_length)
    elif window == "kaiser":
        if beta is None:
            raise ValueError("additional argument beta is empty!")
        else:
            window_sequence = windows.kaiser(window_length, beta)
    else:
        window_func = getattr(windows, window)
        window_sequence = window_func(window_length)
    return window_sequence

def psd_btm(bud, offset, window_length, padding_ratio=0, window=None, beta=None):
    '''
    Correlation (Blackman-Tukey) Method Spectral Estimation

    bud:                preprocessing object to be estimated (including data array (len N), fs)
    offset:             number of IQ pairs to be skipped over
    window_length:      length of the tapering window
    padding_ratio:      >= 1, ratio of the full frame length after zero padding to the window length
                        note that the final frame length will be rounded up to the next power of base 2
    window:             to be chosen from ["bartlett", "blackman", "hamming", "hanning", "kaiser"]
                        if None, a rectangular window is implied
                        if "kaiser" is given, an additional argument of beta is expected
    '''
    n_thread = multiprocessing.cpu_count()
    window_sequence = handle_windows(window_length, window, beta)
    # round the padded frame length up to the next radix-2 power
    n_point = int( np.power(2, np.ceil(np.log2(window_length*padding_ratio))) ) if padding_ratio >= 1 else window_length
    # build the frequency sequence
    frequencies = np.linspace(-bud.sampling_rate/2, bud.sampling_rate/2, n_point+1)[:-1] # Hz
    if n_point % 2 == 1: frequencies += bud.sampling_rate / (2*n_point)
    # number of freedom
    n_dof = 2    
    # create an FFT plan
    dummy = pyfftw.empty_aligned(window_length)
    fft = pyfftw.builders.fft(dummy, n=window_length, overwrite_input=True, threads=n_thread)
    ifft = pyfftw.builders.ifft(dummy, n=window_length, overwrite_input=True, threads=n_thread)
    fft_1 = pyfftw.builders.fft(dummy, n=n_point, overwrite_input=True, threads=n_thread)
    # set the signal
    signal = bud.load(window_length, offset)[1]
    # processing
    psd = np.fft.fftshift(fft_1(ifft(np.absolute(fft(signal))**2) * window_sequence)) / bud.sampling_rate
    return frequencies, psd, n_dof # Hz, V^2/Hz, 1

def psd_welch(bud, offset, window_length, average, overlap_ratio, padding_ratio=0, window=None, beta=None):
    '''
    Average Periodogram (Welch) Method Spectral Estimation

    bud:                preprocessing object to be estimated (including data array (len N), fs)
    offset:             number of IQ pairs to be skipped over
    window_length:      length of the tapering window, len L
    average:            length of the average, len K
    overlap_ratio:      the overlap ratio for the L-D points and if K sequences cover the entire N data points
                        N = L + D * ( K - 1 )
                        overlap_ratio = 1 - D / L
    padding_ratio:      >= 1, ratio of the full frame length after zero padding to the window length
                        note that the final frame length will be rounded up to the next power of base 2
    window:             to be chosen from ["bartlett", "blackman", "hamming", "hanning", "kaiser"]
                        if None, a rectangular window is implied
                        if "kaiser" is given, an additional argument of beta is expected
    ***
    https://ccrma.stanford.edu/~jos/sasp/Welch_s_Method_Windows.html#sec:wwelch
    overlap_ratio should always match the window to reduce side-lobe level
    rectangular window, overlap_ratio = 0, D = L
    non-rectangular window, overlap_ratio >= 0.5 , D <= L / 2
                Hamming, Hanning, and any other generalized Hamming window, overlap_ratio = 0.5
                Blackman window, overlap_ratio = 2/3 , D = L / 3
    '''
    n_thread = multiprocessing.cpu_count()
    window_sequence = handle_windows(window_length, window, beta)
    # round the padded frame length up to the next radix-2 power
    n_point = int( np.power(2, np.ceil(np.log2(window_length*padding_ratio))) ) if padding_ratio >= 1 else window_length
    # build the frequency sequence
    frequencies = np.linspace(-bud.sampling_rate/2, bud.sampling_rate/2, n_point+1)[:-1] # Hz
    if n_point % 2 == 1: frequencies += bud.sampling_rate / (2*n_point)
    if window_length == int (window_length * overlap_ratio): overlap_ratio = 0.5
    D = int((1 - overlap_ratio) * window_length)
    # number of freedom
    n_dof = 2 * average
    # set signal
    x = bud.load(int(window_length + D * (average - 1)), offset)[1] # N = window_length + D * (average - 1)
    signal = np.lib.stride_tricks.as_strided(x, (average, window_length), (x.strides[0] * D, x.strides[0])) * window_sequence
    # create an FFT plan
    dummy = pyfftw.empty_aligned((average, window_length))
    fft = pyfftw.builders.fft(dummy, n=n_point, overwrite_input=True, threads=n_thread)
    psd = np.mean(np.absolute(np.fft.fftshift(fft(signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)
    return frequencies, psd, n_dof # Hz, V^2/Hz, 1

def psd_multitaper(bud, offset, window_length, padding_ratio=0, NW=3, Kmax=4):
    '''
    Multitaper Method (MTM)

    bud:                preprocessing object to be estimated (including data array (len N), fs)
    offset:             number of IQ pairs to be skipped over
    window_length:      length of the tapering window, a.k.a. L
    padding_ratio:      >= 1, ratio of the full frame length after zero padding to the window length
                        note that the final frame length will be rounded up to the next power of base 2
    NW:                 standardized half bandwidth, a.k.a. NW
                        2 * NW = BW * fs
    Kmax:               number of DPSS windows to return (order 0 through Kmax-1)
    '''
    n_thread = multiprocessing.cpu_count()
    window_sequence = windows.dpss(window_length, NW, Kmax, return_ratios=False)
    # round the padded frame length up to the next radix-2 power
    n_point = int( np.power(2, np.ceil(np.log2(window_length*padding_ratio))) ) if padding_ratio >= 1 else window_length
    # build the frequency sequence
    frequencies = np.linspace(-bud.sampling_rate/2, bud.sampling_rate/2, n_point+1)[:-1] # Hz
    if n_point % 2 == 1: frequencies += bud.sampling_rate / (2*n_point)
    # number of degrees of freedom
    n_dof = 2 * Kmax
    # create an FFT plan
    dummy = pyfftw.empty_aligned((Kmax, window_length))
    fft = pyfftw.builders.fft(dummy, n=n_point, overwrite_input=True, threads=n_thread)
    # set signal
    signal = bud.load(window_length, offset)[1] * window_sequence
    eigpsd = np.fft.fftshift(fft(signal))
    psd = np.mean(np.absolute(eigpsd)**2, axis=0) / bud.sampling_rate
    return frequencies, psd, n_dof

def psd_adaptive_multitaper(bud, offset, window_length, padding_ratio=0, NW=3, Kmax=4):
    '''
    Adaptive Multitaper Method (AMTM)

    bud:                preprocessing object to be estimated (including data array (len N), fs)
    offset:             number of IQ pairs to be skipped over
    window_length:      length of the tapering window, a.k.a. L
    padding_ratio:      >= 1, ratio of the full frame length after zero padding to the window length
                        note that the final frame length will be rounded up to the next power of base 2
    NW:                 standardized half bandwidth, a.k.a. NW
                        2 * NW = BW * fs
    Kmax:               number of DPSS windows to return (order 0 through Kmax-1)
    '''
    n_thread = multiprocessing.cpu_count()
    window_sequence, ratio = windows.dpss(window_length, NW, Kmax, return_ratios=True)
    ratio = ratio.reshape(Kmax, 1)
    # round the padded frame length up to the next radix-2 power
    n_point = int( np.power(2, np.ceil(np.log2(window_length*padding_ratio))) ) if padding_ratio >= 1 else window_length
    # build the frequency sequence
    frequencies = np.linspace(-bud.sampling_rate/2, bud.sampling_rate/2, n_point+1)[:-1] # Hz
    if n_point % 2 == 1: frequencies += bud.sampling_rate / (2*n_point)
    # create an FFT plan
    dummy = pyfftw.empty_aligned((Kmax, window_length))
    fft = pyfftw.builders.fft(dummy, n=n_point, overwrite_input=True, threads=n_thread)
    # set signal
    signal = bud.load(window_length, offset)[1] * window_sequence
    eigpsd = np.fft.fftshift(fft(signal))
    psd = np.mean(np.absolute(eigpsd[:2])**2, axis=0) / bud.sampling_rate
    while True:
        variance = np.sum(psd) / n_point
        weight = (psd / (ratio*psd + (1-ratio)*variance))**2 * ratio
        psd_temp = np.average(np.absolute(eigpsd)**2, axis=0, weights=weight) / bud.sampling_rate
        if np.allclose(psd_temp, psd, rtol=1e-5, atol=1e-5): break
        psd = psd_temp
    # number of degrees of freedom
    n_dof = 2 * np.sum(weight, axis=0)**2 / np.sum(weight**2, axis=0)
    return frequencies, psd, n_dof


