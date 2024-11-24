#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
import scipy.signal as sig

sys.path.append(os.getcwd() + '/../..')
#sys.path.append(os.getcwd() + '/../../..')
#sys.path.append(os.getcwd() + '/../..')

from audionet import params
from audionet.preprocess import resample

def get_samples_from_wav(filename):
    wav_samples, wav_sample_rate = resample.wav_to_raw_data(filename)
    resample_samples, _ = resample.resample_data(wav_samples, wav_sample_rate, params.SAMPLE_RATE)
    length_samples = resample.adjust_num_samples(resample_samples, params.SAMPLE_PERIOD*params.SAMPLE_RATE)
    return length_samples

def generate_mfcc(samples):
    pass
    # output: 1D np array


def generate_fft(samples):
    print(samples)
    print("sample rate: {} Hz".format(params.SAMPLE_RATE))
    print("sample_period: {} s".format(params.SAMPLE_PERIOD))
    fft = np.fft.fft(samples)
    fft = fft[:int(len(samples)/2)]
    min_frequency = 1/params.SAMPLE_PERIOD # min frequency, Hz
    max_frequency = params.SAMPLE
    print("min frequency: {} Hz".format(min_frequency))
    print("max frequency: {} Hz".format(min_frequency))
    #freqs = np.logspace(0,

    plt.plot(fft)
    plt.show()
    return fft

def generate_something_like_mfcc(samples):
    print(samples)
    fft = np.fft.fft(samples)
    num_fft_bins = len(fft)
    fft = fft[:int(num_fft_bins/2)]

    fft = np.abs(fft)

    # take moving average
    #feature_vec_size = 128
    #bin_size = math.floor(len(fft)/feature_vec_size)
    bin_size = 100
    averaged = np.convolve(fft, np.ones(bin_size), mode='valid')/bin_size
    print(len(averaged))

    #hist = np.histogram(fft, bins=200)
    plt.plot(np.log(averaged))
    #plt.plot(hist[0])
    plt.show()
    return fft

# samples - 1D np array

def generate_spectrogram(samples):
    params.SAMPLE_RATE # Hz
    params.SAMPLE_PERIOD # s

    pass
    # output - 2D np array


def test():

    samples = get_samples_from_wav('sample_sound.wav')
    #generate_fft(samples)
    generate_something_like_mfcc(samples)
    pass

if __name__=='__main__':
    test()
