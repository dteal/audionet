#!/usr/bin/env python3

# Functions to:
# - open and parse wave files
# - resample/change audio sample frequency/length
# - create feature vectors from audio samples

import math
import sys
import os
import wave
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import random
from pathlib import Path
from typing import List

sys.path.append(os.getcwd() + '/..') # hack
from audionet import params

############################################################
# SAMPLING CODE
# (get audio from wave file, resample to right frequency/length, etc)
############################################################

# open a single wave file and convert to numpy array in [-1,1]
# HOWEVER: IS NOT RESAMPLED TO PROPER TIME PERIOD OR LENGTH YET
#
# RETURNS:
# samples: numpy array of floats in [-1, 1] (holding the audio signal)
# wav_sample_rate: the number of samples per second
def wav_to_raw_data(filename):

    times = [] # (s) array of sample times
    samples = [] # (-1 to +1 [mean 0] signal)
    wav_sample_rate = None

    # open wave file
    with wave.open(filename, mode='rb') as data:

        num_frames = data.getnframes()
        bytes_per_sample = data.getsampwidth()

        max_value = 2**(8*bytes_per_sample-1)*1.0

        wav_sample_rate = data.getframerate() # Hz, sample rate
        cur_time = 0

        while True:
            value = data.readframes(1)[0:bytes_per_sample] # only get first channel (if applicable)
            if value == b'':
                break # end of file
            # value is a signed integer, usually 16-bit,
            # where the audio signal is centered at zero.
            # convert to float in [-1,1]
            samples.append(int.from_bytes(value, byteorder='little', signed=True)/max_value) # convert to float
            times.append(cur_time)
            cur_time = cur_time + 1/wav_sample_rate

    samples = np.array(samples) # convert to numpy array

    #plt.plot(times, samples)
    #plt.xlabel("Time (s)")
    #plt.ylabel("Amplitude (-)")
    #plt.show()

    return samples, wav_sample_rate

# Converts samples from one sample rate to another
# (both up and down) (just wraps scipy function)
#
# INPUT:
# original_samples: numpy array of floats in range [-1, 1]
# original_sample_rate: sampling rate of that data, in Hz
#
# OUTPUT (as tuple):
# new_samples: numpy array of floats in range [-1, 1]
# new_sample_rate: desired sample rate, in Hz
def resample_data(original_samples, original_sample_rate, new_sample_rate, verbose=True):

    # to save time, don't resample if we don't need to
    if original_sample_rate == new_sample_rate:
        return (original_samples, original_sample_rate)

    sampled_period = len(original_samples)/original_sample_rate # (seconds), total time the samples cover
    if verbose:
        print("converting sample rate from {} kHz to {} kHz".format(original_sample_rate/1000, new_sample_rate/1000))
    new_samples = sig.resample(original_samples, int(sampled_period*new_sample_rate))
    return (new_samples, new_sample_rate)

# adjusts a sample to do a thing
# gets correct number of samples
def adjust_num_samples(original_samples, desired_num_samples):
    original_length = len(original_samples)

    # don't do anything if we don't have to
    if original_length == desired_num_samples:
        return original_samples

    # crop the sample if it's too long
    elif original_length > desired_num_samples:
        return original_samples[0:desired_num_samples]

    # if the sample is too short, just copy it multiple times
    else: # original_length < desired_num_samples
        new_samples = np.zeros((desired_num_samples))
        cur_samples = 0
        while cur_samples + original_length < desired_num_samples:
            new_samples[cur_samples:cur_samples+original_length] = original_samples
            cur_samples = cur_samples + original_length
        new_samples[cur_samples:-1] = original_samples[0:desired_num_samples-cur_samples-1]
        return new_samples

def test_resampling():

    # input wave file
    wav_samples, wav_sample_rate = wav_to_raw_data("sample_sound.wav")

    # resample frequency
    resample_sample_rate = 8000 # (Hz) test frequency to resample to
    resample_samples, resample_sample_rate = resample_data(wav_samples, wav_sample_rate, resample_sample_rate)

    # change length
    desired_length = 5 # (seconds) total length
    length_samples = adjust_num_samples(resample_samples, desired_length*resample_sample_rate)

    # plot all signals to make sure they're the same
    wav_times = np.linspace(0, (len(wav_samples)-1.0)/wav_sample_rate, len(wav_samples))
    resample_times = np.linspace(0, (len(resample_samples)-1.0)/resample_sample_rate, len(resample_samples))
    length_times = np.linspace(0, (len(length_samples)-1.0)/resample_sample_rate, len(length_samples))

    # should be three graphs, and the data should line up
    plt.plot(wav_times, wav_samples, label='original')
    plt.plot(resample_times, resample_samples, label='resampled')
    plt.plot(length_times, length_samples, ':', label='length')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (-)")
    plt.legend()
    plt.show()

############################################################
# UTILITY FUNCTIONS
############################################################

def get_samples_from_wav(filename):
    """Open wav file, get sound samples, adjusted to desired sample rate and length"""
    wav_samples, wav_sample_rate = wav_to_raw_data(filename)
    resample_samples, _ = resample_data(wav_samples, wav_sample_rate, params.SAMPLE_RATE)
    length_samples = adjust_num_samples(resample_samples, params.SAMPLE_PERIOD*params.SAMPLE_RATE)
    return length_samples

def normalize_bounds(samples):
    """Adjust/normalize magnitude of samples to fit range [0, 1]"""
    smin = np.min(samples)
    smax = np.max(samples)
    return (samples-smin)/(smax-smin)

def get_wav_files(directory: str):
    """
    Get all .wav files from the specified directory.

    Args:
        directory (str): Path to the directory containing .wav files

    Returns:
        List[str]: List of .wav file paths

    Raises:
        FileNotFoundError: If the directory doesn't exist
        PermissionError: If there's no permission to access the directory
    """
    try:
        # Convert to Path object for better cross-platform compatibility
        path = Path(directory)

        # Verify directory exists
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Get all .wav files
        wav_files = [str(f) for f in path.glob("*.wav")]

        # Sort for consistent ordering
        wav_files.sort()

        return wav_files

    except PermissionError:
        raise PermissionError(f"Permission denied accessing directory: {directory}")

############################################################
# FEATURE VECTOR GENERATION
# (transform wave file to features, run tests, etc)
############################################################

# assorted functions to process samples into feature vectors

def generate_nothing(samples):
    """The simplest feature vector: just return raw audio data. Not very useful."""
    return samples

def generate_fft(samples):
    """Take FFT (and nothing else so far)"""
    #print(samples)
    #print("sample rate: {} Hz".format(params.SAMPLE_RATE))
    #print("sample_period: {} s".format(params.SAMPLE_PERIOD))

    fft = np.fft.fft(samples)
    num_fft_points = params.SAMPLE_PERIOD*params.SAMPLE_RATE
    # fft[0] is DC signal; there shouldn't be any for audio, remove it
    # fft[1] is params.SAMPLE_RATE/num_fft_points
    # fft[floor(num_fft_points/2-1)] = just below nyquist
    fft = fft[1:math.floor(num_fft_points/2-1)]
    min_frequency = params.SAMPLE_RATE/num_fft_points
    max_frequency = params.SAMPLE_RATE/2 # might be fencepost error but whatever
    frequency_bins = np.linspace(min_frequency, max_frequency, len(fft))

    #plt.plot(fft)
    #plt.xlabel("Frequency (Hz)")
    #plt.ylabel("Amplitude (-)")
    #plt.show()
    print(len(fft))

    # length of output vector is (original number of samples/2)-2
    return fft

def generate_something_like_mfcc(samples, desired_feature_vector_size = 128):
    """This is our best attempt at a feature vector."""
    # desired_feature_vector_size = 128 seems to contain a good amount of info (see plots)

    # first take fft
    fft = np.fft.fft(samples)
    num_fft_points = params.SAMPLE_PERIOD*params.SAMPLE_RATE
    # fft[0] is DC signal; there shouldn't be any for audio, remove it
    # fft[1] is params.SAMPLE_RATE/num_fft_points
    # fft[floor(num_fft_points/2-1)] = just below nyquist
    fft = fft[1:math.floor(num_fft_points/2-1)]
    min_frequency = params.SAMPLE_RATE/num_fft_points
    max_frequency = params.SAMPLE_RATE/2 # might be fencepost error but whatever
    frequency_bins = np.linspace(min_frequency, max_frequency, len(fft))

    # now, take the absolute value
    fft = np.abs(fft+0.01)

    # take moving average
    moving_average_size = math.ceil(len(fft)/desired_feature_vector_size)
    #averaged = np.convolve(fft, np.ones(moving_average_size), mode='valid')/moving_average_size
    averaged = np.convolve(fft, np.ones(moving_average_size), mode='same')/moving_average_size

    # take logarithm
    logged = np.log(averaged+0.01) # offset from zero so there isn't a log(0) error

    # now sample at fixed points
    binned = sig.resample(logged, desired_feature_vector_size)
    new_frequency_bins = np.linspace(min_frequency, max_frequency, len(binned))

    # finally, normalize to [0, 1]
    rescaled = normalize_bounds(binned)

    #plt.plot(new_frequency_bins, rescaled)
    #plt.xlabel("Frequency (Hz)")
    #plt.ylabel("Amplitude (-)")
    #plt.show()

    #print(len(rescaled)) # should be desired_feature_vector_size
    return rescaled

def test_vectors_with_humans():
    """Displays feature vectors of random files to see if humans can tell the difference between drone / no drone"""

    yes_drone_files = get_wav_files("../data/drone_audio_dataset/Binary_Drone_Audio/yes_drone")
    no_drone_files = get_wav_files("../data/drone_audio_dataset/Binary_Drone_Audio/unknown")

    fig, axs = plt.subplots(4, 4)
    for x in [0,1]:
        for y in range(4):
            samples = get_samples_from_wav(random.choice(yes_drone_files))
            features = generate_something_like_mfcc(samples)
            #features = generate_nothing(samples)
            axs[x,y].plot(features, 'red')
            axs[x,y].set_title("yes drone")
    for x in [2,3]:
        for y in range(4):
            samples = get_samples_from_wav(random.choice(no_drone_files))
            features = generate_something_like_mfcc(samples)
            #features = generate_nothing(samples)
            axs[x,y].plot(features, 'blue')
            axs[x,y].set_title("no drone")
    plt.show()

if __name__=='__main__':

    # confirm resampling functions work right
    #test_resampling()

    # confirm feature vector algorithm looks good
    #samples = get_samples_from_wav('sample_sound.wav')
    #generate_something_like_mfcc(samples, 128)

    # check if everything works well
    test_vectors_with_humans()

