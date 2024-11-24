#!/usr/bin/env python3

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

sys.path.append(os.getcwd() + '/..')
#sys.path.append(os.getcwd() + '/../../..')
#sys.path.append(os.getcwd() + '/../..')
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


def test_sampling():

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

    plt.plot(wav_times, wav_samples, label='original')
    plt.plot(resample_times, resample_samples, label='resampled')
    plt.plot(length_times, length_samples, ':', label='length')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (-)")
    plt.legend()
    plt.show()


############################################################
# FEATURE VECTOR GENERATION
# (transform wave file to features, run tests, etc)
############################################################


def get_samples_from_wav(filename):
    wav_samples, wav_sample_rate = wav_to_raw_data(filename)
    resample_samples, _ = resample_data(wav_samples, wav_sample_rate, params.SAMPLE_RATE)
    length_samples = adjust_num_samples(resample_samples, params.SAMPLE_PERIOD*params.SAMPLE_RATE)
    return length_samples

def generate_mfcc(samples):
    pass
    # output: 1D np array

def generate_nothing(samples):
    return samples

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
    #print(len(averaged))

    #hist = np.histogram(fft, bins=200)
    #plt.plot(np.log(averaged))
    #plt.plot(hist[0])
    #plt.show()
    return averaged

def generate_closer_to_mfcc(samples):
    predct = generate_something_like_mfcc(samples)
    timedomain = np.fft.fft(predct)
    return timedomain

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

def rescale_to_bounds(input_vec):
    vmin = np.min(input_vec)
    vmax = np.max(input_vec)
    return (input_vec-vmin)/(vmax-vmin)

def test_vectors_with_humans():
    """Returns a shuffled batch of examples of all audio classes.

    Note that this is just a toy function because this is a simple demo intended
    to illustrate how the training code might work.

    Returns:
      a tuple (features, labels) where features is a NumPy array of shape
      [batch_size, num_frames, num_bands] where the batch_size is variable and
      each row is a log mel spectrogram patch of shape [num_frames, num_bands]
      suitable for feeding VGGish, while labels is a NumPy array of shape
      [batch_size, num_classes] where each row is a multi-hot label vector that
      provides the labels for corresponding rows in features.
    """

    yes_drone_files = get_wav_files("../data/drone_audio_dataset/Binary_Drone_Audio/yes_drone")
    no_drone_files = get_wav_files("../data/drone_audio_dataset/Binary_Drone_Audio/unknown")

    fig, axs = plt.subplots(4, 4)
    for x in [0,1]:
        for y in range(4):
            samples = get_samples_from_wav(random.choice(yes_drone_files))
            #data = generate_something_like_mfcc(samples)
            data = generate_nothing(samples)
            #data = generate_closer_to_mfcc(samples)
            axs[x,y].plot(rescale_to_bounds(data[150:]), 'k')
            axs[x,y].set_title("yes drone")
    for x in [2,3]:
        for y in range(4):
            samples = get_samples_from_wav(random.choice(no_drone_files))
            #data = generate_something_like_mfcc(samples)
            data = generate_nothing(samples)
            #data = generate_closer_to_mfcc(samples)
            axs[x,y].plot(rescale_to_bounds(data[150:]))
            axs[x,y].set_title("no drone")
    plt.show()

if __name__=='__main__':
    #test_sampling()
    #test()
    test_vectors_with_humans()

