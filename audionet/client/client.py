#!/usr/bin/env python3

# Main sensor node script.
#
# - connect to microphone (or test audio file)
# - preprocess audio data:
#   - maybe some filtering, TBD
#   - create spectrogram every N seconds
# - process data
#   - run ML model
# - send result to server

import pyaudio      # audio processing; see https://people.csail.mit.edu/hubert/pyaudio/docs/#
import wave         # wave file input
import numpy as np
import time
import sys
import os

#################################################
# AUDIO SYSTEM PARAMETERS
#################################################

# Get shared audio processing parameters
sys.path.append(os.getcwd() + '/../..')
from audionet import params

sample_rate = params.SAMPLE_RATE # (Hz) sample rate
sample_period = params.SAMPLE_PERIOD # (s) total sampling time
# use 32-bit floats for all audio processing (hardcoded)

samples_per_period = sample_rate*sample_period

print('Using sample rate {} kHz'.format(sample_rate/1000))
print('Using sampling period {} s'.format(sample_period))

# circular buffer containing all current audio samples
# (continuously feed data into this buffer, then
# extract detection_period worth of data when ready)
data_buffer = np.zeros(samples_per_period)

#################################################
# AUDIO INPUT
#################################################

def initialize_audio():
    """Set up audio stream."""

    print("Initializing audio...")

    # ALSA gives error messages; can ignore these
    # https://stackoverflow.com/questions/7088672/pyaudio-working-but-spits-out-error-messages-each-time

    pya = pyaudio.PyAudio()

    # find microphone device
    num_devices = pya.get_device_count()
    dev_number = None
    print('Found {} audio devices'.format(num_devices))
    for i in range(num_devices):
        dev_info = pya.get_device_info_by_index(i)
        # returns dictionary:
        # {'index': 5, 'structVersion': 2, 'name': 'USB Camera: Audio (hw:1,0)', 'hostApi': 0, 'maxInputChannels': 1, 'maxOutputChannels': 0, 'defaultLowInputLatency': 0.008707482993197279, 'defaultLowOutputLatency': -1.0, 'defaultHighInputLatency': 0.034829931972789115, 'defaultHighOutputLatency': -1.0, 'defaultSampleRate': 44100.0}
        # assume correct device is the first one connected via USB
        if 'usb' in dev_info['name'].lower():
            dev_number = i
            print("Using device {}: '{}'".format(i, dev_info['name']))
            print("Device information:")
            print(dev_info)
            break

    # pyaudio formats:
    # https://people.csail.mit.edu/hubert/pyaudio/docs/#pasampleformat

    sample_rate = None
    for rate in [16000, 32000, 44100]: # Hz
        try:
            rate_works = pya.is_format_supported(rate=rate, input_device=dev_number, input_channels=1, input_format=pyaudio.paFloat32)
            sample_rate = rate
            break
        except ValueError:
            print("invalid sample rate")
    print("Using sample rate {} kHz".format(sample_rate/1000))

    #stream = None
    #stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
    #                channels=1,
    #                rate=sample_rate,
    #                output=True)

    stream = None
    #stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
    #                channels=wf.getnchannels(),
    #                rate=wf.getframerate(),
    #                output=True)
    return (pya, stream)

def initialize_server():
    """Connect to server to send data."""
    print("Testing server...")

def sample_and_process_data():
    """Get audio sample, process, run model."""


    # maybe rescale data?

    sample_rate = 16000 # Hz
    sample_period = 1.0 # seconds

    read();


def send_data_to_server():
    """Send data to server."""
    pass

def run():
    time.sleep(1) # TODO remove
    sample_and_process_data()
    send_data_to_server()

if __name__=='__main__':
    pya, stream = initialize_audio() # get audio input
    initialize_server()
    print("Running...")
    while True:
        run()
