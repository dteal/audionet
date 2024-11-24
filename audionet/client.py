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

import pyaudio  # audio processing; see https://people.csail.mit.edu/hubert/pyaudio/docs/#
import wave  # wave file input
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import os

# Get shared audio processing parameters
sys.path.append(os.getcwd() + "/..")
from audionet import params
import preprocess


class AudioChart:
    def __init__(self):
        fig, axs = plt.subplots(1, 2)
        (self.line1,) = axs[0].plot([], [], "red")
        axs[0].set_title("Raw Data")
        (self.line2,) = axs[1].plot([], [], "blue")
        axs[1].set_title("Processed Data")

        self.line1.set_data([], [])
        self.line2.set_data([], [])

    def update(self, data, processed_data):
        x = range(len(data))
        self.line1.set_data(x, data)
        self.line2.set_data(x, processed_data)


class AudioSensorState:
    def __init__(self):
        self.desired_sample_rate = params.SAMPLE_RATE  # (Hz) sample rate we want
        self.desired_sample_period = params.SAMPLE_PERIOD  # (s) total sampling time
        self.mic_sample_rate = (
            None  # (Hz) actual number of samples from microphone per second
        )
        self.pya = None  # pyaudio instance
        self.stream = None  # pyaudio stream from microphone

    #################################################
    # AUDIO INPUT
    #################################################

    def initialize_audio(self):
        """Set up audio stream."""

        print("Initializing audio stream...")

        # initialize audio input library
        # ALSA gives error messages; can ignore these
        # https://stackoverflow.com/questions/7088672/pyaudio-working-but-spits-out-error-messages-each-time
        self.pya = pyaudio.PyAudio()

        # find the microphone we want to use
        num_devices = self.pya.get_device_count()
        dev_number = -1
        print("Found {} audio devices".format(num_devices))
        for i in range(num_devices):
            dev_info = self.pya.get_device_info_by_index(i)
            # print(dev_info)
            # returns dictionary:
            # {'index': 5, 'structVersion': 2, 'name': 'USB Camera: Audio (hw:1,0)', 'hostApi': 0, 'maxInputChannels': 1, 'maxOutputChannels': 0, 'defaultLowInputLatency': 0.008707482993197279, 'defaultLowOutputLatency': -1.0, 'defaultHighInputLatency': 0.034829931972789115, 'defaultHighOutputLatency': -1.0, 'defaultSampleRate': 44100.0}
            # assume correct device is the first one connected via USB
            if "usb" in dev_info["name"].lower():
                dev_number = i  # this is the pyaudio index of the device we will use
                print("Using device {}: '{}'".format(i, dev_info["name"]))
                print("Device information:")
                print(dev_info)
                # break
        # if dev_number == -1: # couldn't find microphone
        #    raise ValueError('Could not find microphone, exiting')

        # pyaudio formats:
        # https://people.csail.mit.edu/hubert/pyaudio/docs/#pasampleformat
        # (1: PCM integer, 3: IEEE 754 float)

        self.mic_sample_rate = 16000  # Hz
        # get some sample rate (whatever microphone can do, doesn't matter)
        # for rate in [16000, 32000, 44100]: # Hz
        for rate in [self.mic_sample_rate]:
            try:
                # rate_works = self.pya.is_format_supported(rate=rate, input_device=dev_number, input_channels=1, input_format=pyaudio.paInt16)
                mic_sample_rate = rate
                break
            except ValueError:
                print("invalid sample rate")

        # circular buffer containing all current audio samples
        # (continuously feed data into this buffer, then
        # extract detection_period worth of data when ready)

        mic_samples_per_period = mic_sample_rate * self.desired_sample_period

        # data_buffer = np.zeros(samples_per_period)

        print("Desired sampling rate {} kHz".format(self.desired_sample_rate / 1000))
        print("Using mic sampling rate {} kHz".format(self.mic_sample_rate / 1000))
        print("Desired sampling period {} s".format(self.desired_sample_period))

        # stream = None
        self.stream = self.pya.open(
            format=pyaudio.paInt16,
            # input_device_index = dev_number, # don't specify this; use default input device
            channels=1,
            rate=self.mic_sample_rate,
            frames_per_buffer=2
            * mic_samples_per_period,  # buffer at least this many samples
            input=True,
        )

        # 1 -> 191 (1024 bytes
        # 1 -> 191
        # 2 -> 191
        # 10 -> 191
        # 100 -> 299 (598 bytes)
        # 200 -> 599 (1198 bytes)
        # 1000 -> 2999 (5998 bytes)
        # 10000 -> 16016 (32032 bytes)

    def initialize_server(self):
        """Connect to server to send data."""
        print("Testing server...")

    def sample_and_process_data(self):
        """Get audio sample, process, run model."""
        print("---")
        # print(self.stream.is_active())
        print(time.time())
        num_available = self.stream.get_read_available()
        print(num_available)

        fig, axs = plt.subplots(1, 2)
        data = self.stream.read(num_available)
        print("read {} bytes".format(len(data)))
        return data

    def send_data_to_server(self):
        """Send data to server."""
        pass

    def run(self):
        self.sample_and_process_data()
        self.send_data_to_server()
        time.sleep(1)  # TODO remove


if __name__ == "__main__":
    chart = AudioChart()
    state = AudioSensorState()
    state.initialize_audio()
    state.initialize_server()
    print("Running...")
    while True:
        data = state.run()
        processed_data = preprocess.generate_something_like_mfcc(data)
        chart.update(data, processed_data)
