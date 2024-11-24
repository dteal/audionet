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
        #(self.line2,) = axs[1].plot([], [], "blue")
        #axs[1].set_title("Processed Data")

        self.line1.set_data([], [])
        #self.line2.set_data([], [])

    #def update(self, data, processed_data):
    def update(self, data):
        x = range(len(data))
        self.line1.set_data(x, data)
        #self.line2.set_data(x, processed_data)


class AudioSensorState:
    def __init__(self):
        self.desired_sample_rate = params.SAMPLE_RATE  # (Hz) sample rate we want
        self.desired_sample_period = params.SAMPLE_PERIOD  # (s) total sampling time
        self.mic_sample_rate = (
            None  # (Hz) actual number of samples from microphone per second
        )
        self.pya = None  # pyaudio instance
        self.stream = None  # pyaudio stream from microphone
        # circular buffer containing raw microphone data
        self.current_raw_samples = [] #np.ones(self.mic_sample_rate*self.desired_sample_period)
        self.current_raw_sample_position = 0; # index of oldest data in circular buffer (insert new data here)
        self.current_features = [] # processed data, ready to be sent to ML model
        self.previous_sampling_time = time.time()
        self.previous_notification_time = time.time()

    #################################################
    # AUDIO INPUT
    #################################################

    def initialize_audio(self, buffer_length = 16000): # buffer length is approx. the number of samples desired
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
                self.mic_sample_rate = rate
                break
            except ValueError:
                print("invalid sample rate")

        # circular buffer containing all current audio samples
        # (continuously feed data into this buffer, then
        # extract detection_period worth of data when ready)

        mic_samples_per_period = self.mic_sample_rate*self.desired_sample_period
        self.current_raw_samples = np.ones(mic_samples_per_period)

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
            #frames_per_buffer=2 * mic_samples_per_period,  # buffer at least this many samples
            frames_per_buffer=buffer_length,#2 * mic_samples_per_period,  # buffer at least this many samples
            input=True,
        )

        # frames_per_buffer doesn't exactly correlate with number of frames in the buffer?:
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

        current_time = time.time()
        time_since_last_sample = current_time - self.previous_sampling_time
        self.previous_sampling_time = current_time
        ideal_num_samples_since_last = self.mic_sample_rate*time_since_last_sample

        # we need to assemble a sample of this much data
        desired_num_samples = self.mic_sample_rate * self.desired_sample_period

        # get new data
        new_data_available = self.stream.get_read_available() # number of samples available
        new_data_bytes = self.stream.read(new_data_available) # actual data
        # new_data is a bytes object with 2-byte values
        # convert set of two bytes to float in [-1,1]
        new_data = [int.from_bytes(new_data_bytes[2*i:2*i+2], byteorder='little', signed=True)/32768.0
            for i in range(new_data_available)]
        new_data = np.array(new_data)

        if len(new_data) > 0 and len(new_data) < self.mic_sample_rate*self.desired_sample_period and len(new_data) < ideal_num_samples_since_last:
            print('not enough data; need to make buffer longer!')
        if len(new_data) > desired_num_samples:
            new_data = new_data[0:desired_num_samples]
            #print("new_data too long, cropping to {} (should equal {})".format(len(new_data), desired_num_samples))
        if len(new_data) == 0:
            print('data ok but make buffer shorter for faster response')
            return

        #print("new data length: {}".format(len(new_data)))
        #print("current_pos: {}".format(self.current_raw_sample_position))

        # put new data into circular buffer
        space_at_end_of_buffer = desired_num_samples - self.current_raw_sample_position
        #print(space_at_end_of_buffer)
        if space_at_end_of_buffer >= len(new_data):
            #print("filling without circular")
            self.current_raw_samples[self.current_raw_sample_position:self.current_raw_sample_position
                +len(new_data)] = new_data
            self.current_raw_sample_position = self.current_raw_sample_position + len(new_data)
        else:
            #print("filling with circular")
            self.current_raw_samples[self.current_raw_sample_position:] = new_data[0:space_at_end_of_buffer]
            self.current_raw_samples[0:len(new_data)-space_at_end_of_buffer] = new_data[space_at_end_of_buffer:]
            self.current_raw_sample_position = len(new_data)-space_at_end_of_buffer
        if self.current_raw_sample_position > desired_num_samples-1:
            self.current_raw_sample_position -= desired_num_samples

        #self.current_raw_samples = np.ones(self.mic_sample_rate*self.desired_sample_period)
        #self.current_raw_sample_position = 0; # index of oldest data in circular buffer (insert new data here)
        #self.current_features = [] # processed data, ready to be sent to ML model
        #self.desired_sample_rate = params.SAMPLE_RATE  # (Hz) sample rate we want
        #self.desired_sample_period = params.SAMPLE_PERIOD  # (s) total sampling time

        final_data = np.concatenate((self.current_raw_samples[self.current_raw_sample_position:], self.current_raw_samples[0:self.current_raw_sample_position]))
        #print(len(final_data))
        #print(final_data)

        # reformat sample
        resample_samples, _ = preprocess.resample_data(final_data, self.mic_sample_rate, params.SAMPLE_RATE)
        length_samples = preprocess.adjust_num_samples(resample_samples, params.SAMPLE_PERIOD*params.SAMPLE_RATE)
        self.current_features = preprocess.generate_something_like_mfcc(length_samples, 128)

        if current_time - self.previous_notification_time > 1: # notifications at least 1 second apart
            print("Running detection {:.3} times per second".format(1/time_since_last_sample))
            self.previous_notification_time = current_time

    def send_data_to_server(self):
        """Send data to server."""
        pass

    def run(self):
        self.sample_and_process_data()
        self.send_data_to_server()
        time.sleep(0.05)  # TODO remove

if __name__ == "__main__":
    chart = AudioChart()
    state = AudioSensorState()
    state.initialize_audio(buffer_length=8000)
    state.initialize_server()
    print("Running...")
    while True:
        state.run()
        #chart.update(state.current_features)
