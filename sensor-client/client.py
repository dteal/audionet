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

import pyaudio
import time

def initialize_audio():
    """Set up audio stream."""
    print("Initializing audio...")

def initialize_server():
    """Connect to server to send data."""
    print("Testing server...")

def sample_and_process_data():
    """Get audio sample, process, run model."""
    pass

def send_data_to_server():
    """Send data to server."""
    pass

def run():
    time.sleep(1) # TODO remove
    sample_and_process_data()
    send_data_to_server()

if __name__=='__main__':
    initialize_audio()
    initialize_server()
    print("Running...")
    while True:
        run()
