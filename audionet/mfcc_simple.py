import os
import numpy as np
import scipy
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
import IPython.display as ipd
import matplotlib.pyplot as plt


def normalize_audio(audio):
    return audio / np.max(np.abs(audio))


def load_and_normalize_audio(audio_path):
    TRAIN_PATH = "../input/audio_train/"
    ipd.Audio(TRAIN_PATH + "a439d172.wav")

    sample_rate, audio = wavfile.read(TRAIN_PATH + "a439d172.wav")
    print("Sample rate: {0}Hz".format(sample_rate))
    print("Audio duration: {0}s".format(len(audio) / sample_rate))
    audio = normalize_audio(audio)
    plt.figure(figsize=(15, 4))
    plt.plot(np.linspace(0, len(audio) / sample_rate, num=len(audio)), audio)
    plt.grid(True)
    return sample_rate, audio


hop_size = 15  # ms
FFT_size = 2048


def frame_audio(audio, FFT_size=2048, hop_size=10, sample_rate=44100):
    # hop_size in ms

    audio = np.pad(audio, int(FFT_size / 2), mode="reflect")
    frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
    frame_num = int((len(audio) - FFT_size) / frame_len) + 1
    frames = np.zeros((frame_num, FFT_size))

    for n in range(frame_num):
        frames[n] = audio[n * frame_len : n * frame_len + FFT_size]

    return frames


sample_rate, audio = load_and_normalize_audio("")

audio_framed = frame_audio(
    audio,
    FFT_size=FFT_size,
    hop_size=hop_size,
    sample_rate=sample_rate,
)
print("Framed audio shape: {0}".format(audio_framed.shape))

# transform to frequency domain
window = get_window("hann", FFT_size, fftbins=True)
plt.figure(figsize=(15, 4))
plt.plot(window)
plt.grid(True)
