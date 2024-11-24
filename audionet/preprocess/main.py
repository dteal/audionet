import scipy.signal
import numpy as np

def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform

def zero_crossing(waveform, sample_rate):
    # Compute the derivative of the waveform.
    derivative = scipy.signal.lfilter([1], [1, -1], waveform)
    indices = np.flatnonzero(np.diff(np.sign(derivative)))

    return indices / float(sample_rate)


def auto_correlation(waveform, sample_rate):
    # Compute the auto-correlation of the waveform.
    autocorrelation = scipy.signal.correlate(waveform, waveform, mode='full')
    autocorrelation = autocorrelation[len(waveform):]

    return autocorrelation / np.max(np.abs(autocorrelation))
