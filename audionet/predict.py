import numpy as np
import librosa
import tensorflow as tf
from keras.models import load_model
import joblib

# Parameters
SAMPLE_RATE = 16000  # Sample rate for audio files
NUM_MFCC = 128  # Number of MFCC features to extract
MAX_PAD_LEN = 100  # Max length to pad/truncate MFCCs
MODEL_PATH = "/home/user/dev/drone/drone_detection_model.h5"
SCALER_PATH = "/home/user/dev/drone/scaler.pkl"

# Load the trained model
model = load_model(MODEL_PATH)

# Load the scaler
scaler = joblib.load(SCALER_PATH)


def predict_drone(audio_data):
    """
    Predicts whether the given 1-second audio data contains a drone sound.

    Parameters:
        audio_data (numpy.ndarray): 1D NumPy array containing audio samples at 16kHz.

    Returns:
        int: 1 if drone sound is detected, 0 otherwise.
    """
    # Check if audio data is 1 second long
    if len(audio_data) != SAMPLE_RATE:
        raise ValueError(
            f"Audio data must be 1 second long ({SAMPLE_RATE} samples at 16kHz)."
        )

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=NUM_MFCC)

    # Pad or truncate MFCCs to MAX_PAD_LEN
    if mfcc.shape[1] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_PAD_LEN]

    mfcc = mfcc.T  # Transpose to fit Conv1D input shape

    # Normalize the MFCC features using the loaded scaler
    mfcc_reshaped = mfcc.reshape(-1, NUM_MFCC)
    mfcc_normalized = scaler.transform(mfcc_reshaped)
    mfcc_normalized = mfcc_normalized.reshape(1, MAX_PAD_LEN, NUM_MFCC)

    # Predict using the loaded model
    prediction = model.predict(mfcc_normalized)
    predicted_label = int(prediction[0][0] >= 0.5)

    return predicted_label


# Example usage:
if __name__ == "__main__":
    # Load a 1-second audio sample from a WAV file
    import sys

    if len(sys.argv) != 2:
        print("Usage: python drone_detector.py path_to_1s_audio.wav")
        sys.exit(1)

    audio_file_path = sys.argv[1]

    # Load the audio file
    audio, sr = librosa.load(audio_file_path, sr=SAMPLE_RATE)

    # Ensure the audio is exactly 1 second long
    if len(audio) > SAMPLE_RATE:
        audio = audio[:SAMPLE_RATE]
    elif len(audio) < SAMPLE_RATE:
        # Pad with zeros if shorter than 1 second
        pad_width = SAMPLE_RATE - len(audio)
        audio = np.pad(audio, (0, pad_width), mode="constant")

    # Predict and print the result
    result = predict_drone(audio)
    if result == 1:
        print("Drone sound detected.")
    else:
        print("No drone sound detected.")
