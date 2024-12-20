import socketio
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import numpy as np
import librosa
from keras.models import load_model
import joblib
import audionet.client as client
from predict import predict_drone

# Parameters
SAMPLE_RATE = 16000  # Sample rate for audio files
NUM_MFCC = 128  # Number of MFCC features to extract
MAX_PAD_LEN = 100  # Max length to pad/truncate MFCCs
MODEL_PATH = "drone_detection_model.h5"
SCALER_PATH = "scaler.pkl"

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


# Create a Socket.IO client
sio = socketio.Client(
    reconnection=True,
    reconnection_attempts=3,
    reconnection_delay=1,
)


@sio.event
def connect():
    logger.info("Connected to server!")
    logger.info(f"Transport method: {sio.transport()}")
    logger.info(f"SID: {sio.get_sid()}")


@sio.event
def connect_error(data):
    logger.error(f"Connection failed: {data}")


@sio.event
def disconnect():
    logger.info("Disconnected from server")


buffer = []


async def test_server():
    ass = client.AudioSensorState()
    ass.initialize_audio()
    await asyncio.sleep(2)
    SERVER_URL = "https://e82e-98-97-27-170.ngrok-free.app"

    try:
        logger.info(f"Attempting to connect to server at {SERVER_URL}...")

        # Connect with both WebSocket and polling as fallback
        sio.connect(
            SERVER_URL,
            transports=["websocket"],
        )

        while asyncio.get_event_loop().is_running():
            ass.run()
            prediction = predict_drone(ass.current_features)
            sio.emit("message", {"id": 0, "probability": prediction})
            # Keep the connection alive for a while to receive data
            await asyncio.sleep(2)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(f"Error type: {type(e)}")
    finally:
        if sio.connected:
            sio.disconnect()


if __name__ == "__main__":
    print("Socket.IO Test Client Starting...")
    print("=" * 50)
    asyncio.run(test_server())
