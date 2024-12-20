"""Global parameters"""

##########################
# AUDIO SAMPLING CONSTANTS
##########################

# Input from anywhere (wave files or microphone) will be resampled
# to the following format before further processing:

# Make a numpy array of 32-bit floats (the amplitude of sound
# at each point) of length SAMPLE_RATE * SAMPLE_PERIOD.
# Each float should be a value between -1 and +1 (0 = no sound).
SAMPLE_RATE = 16000  # (Hz) sample rate
SAMPLE_PERIOD = 1  # (seconds) length of a recorded audio sample

# This numpy array will be used to create a spectrogram/MFCC/whatever.

##########################
# MODEL CONSTANTS
##########################

# Architectural constants.
NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
EMBEDDING_SIZE = 128  # Size of embedding layer.

# Hyperparameters used in feature and example generation.
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
EXAMPLE_HOP_SECONDS = 0.96  # with zero overlap.

# Hyperparameters used in training.
INIT_STDDEV = 0.01  # Standard deviation used to initialize weights.
LEARNING_RATE = 1e-4  # Learning rate for the Adam optimizer.
ADAM_EPSILON = 1e-8  # Epsilon for the Adam optimizer.

# Names of ops, tensors, and features.
INPUT_OP_NAME = "vggish/input_features"
INPUT_TENSOR_NAME = INPUT_OP_NAME + ":0"
OUTPUT_OP_NAME = "vggish/embedding"
OUTPUT_TENSOR_NAME = OUTPUT_OP_NAME + ":0"
AUDIO_EMBEDDING_FEATURE_NAME = "audio_embedding"
