r"""A simple demonstration of running VGGish in training mode.

This is intended as a toy example that demonstrates how to use the VGGish model
definition within a larger model that adds more layers on top, and then train
the larger model. If you let VGGish train as well, then this allows you to
fine-tune the VGGish model parameters for your application. If you don't let
VGGish train, then you use VGGish as a feature extractor for the layers above
it.

For this toy task, we are training a classifier to distinguish between three
classes: sine waves, constant signals, and white noise. We generate synthetic
waveforms from each of these classes, convert into shuffled batches of log mel
spectrogram examples with associated labels, and feed the batches into a model
that includes VGGish at the bottom and a couple of additional layers on top. We
also plumb in labels that are associated with the examples, which feed a label
loss used for training.

Usage:
  # Run training for 100 steps using a model checkpoint in the default
  # location (vggish_model.ckpt in the current directory). Allow VGGish
  # to get fine-tuned.
  $ python vggish_train_demo.py --num_batches 100

  # Same as before but run for fewer steps and don't change VGGish parameters
  # and use a checkpoint in a different location
  $ python vggish_train_demo.py --num_batches 50 \
                                --train_vggish=False \
                                --checkpoint /path/to/model/checkpoint
"""

from __future__ import print_function

from random import shuffle

import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim as slim

import audionet.preprocess.input as input
import audionet.params as params
import audionet.preprocess.slim as audio_slim
from pathlib import Path
from typing import List


def get_wav_files(directory: str) -> List[str]:
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


flags = tf.app.flags

flags.DEFINE_integer(
    "num_batches",
    30,
    "Number of batches of examples to feed into the model. Each batch is of "
    "variable size and contains shuffled examples of each class of audio.",
)

flags.DEFINE_boolean(
    "train_vggish",
    True,
    "If True, allow VGGish parameters to change during training, thus "
    "fine-tuning VGGish. If False, VGGish parameters are fixed, thus using "
    "VGGish as a fixed feature extractor.",
)

flags.DEFINE_string(
    "checkpoint", "vggish_model.ckpt", "Path to the VGGish checkpoint file."
)

FLAGS = flags.FLAGS

_NUM_CLASSES = 2

save_frequency = 10


tf.disable_eager_execution()


def _get_examples_batch():
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
    sr = 44100  # Sampling rate.

    drone_files = get_wav_files("data/drone_audio_dataset/Binary_Drone_Audio/yes_drone")
    drone_examples = np.array([])
    for drone_file in drone_files:
        drone_examples = np.concatenate(
            (drone_examples, input.waveform_to_examples(drone_file, sr))
        )
    drone_labels = np.array([[0, 1]] * drone_examples.shape[0])

    no_drone_files = get_wav_files(
        "data/drone_audio_dataset/Binary_Drone_Audio/unknown"
    )
    no_drone_examples = np.array([])
    for no_drone_file in no_drone_files:
        no_drone_examples = np.concatenate(
            (no_drone_examples, input.waveform_to_examples(no_drone_file, sr))
        )
    no_drone_labels = np.array([[1, 0]] * no_drone_examples.shape[0])

    # Shuffle (example, label) pairs across all classes.
    all_examples = np.concatenate((drone_examples, no_drone_examples))
    all_labels = np.concatenate((drone_labels, no_drone_labels))
    labeled_examples = list(zip(all_examples, all_labels))
    shuffle(labeled_examples)

    # Separate and return the features and labels.
    features = [example for (example, _) in labeled_examples]
    labels = [label for (_, label) in labeled_examples]
    return (features, labels)


def main(_):
    with tf.Graph().as_default(), tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=5)
        # Define VGGish.
        embeddings = audio_slim.define_slim(training=FLAGS.train_vggish)

        # Define a shallow classification model and associated training ops on top
        # of VGGish.
        with tf.variable_scope("mymodel"):
            # Add a fully connected layer with 100 units. Add an activation function
            # to the embeddings since they are pre-activation.
            num_units = 100
            fc = slim.fully_connected(tf.nn.relu(embeddings), num_units)

            # Add a classifier layer at the end, consisting of parallel logistic
            # classifiers, one per class. This allows for multi-class tasks.
            logits = slim.fully_connected(
                fc, _NUM_CLASSES, activation_fn=None, scope="logits"
            )
            tf.sigmoid(logits, name="prediction")

            # Add training ops.
            with tf.variable_scope("train"):
                global_step = tf.train.create_global_step()

                # Labels are assumed to be fed as a batch multi-hot vectors, with
                # a 1 in the position of each positive class label, and 0 elsewhere.
                labels_input = tf.placeholder(
                    tf.float32, shape=(None, _NUM_CLASSES), name="labels"
                )

                # Cross-entropy label loss.
                xent = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits, labels=labels_input, name="xent"
                )
                loss = tf.reduce_mean(xent, name="loss_op")
                tf.summary.scalar("loss", loss)

                # We use the same optimizer and hyperparameters as used to train VGGish.
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=params.LEARNING_RATE,
                    epsilon=params.ADAM_EPSILON,
                )
                train_op = optimizer.minimize(loss, global_step=global_step)

        # Initialize all variables in the model, and then load the pre-trained
        # checkpoint.
        sess.run(tf.global_variables_initializer())
        # audio_slim.load_slim_checkpoint(sess, FLAGS.checkpoint)
        audio_slim.initialize_uniform_weights(sess)

        # The training loop.
        features_input = sess.graph.get_tensor_by_name(params.INPUT_TENSOR_NAME)
        for epoch in range(FLAGS.num_batches):
            (features, labels) = _get_examples_batch()
            [num_steps, loss_value, _] = sess.run(
                [global_step, loss, train_op],
                feed_dict={features_input: features, labels_input: labels},
            )
            print("Step %d: loss %g" % (num_steps, loss_value))
            # Save every N epochs
            if epoch % save_frequency == 0:
                save_path = saver.save(
                    sess, "path/to/checkpoints/model.ckpt", global_step=epoch
                )
                print(f"Model checkpoint saved at epoch {epoch}: {save_path}")


if __name__ == "__main__":
    tf.app.run()
