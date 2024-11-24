import tensorflow as tf

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            10, activation=tf.nn.relu, input_shape=(4,)
        ),  # input shape required
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(3),
    ]
)

save_frequency = 10
num_epochs = 10


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

    drone_files = get_wav_files("data/drone_audio_dataset/Binary_Drone_Audio/yes_drone")
    drone_examples = None
    for drone_file in drone_files:
        if drone_examples is None:
            drone_examples = input.wavfile_to_examples(drone_file)
        else:
            drone_examples = np.concatenate(
                (drone_examples, input.wavfile_to_examples(drone_file))
            )
    assert drone_examples is not None
    drone_labels = np.array([[0, 1]] * drone_examples.shape[0])

    no_drone_files = get_wav_files(
        "data/drone_audio_dataset/Binary_Drone_Audio/unknown"
    )
    no_drone_examples = None
    for no_drone_file in no_drone_files:
        if no_drone_examples is None:
            no_drone_examples = input.wavfile_to_examples(no_drone_file)
        else:
            no_drone_examples = np.concatenate(
                (no_drone_examples, input.wavfile_to_examples(no_drone_file))
            )
    assert no_drone_examples is not None
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
    model.compile(optimizer="adam", loss="mse")

    for epoch in range(num_epochs):
        for batch in range(num_batches):
            x = np.random.rand(batch_size, 4)
            y = np.random.rand(batch_size, 3)
            model.train_on_batch(x, y)

        if epoch % save_frequency == 0:
            model.save(f"model/model_{epoch}.h5")
            print(f"Model checkpoint saved at epoch {epoch}")
            print(f"Model saved to model/model_{epoch}.h5")

    return model
