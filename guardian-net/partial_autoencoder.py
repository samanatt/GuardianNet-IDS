import tensorflow as tf


def ae_factory(in_shape, hidden_size, activation):
    autoencoder = tf.keras.models.Sequential()
    autoencoder.add(tf.keras.layers.Input(shape=(in_shape,)))
    for size in hidden_size:
        autoencoder.add(tf.keras.layers.Dense(size, activation=activation))
    return autoencoder


def partial_ae_factory(in_shape, hidden_size, activation):
    input_img = tf.keras.layers.Input(shape=(in_shape,))
    encoded = tf.keras.layers.Dense(hidden_size, activation=activation,
                                    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
                                    bias_initializer=tf.keras.initializers.Zeros())(input_img)
    decoded = tf.keras.layers.Dense(in_shape, activation=activation,
                                    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
                                    bias_initializer=tf.keras.initializers.Zeros())(encoded)

    autoencoder = tf.keras.models.Model(inputs=input_img, outputs=decoded)
    encoder = tf.keras.models.Model(inputs=input_img, outputs=encoded)

    return autoencoder, encoder


def partial_lstm_ae_factory(in_shape, hidden_size, activation):
    input_img = tf.keras.layers.Input(shape=(1, in_shape))
    encoded = tf.keras.layers.LSTM(hidden_size, activation=activation, return_sequences=True,
                                   kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
                                   bias_initializer=tf.keras.initializers.Zeros())(input_img)
    decoded = tf.keras.layers.LSTM(in_shape, activation=activation, return_sequences=True,
                                   kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
                                   bias_initializer=tf.keras.initializers.Zeros())(encoded)

    autoencoder = tf.keras.models.Model(inputs=input_img, outputs=decoded)
    encoder = tf.keras.models.Model(inputs=input_img, outputs=encoded)

    return autoencoder, encoder


# def ae_factory(in_shape, hidden_size, activation):
#     autoencoder = Sequential()
#     autoencoder.add(Input(shape=(in_shape,)))
#     # ENCODER
#     for size in hidden_size:
#         autoencoder.add(Dense(size, activation=activation))
#         autoencoder.add(BatchNormalization())
#
#     # DECODER
#     if len(hidden_size) >= 1:
#         hidden_size.pop()
#         rev_hidden_size = hidden_size[::-1]
#         if len(rev_hidden_size) > 0:
#             for size in rev_hidden_size:
#                 autoencoder.add(Dense(size, activation=activation))
#
#     autoencoder.add(Dense(in_shape, activation=activation))
#     return autoencoder


if __name__ == "__main__":
    h = [128, 64, 32]
    ae = ae_factory(in_shape=118, hidden_size=h, activation='tanh')
    ae.summary()
