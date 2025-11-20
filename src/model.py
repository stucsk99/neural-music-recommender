import tensorflow as tf
from tensorflow.keras import layers, models


from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
def l2_normalize(t):
    """Top-level function for Lambda layer to allow safe serialization."""
    return tf.math.l2_normalize(t, axis=-1)


def build_embedding_model(input_shape=(128, 129, 1), embedding_dim=128):
    """
    Simple CNN that maps mel-spectrogram chunks to an embedding vector.
    """
    inputs = layers.Input(shape=input_shape)

    # --- Convolutional feature extractor ---
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # --- Embedding head ---
    x = layers.Dense(embedding_dim)(x)
    x = layers.Lambda(l2_normalize, name="l2_normalize")(x)

    model = models.Model(inputs, x, name="embedding_model")
    return model
