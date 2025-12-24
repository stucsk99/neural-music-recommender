import tensorflow as tf
import keras

SRC = "models/embedding_model_contrastiveone_ep.keras"   # adjust name
DST = "models/embedding_model_savedmodel"                # output folder

def l2_normalize(x):
    # Match your training behavior: normalize across embedding dimension
    return tf.math.l2_normalize(x, axis=1)

custom_objects = {
    "l2_normalize": l2_normalize,
    "Custom>l2_normalize": l2_normalize,
}

model = keras.saving.load_model(SRC, compile=False, custom_objects=custom_objects)

# Export to SavedModel
model.export(DST)

print("Exported SavedModel to:", DST)

