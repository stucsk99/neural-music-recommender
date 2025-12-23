import os
import keras

SRC = "models/embedding_model_contrastiveone_ep.keras"
DST = "models/embedding_model_savedmodel"  # directory

# Load with Keras 3
model = keras.saving.load_model(SRC, compile=False)

# Export as TensorFlow SavedModel (most compatible for tf.keras)
model.export(DST)

print("Exported to:", DST)
