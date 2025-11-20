import tensorflow as tf
import numpy as np
from model import build_embedding_model
from data_loader import SpectrogramDataset

# --- Hyperparameters ---
batch_size = 8
embedding_dim = 128
temperature = 0.1
epochs = 5

# --- Dataset ---
ds = SpectrogramDataset("E:/song_project/spectrograms", batch_size=batch_size)

# --- Model ---
embedding_model = build_embedding_model(input_shape=(128, 129, 1), embedding_dim=embedding_dim)
optimizer = tf.keras.optimizers.Adam(1e-4)

# --- Simple augmentation function ---
@tf.function
def augment(x):
    """Tiny time-frequency noise."""
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.05)
    return tf.clip_by_value(x + noise, -80.0, 0.0)  # mel dB range usually -80–0

# --- Contrastive loss ---
def contrastive_loss(z_i, z_j, temperature=0.1):
    """
    Compute InfoNCE loss between two batches of embeddings.
    z_i, z_j: (batch_size, embedding_dim)
    """
    z_i = tf.math.l2_normalize(z_i, axis=1)
    z_j = tf.math.l2_normalize(z_j, axis=1)

    logits = tf.matmul(z_i, z_j, transpose_b=True) / temperature
    labels = tf.range(tf.shape(logits)[0])
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return tf.reduce_mean(loss)

# --- Training loop ---
for epoch in range(epochs):
    epoch_loss = []
    for step in range(len(ds)):
        X, _ = ds[step]  # (batch_size, 128, 129, 1)

        # two augmented "views" of the same batch
        X_i = augment(X)
        X_j = augment(X)

        
        with tf.GradientTape() as tape:
            z_i = embedding_model(X_i, training=True)
            z_j = embedding_model(X_j, training=True)
            loss = contrastive_loss(z_i, z_j, temperature)
        grads = tape.gradient(loss, embedding_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, embedding_model.trainable_variables))
        epoch_loss.append(loss.numpy())

        if step % 10 == 0:
            print(f"Epoch {epoch+1} Step {step}/{len(ds)} Loss={loss.numpy():.4f}")

    print(f"Epoch {epoch+1} mean loss: {np.mean(epoch_loss):.4f}")

embedding_model.save("E:/song_project/models/embedding_model_contrastive.keras")
print("✅ Saved trained embedding model.")
