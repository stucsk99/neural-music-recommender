import os
import numpy as np
import tensorflow as tf

class SpectrogramDataset(tf.keras.utils.Sequence):
    """
    Loads .npz mel spectrograms from disk and performs chunking on the fly.
    """

    def __init__(self, data_dir, batch_size=8, chunk_length=129, n_mels=128, shuffle=True):
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]
        self.batch_size = batch_size
        self.chunk_length = chunk_length
        self.n_mels = n_mels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.files) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.files)

    def __getitem__(self, idx):
        batch_files = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = []

        for file_path in batch_files:
            data = np.load(file_path)
            mel = data["mel"]

            # Automatically chunk if 2D
            if mel.ndim == 2:
                mel = self.chunk_spectrogram(mel)

            # Randomly pick one chunk
            mel = mel[np.random.randint(0, mel.shape[0])]
            X_batch.append(mel)

        X_batch = np.expand_dims(np.stack(X_batch), -1)
        return X_batch, X_batch

    def chunk_spectrogram(self, mel_db):
        """
        Divide mel spectrogram (n_mels, n_frames) into time chunks of equal length.
        """
        n_frames = mel_db.shape[1]
        chunks = []
        for start in range(0, n_frames, self.chunk_length):
            end = start + self.chunk_length
            if end <= n_frames:
                chunks.append(mel_db[:, start:end])
        return np.stack(chunks)
