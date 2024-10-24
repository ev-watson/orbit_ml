import os

import numpy as np
from lightning.pytorch import LightningDataModule
from scipy.spatial.transform import Rotation as Rot
from torch.utils.data import DataLoader, Dataset

from models import *

torch.set_default_dtype(torch.float64) if not config.MAC else torch.set_default_dtype(torch.float32)


@config.register('interp')
class InterpolationDataset(Dataset):
    def __init__(self, features):
        super().__init__()
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx, :-1]
        y = self.features[idx, -1]
        return x, y


@config.register('gnn')
class GNNDataset(Dataset):
    def __init__(self, features):
        super().__init__()
        self.features = features    # [N, S, F]
        self.input_slice = config.retrieve('model').input_slice
        self.target_slice = config.retrieve('model').output_slice

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        x = self.features[idx, :, self.input_slice]
        y = self.features[idx, :, self.target_slice]
        return x, y


class NNDataModule(LightningDataModule):
    def __init__(self, batch_size=None):
        super().__init__()
        self.dataset = config.retrieve('ds')
        self.batch_size = batch_size if batch_size else config.BATCH_SIZE
        self.features = load_np(config.retrieve('file'), nrows=config.NUM_SAMPLES, step=config.STEP)

        if config.MAC:  # MAC rejects float64
            self.features = self.features.astype(np.float32)

        self.input_slice = config.retrieve('model').input_slice
        self.target_slice = config.retrieve('model').output_slice

        if config.TYPE == 'gnn':
            self.S = config.SEQUENCE_LENGTH
            self.features = np.lib.stride_tricks.sliding_window_view(self.features,
                                                                     window_shape=self.S,
                                                                     axis=0)  # Shape [N-S+1, F, S]
            self.features = self.features.transpose(0, 2, 1)    # [[N-S+1, S, F]
            if config.ROTATIONAL_EQUIVARIANCE:
                self.features = self.windowed_rotation(self.features)

        total_windows = len(self.features)
        train_size = int(0.8 * total_windows)
        val_size = int(0.1 * total_windows)

        if config.SCALE:
            self.input_scaler = Scaler()
            self.target_scaler = Scaler()
            self.inputs = self.input_scaler.fit_transform(self.features[:, :, self.input_slice])
            self.targets = self.target_scaler.fit_transform(self.features[:, :, self.target_slice])

            self.train_inputs = self.inputs[:train_size]
            self.val_inputs = self.inputs[train_size:train_size + val_size]
            self.test_inputs = self.inputs[train_size + val_size:]

            self.train_targets = self.targets[:train_size]
            self.val_targets = self.targets[train_size:train_size + val_size]
            self.test_targets = self.targets[train_size + val_size:]

            self.train_features = np.concatenate((self.train_inputs, self.train_targets), axis=-1)
            self.val_features = np.concatenate((self.val_inputs, self.val_targets), axis=-1)
            self.test_features = np.concatenate((self.test_inputs, self.test_targets), axis=-1)

            if config.SCALER_FILE and not os.path.exists(config.SCALER_FILE):
                joblib.dump({
                    'input_scaler': self.input_scaler,
                    'target_scaler': self.target_scaler,
                }, config.SCALER_FILE)

        else:
            self.train_features = self.features[:train_size]
            self.val_features = self.features[train_size:train_size + val_size]
            self.test_features = self.features[train_size + val_size:]

        self.train_dataset = self.dataset(self.train_features)
        self.val_dataset = self.dataset(self.val_features)
        self.test_dataset = self.dataset(self.test_features)

    @staticmethod
    def windowed_rotation(windowed_data):
        N, S, F = windowed_data.shape
        results = []
        chunk_size = 500
        num_chunks = (N + chunk_size - 1) // chunk_size
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, N)
            chunk = windowed_data[start:end]  # [chunk_size, S, F]
            x = chunk[..., :3]
            cart = sph_to_cart_windowed(x)  # [chunk_size, S, 3]
            R = Rot.random(x.shape[0]).as_matrix()  # [chunk_size, 3, 3]
            rotated = np.einsum('nij,nsj->nsi', R, cart)  # [chunk_size, S, 3]
            sph = cart_to_sph_windowed(rotated)

            v = socfdw(sph, dtype=windowed_data.dtype)
            y = socfdw(v, dtype=windowed_data.dtype)
            concatenated = np.concatenate((sph, v, y), axis=-1, dtype=windowed_data.dtype)

            # Append the processed chunk to the results list
            results.append(concatenated)
            print(f"Processed chunk {i + 1}/{num_chunks}: samples {start} to {end}")

        final_result = np.concatenate(results, axis=0, dtype=windowed_data.dtype)
        return final_result

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=config.NUM_WORKERS,
                          pin_memory=config.PIN_MEMORY)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=config.NUM_WORKERS,
                          pin_memory=config.PIN_MEMORY)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=config.NUM_WORKERS,
                          pin_memory=config.PIN_MEMORY)
