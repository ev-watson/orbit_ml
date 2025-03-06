import os

import joblib
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
        self.features = features  # [N, S, F]
        self.input_dim = config.retrieve('model').input_dim
        self.output_dim = config.retrieve('model').output_dim
        self.windowed = config.WINDOWED

        self.x_slc = slice(None, self.input_dim)
        self.y_slc = slice(self.input_dim, self.input_dim + self.output_dim)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if self.windowed:
            x = self.features[idx, :, self.x_slc]
            y = self.features[idx, :, self.y_slc]
        else:
            x = self.features[idx, self.x_slc]
            y = self.features[idx, self.y_slc]
        return x, y


class NNDataModule(LightningDataModule):
    def __init__(self, batch_size=None):
        super().__init__()
        self.dataset = config.retrieve('ds')
        self.batch_size = batch_size if batch_size else config.BATCH_SIZE
        self.features = np.load(config.retrieve('file'))[:config.NUM_SAMPLES:config.STEP]

        self.input_slice = config.retrieve('model').input_slice
        self.target_slice = config.retrieve('model').output_slice

        if config.TYPE == 'gnn':
            if config.WINDOWED:
                self.S = config.SEQUENCE_LENGTH
                self.features = np.lib.stride_tricks.sliding_window_view(self.features,
                                                                         window_shape=int(self.S),
                                                                         axis=0)  # Shape [N-S+1, F, S]
                self.features = self.features.transpose(0, 2, 1)  # [[N-S+1, S, F]
                if config.ROTATIONAL_EQUIVARIANCE:
                    self.features = self.windowed_rotation(self.features)
            else:
                if config.ROTATIONAL_EQUIVARIANCE:
                    self.features = self.rotation(self.features)

        if config.MAC:  # MAC rejects float64
            self.features = self.features.astype(np.float32)

        total_len = self.features.shape[0]
        train_size = int(0.8 * total_len)
        val_size = int(0.1 * total_len)

        if config.SCALE:
            self.input_scaler = Scaler()
            self.target_scaler = Scaler()
            self.inputs = self.input_scaler.fit_transform(self.features[..., self.input_slice])
            self.targets = self.target_scaler.fit_transform(self.features[..., self.target_slice])
            self.features = np.column_stack((self.inputs, self.targets))

            if config.SCALER_FILE and not os.path.exists(config.SCALER_FILE):
                joblib.dump({
                    'input_scaler': self.input_scaler,
                    'target_scaler': self.target_scaler,
                }, config.SCALER_FILE)

        self.train_dataset = self.dataset(self.features[:train_size])
        self.val_dataset = self.dataset(self.features[train_size:train_size + val_size])
        self.test_dataset = self.dataset(self.features[train_size + val_size:])

    @staticmethod
    def rotation(data):
        x = data[..., :7]
        x[..., 2] = np.random.uniform(0, np.pi, size=x.shape[0])
        x[..., 3] = np.random.uniform(-np.pi, np.pi, size=x.shape[0])
        return get_movements(x)

    @staticmethod
    def windowed_rotation(windowed_data):
        x = windowed_data[..., :3]
        cart = sph_to_cart_windowed(x)  # [N, S, 3]
        R = Rot.random(x.shape[0]).as_matrix()
        rotated = np.einsum('nij,nsj->nsi', R, cart)  # [N, S, 3]
        sph = cart_to_sph_windowed(rotated)  # [N, S, 3]

        v = socfdw(sph)  # [N, S, 3]
        y = socfdw(v)

        return np.concatenate((sph, v, y), axis=-1)

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
