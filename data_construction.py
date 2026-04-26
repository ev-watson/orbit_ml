import os

from lightning.pytorch import LightningDataModule
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


@config.register('semlp')
class SEMLPDataset(Dataset):
    """
    Legacy flat dataset for the MLP-with-SE baseline. Each sample is a single timestep's
    feature vector, optionally a sliding window of timesteps if ``config.WINDOWED``.
    """
    def __init__(self, features):
        super().__init__()
        self.features = features  # [N, F] or [N, S, F]
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


@config.register('gnn')
class GraphDataset(Dataset):
    """
    Per-snapshot graph dataset for the message-passing GNN. Each sample is the full N-body
    state at a single timestamp, returned as a tuple ``(node_features, node_targets)`` where
    ``node_features`` has shape (B, F_in) and ``node_targets`` has shape (B, F_out). The graph
    topology is fully connected and shared across samples; the edge_index lives on the
    DataModule, not the dataset.

    :param features: np.ndarray of shape (T, B, F), the graph snapshot tensor produced by
        :func:`utils.data_processing.build_graph_snapshots`.
    """
    def __init__(self, features):
        super().__init__()
        self.features = features  # [T, B, F]
        self.input_slice = config.retrieve('model').input_slice
        self.output_slice = config.retrieve('model').output_slice

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        snap = self.features[idx]                      # [B, F]
        x = snap[..., self.input_slice]                # [B, F_in]
        y = snap[..., self.output_slice]               # [B, F_out]
        return x, y


class NNDataModule(LightningDataModule):
    def __init__(self, batch_size=None):
        super().__init__()
        self.dataset = config.retrieve('ds')
        self.batch_size = batch_size if batch_size else config.BATCH_SIZE
        self.features = np.load(config.retrieve('file'))[:config.NUM_SAMPLES:config.STEP]

        self.input_slice = config.retrieve('model').input_slice
        self.target_slice = config.retrieve('model').output_slice

        # Graph-network branch: per-snapshot multi-body data of shape [T, B, F]
        if config.TYPE == 'gnn':
            if self.features.ndim != 3:
                raise ValueError(
                    f"GraphDataset expects features with shape [T, B, F]; got {self.features.shape}. "
                    f"Run utils.data_processing.build_graph_snapshots() to produce {config.GRAPH_FILE}."
            )
            self.num_bodies = self.features.shape[1]
            self.src_index, self.dst_index = fully_connected_edges(self.num_bodies)
            # By default predict every body's acceleration except the Sun's (held fixed at index 0).
            self.predict_mask = torch.arange(1, self.num_bodies, dtype=torch.long)

            if config.ROTATIONAL_EQUIVARIANCE:
                self.features = self._rotation_augment(self.features)
        else:
            self.num_bodies = None
            self.src_index = None
            self.dst_index = None
            self.predict_mask = None

            # Legacy SEMLP / interp branches keep their original windowed + rotation hooks.
            if config.TYPE == 'semlp':
                if config.WINDOWED:
                    self.S = config.SEQUENCE_LENGTH
                    self.features = np.lib.stride_tricks.sliding_window_view(self.features,
                                                                             window_shape=int(self.S),
                                                                             axis=0)  # [N-S+1, F, S]
                    self.features = self.features.transpose(0, 2, 1)  # [N-S+1, S, F]
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
            self.features = np.concatenate((self.inputs, self.targets), axis=-1)

            if config.SCALER_FILE and not os.path.exists(config.SCALER_FILE):
                ScalerBundle(self.input_scaler, self.target_scaler).dump(config.SCALER_FILE)

        self.train_dataset = self.dataset(self.features[:train_size])
        self.val_dataset = self.dataset(self.features[train_size:train_size + val_size])
        self.test_dataset = self.dataset(self.features[train_size + val_size:])

    @staticmethod
    def _rotation_augment(graph_features):
        """
        Apply a single random SO(3) rotation to position, velocity, and acceleration columns of
        every body in every snapshot. Mass column is invariant. Provides rotational data
        augmentation in cartesian space; the network is not architecturally equivariant.

        :param graph_features: np.ndarray of shape (T, B, F) with column order
            [mass, x, y, z, vx, vy, vz, ax, ay, az].
        :return: np.ndarray of the same shape with positions, velocities, and accelerations
            rotated by a fresh random SO(3) matrix per snapshot.
        """
        out = graph_features.copy()
        T = out.shape[0]
        # uniformly sampled rotation matrices via QR of standard normal
        A = np.random.normal(size=(T, 3, 3))
        Q, _ = np.linalg.qr(A)
        # Right-multiply each (B, 3) vector field block by the per-snapshot rotation Q[t].T
        for blk in (slice(1, 4), slice(4, 7), slice(7, 10)):
            v = out[..., blk]                  # [T, B, 3]
            out[..., blk] = np.einsum('tbi,tij->tbj', v, Q)
        return out

    @staticmethod
    def rotation(data):
        """
        Legacy SEMLP rotational augmentation: scrubs theta and phi while leaving derived
        velocity / acceleration components untouched. Kept for backward compatibility with the
        flat per-step datasets.

        :param data: [..., F] shaped array where columns at index 2 and 3 are theta and phi.
        :return: data-like array with random theta and phi.
        """
        x = data.copy()
        x[..., 2] = np.random.uniform(0, np.pi, size=x.shape[:-1])
        x[..., 3] = np.random.uniform(-np.pi, np.pi, size=x.shape[:-1])

        return x

    def _graph_collate(self, samples):
        """
        Stack per-snapshot (V, Y) pairs into a graph-batch dict and target tensor. The edge
        topology is shared, so it lives on self and is broadcast at forward time.

        :param samples: list of length B of tuples (V, Y) from GraphDataset.
        :return: tuple (batch_dict, targets) where batch_dict carries 'nodes', 'src_index',
            'dst_index', and 'predict_mask'; targets has shape (B, N, F_out).
        """
        Vs = torch.stack([torch.as_tensor(v, dtype=torch.get_default_dtype()) for v, _ in samples], dim=0)
        Ys = torch.stack([torch.as_tensor(y, dtype=torch.get_default_dtype()) for _, y in samples], dim=0)
        batch = {
            'nodes': Vs,
            'src_index': self.src_index,
            'dst_index': self.dst_index,
            'predict_mask': self.predict_mask,
        }
        return batch, Ys

    def _make_loader(self, dataset, shuffle, batch_size=None):
        kwargs = dict(
            batch_size=batch_size if batch_size is not None else self.batch_size,
            shuffle=shuffle,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
        )
        if config.TYPE == 'gnn':
            kwargs['collate_fn'] = self._graph_collate
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._make_loader(self.test_dataset, shuffle=False, batch_size=1)
