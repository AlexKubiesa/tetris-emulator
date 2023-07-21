import os
from pathlib import Path
import logging

import numpy as np


class RecordingDatabase:
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError()

        self._path = path
        self._ext = ".npy"

        with os.scandir(path) as it:
            self._count = (
                max(
                    (int(Path(file.path).stem) for file in it if file.is_file()),
                    default=-1,
                )
                + 1
            )

    @property
    def path(self):
        return self._path

    def __len__(self):
        return self._count

    def __getitem__(self, idx):
        file = os.path.join(self._path, str(idx) + self._ext)
        if not os.path.exists(file):
            raise IndexError()
        return np.load(file)

    def insert(self, item):
        file = os.path.join(self._path, str(self._count) + self._ext)
        np.save(file, item)
        self._count += 1

    # Doesn't preserve ordering
    def delete(self, idx):
        self.delete_batch([idx])

    # Doesn't preserve ordering
    def delete_batch(self, idxs):
        # Delete files
        for idx in idxs:
            file = os.path.join(self._path, str(idx) + self._ext)
            logging.debug("Deleting '%s'", file)
            os.remove(file)

        # Reindex remaining files
        old_idx = self._count
        new_idx = min(idxs) - 1

        while True:
            old_exists = False
            while not old_exists:
                old_idx -= 1
                old_file = os.path.join(self._path, str(old_idx) + self._ext)
                old_exists = os.path.exists(old_file)

            new_exists = True
            while new_exists:
                new_idx += 1
                new_file = os.path.join(self._path, str(new_idx) + self._ext)
                new_exists = os.path.exists(new_file)

            if old_idx <= new_idx:
                break

            logging.debug("Renaming '%s' to '%s'", old_file, new_file)
            os.rename(old_file, new_file)

        # Adjust count
        self._count -= len(idxs)
