import os
from pathlib import Path


class RecordingFolder:
    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            os.mkdir(path)
        files = os.listdir(path)
        highest_index = max((int(Path(file).stem) for file in files), default=-1)
        self.file_index = highest_index

    def next_file(self) -> str:
        self.file_index += 1
        return os.path.join(self.path, str(self.file_index))
