import os, csv
from functools import partial
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from typing import Sequence

import torch
from torch import get_num_threads
from torch.utils.data import Dataset, ConcatDataset, DataLoader


class CSVFile(Dataset):
    def __init__(self, path: Path, window_size: int):
        self.window_size = window_size
        self.data = []

        with open(path, newline="") as f:
            for n, line in enumerate(csv.reader(f)):
                if len(line) != 6:
                    print(f"Unexpected length {len(line)} on line {n} of {path}.")
                self.data.append(list(map(float, line[2:5])))

        if len(self.data) < window_size:
            print(f"{len(self.data)} lines in {path}. Expected at least {window_size}.")

        parent = path.parent.name.upper()
        self.shift = parent[0] if parent[0] in ("D", "N") else None
        self.active = parent[1] if parent[1] in ("B", "S") else None
        self.sleep = parent[2] if parent[2] in ("L", "R") else None

        if not self.shift or not self.active or not self.sleep:
            print(f"Unexpected parent directory: {path.parent.name}")

    def __len__(self) -> int:
        return max(len(self.data) - self.window_size + 1, 0)

    def __getitem__(self, idx: int):
        item = self.data[idx : idx + self.window_size]
        item = torch.tensor(item, dtype=torch.float32).T
        label = torch.tensor(self.sleep == "L", dtype=torch.float32)
        return item, label


def load(
    files: Sequence[Path] = None,
    dirs: Sequence[Path] = None,
    suffix: str = ".csv",
    window_size: int = 4096,
    batch_size: int = 128,
    shuffle: bool = False,
) -> DataLoader:
    files = [] if files is None else files
    dirs = [] if dirs is None else dirs

    # get all files with the correct suffix from dirs
    files = chain(
        files,
        filter(
            lambda f: f.suffix == "." + suffix.strip("."),
            (Path(p) / f for d in dirs for p, _, fs in os.walk(d) for f in fs),
        ),
    )

    # load data from files
    with Pool() as p:
        data = ConcatDataset(p.map(partial(CSVFile, window_size=window_size), files))

    return DataLoader(data, batch_size, shuffle, num_workers=get_num_threads())
