import os, csv, random
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Optional, Union

import torch
from torch import Tensor, get_num_threads
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

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        data = self.data[idx : idx + self.window_size]
        item = torch.tensor(data, dtype=torch.float32).T
        label = torch.tensor(self.sleep == "L", dtype=torch.float32)
        return item, label


def find_files(
    dirs: Iterable[Union[Path, str]],
    suffix: Optional[str] = None,
    shuffle: bool = False,
) -> list[Path]:
    files = list(
        filter(
            lambda f: True if suffix is None else f.suffix == "." + suffix.strip("."),
            (Path(p) / f for d in dirs for p, _, fs in os.walk(d) for f in fs),
        )
    )

    if shuffle:
        random.shuffle(files)

    return files


def load(
    files: Iterable[Path],
    window_size: int = 4096,
    batch_size: int = 128,
    shuffle: bool = False,
) -> DataLoader:
    with Pool() as p:
        data: Dataset = ConcatDataset(
            p.map(partial(CSVFile, window_size=window_size), files)
        )

    return DataLoader(data, batch_size, shuffle, num_workers=get_num_threads())
