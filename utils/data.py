import csv
from os import walk
from itertools import chain
from resource import getrlimit, setrlimit, RLIMIT_NOFILE
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Union

import torch
from torch import Tensor, get_num_threads
from torch.utils.data import Dataset, ConcatDataset, DataLoader


class CSVFile(Dataset):
    def __init__(self, path: Path, window_size: int):
        self.window_size = window_size
        data = []

        with open(path, newline="") as f:
            for n, line in enumerate(csv.reader(f)):
                assert (
                    len(line) == 6
                ), f"Unexpected length {len(line)} on line {n} of {path}."
                data.append(list(map(float, line[2:5])))

        if len(data) < window_size:
            print(f"{len(data)} lines in {path}. Expected at least {window_size}.")

        self.data = torch.tensor(data, dtype=torch.float32)
        # self.data = self.data[1:] - self.data[:-1]
        self.data = (self.data - 512) / 512

        parent = path.parent.name.upper()
        self.shift = parent[0] if parent[0] in ("D", "N") else None
        self.active = parent[1] if parent[1] in ("B", "S") else None
        self.sleep = parent[2] if parent[2] in ("L", "R") else None

        assert (
            self.shift and self.active and self.sleep
        ), f"Unexpected parent directory: {path.parent.name}"

        stem = path.stem.upper().split("_")
        self.session = stem[-1] if stem[-1] in ("M", "A") else None
        self.day = int(stem[-2][1]) if stem[-2][0] == "D" else None

        assert self.session and isinstance(
            self.day, int
        ), f"Unexpected file name: {path.name}"

        self.label = torch.tensor(self.sleep == "L", dtype=torch.float32)

    def __len__(self) -> int:
        return max(len(self.data) - self.window_size + 1, 0)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        item = self.data[idx : idx + self.window_size]
        return item.T, self.label


def set_resource_limit(n: int):
    setrlimit(RLIMIT_NOFILE, (n, getrlimit(RLIMIT_NOFILE)[1]))


def find_files(
    dirs: Iterable[Union[Path, str]], suffix: Optional[str] = None,
) -> Iterator[Path]:
    return filter(
        lambda f: True if suffix is None else f.suffix == "." + suffix.strip("."),
        (Path(p) / f for d in dirs for p, _, fs in walk(d) for f in fs),
    )


def split(s: Sequence, f: float = 0.5) -> tuple[list, list]:
    n = int(f * len(s))
    return s[:n], s[n:]


def k_fold_splits(s: Sequence, k: int = 5) -> list[tuple[list, list]]:
    p = [s[n::k] for n in range(k)]
    return [(list(chain(*p[:n], *p[n + 1 :])), p[n]) for n in range(k)]


def load(
    files: Iterable[Path],
    window_size: int,
    batch_size: int,
    shuffle: bool = False,
    resource_limit: Optional[int] = None,
) -> DataLoader:
    if resource_limit is not None:
        set_resource_limit(resource_limit)

    with get_context("spawn").Pool() as p:
        data = p.map(partial(CSVFile, window_size=window_size), files)

    return DataLoader(
        ConcatDataset(data), batch_size, shuffle, num_workers=get_num_threads()
    )
