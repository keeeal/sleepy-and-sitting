from ast import Call
from csv import reader
from functools import partial
from itertools import chain
from multiprocessing import get_context
from os import walk
from pathlib import Path
from resource import RLIMIT_NOFILE, getrlimit, setrlimit
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence, TypeVar, Union

import torch
from torch import Tensor, get_num_threads
from torch.utils.data import Dataset, ConcatDataset, DataLoader

T = TypeVar("T")


class CSVFile(Dataset):
    """
    Creates a pytorch dataset from a CSV file.

    Args:
        path: The location of the CSV file to be read.
        columns (optional): The indices of the columns to read. If 'None', all
            columns will be included. Default = 'None'.
        windows_size (optional): The number of lines to include in one sample
            from the file. Default = 1.
    """

    def __init__(
        self,
        path: Path,
        label_fn: Callable[[Any], bool],
        columns: Optional[Iterable[int]] = None,
        window_size: int = 1,
    ):
        self.window_size = window_size

        if columns is None:
            with open(path) as f:
                columns = range(len(f.readline()))

        with open(path, newline="") as f:
            data = [[float(line[i]) for i in columns] for line in reader(f)]

        if len(data) < window_size:
            print(f"{len(data)} lines in {path}. Expected at least {window_size}.")

        self.data = torch.tensor(data, dtype=torch.float32)
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

        self.label = torch.tensor(label_fn(self), dtype=torch.float32)

    def __len__(self) -> int:
        return max(len(self.data) - self.window_size + 1, 0)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        item = self.data[idx : idx + self.window_size]
        return item.T, self.label


def set_resource_limit(n: int):
    """Sets a new soft resource limits for the current process."""
    setrlimit(RLIMIT_NOFILE, (n, getrlimit(RLIMIT_NOFILE)[1]))


def find_files(
    dirs: Iterable[Union[Path, str]], suffix: Optional[str] = None,
) -> Iterator[Path]:
    """
    Gets the paths of all files found in multiple directories.

    args:
        dirs: The directories to search for files.
        suffix (optional): If provided, only return the files with this
            suffix. Default = "None".
    """
    return filter(
        lambda f: True if suffix is None else f.suffix == "." + suffix.strip("."),
        (Path(p) / f for d in dirs for p, _, fs in walk(d) for f in fs),
    )


def fraction_split(s: Sequence[T], f: float = 0.5) -> tuple[list[T], list[T]]:
    """
    Splits a sequence into two lists.

    args:
        s: The sequence to be split.
        f: The fraction of the elements of 's' in the first list, leaving the
            remaining '1 - f' elements in the second list.
    """
    n = int(f * len(s))
    return list(s[:n]), list(s[n:])


def k_fold_splits(s: Sequence[T], k: int = 5) -> list[tuple[list[T], list[T]]]:
    """
    Splits a sequence into 'k' parts, then returns each one paired with the
    remaining data.

    args:
        s: The sequence to be split.
        k: The number of parts to split the data into.
    """
    p = [s[n::k] for n in range(k)]
    return [(list(chain(*p[:n], *p[n + 1 :])), p[n]) for n in range(k)]


def load_csv_files(
    files: Iterable[Path],
    columns: Iterable[int],
    window_size: int,
    label_fn: Callable[[CSVFile], bool],
) -> list[CSVFile]:
    """
    Creates CSVFile datasets by processing CSV files in parallel.

    args:
        files: The paths of each CSV file.
        columns: The indices of the columns to read.
        window_size: The number of lines per sample.
    """
    with get_context("spawn").Pool() as p:
        return p.map(
            partial(
                CSVFile, columns=columns, window_size=window_size, label_fn=label_fn
            ),
            files,
        )


def batch_data(
    datasets: Iterable[Dataset], batch_size: int, shuffle: bool = False,
) -> DataLoader:
    """
    Combines multiple datasets and creates a pytorch dataloader.

    args:
        datasets: The pytorch datasets to combine.
        batch_size: How many samples per batch to load.
        shuffle (optional): Set to 'True' to have the data reshuffled at every
            epoch. Default = 'False'.
    """
    return DataLoader(
        ConcatDataset(datasets), batch_size, shuffle, num_workers=get_num_threads()
    )
