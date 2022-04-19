from csv import reader
from functools import partial
from itertools import chain
from os import walk
from pathlib import Path
from typing import *

from scipy.signal import butter, medfilt, sosfilt

import torch
from torch import Tensor, get_num_threads
from torch.multiprocessing import Pool
from torch.utils.data import Dataset, ConcatDataset, DataLoader

T = TypeVar("T")

SAMPLE_RATE = 20  # Hz


class CSVFile(Dataset):
    """
    Creates a PyTorch dataset from a CSV file. Columns specified by index are
    read and their values converted to floats. The dataset will consist of
    every possible consecutive slice of data having the requested window size.
    Other attributes of the data are assertained from the CSV filename.

    Parameters
    ==========
    file: The path specifying the CSV file to be read.
    label_fn: A function that takes a CSVFile and returns a binary label.
    columns (optional): The indices of the columns to be read. If 'None', all
        columns will be included. Default = 'None'.
    windows_size (optional): The number of rows to include in one sample
        of the dataset. Default = 1.
    median_filter_size: The kernel size of the median filter applied to each
        column of data. Must be odd. If 0, not applied. Default = 0.
    low_pass_frequency: The frequency (in Hertz) above which frequencies are
        removed via Butterworth filter. If 0, not applied. Default = 0.
    """

    def __init__(
        self,
        file: Path,
        *,
        label_fn: Callable[[Any], bool],
        columns: Optional[Iterable[int]] = None,
        window_size: int = 1,
        median_filter_size: int = 0,
        low_pass_frequency: float = 0,
        stride: int = 1,
    ):
        self.window_size = max(window_size, 1)
        self.stride = max(stride, 1)

        # If no columns specified, determine how many are in the file by
        # reading the first line.
        if columns is None:
            with open(file) as f:
                columns = range(len(f.readline()))

        # Load the columns and convert each item to a float.
        with open(file, newline="") as f:
            data = [[float(line[i]) for i in columns] for line in reader(f)]

        # We need at least one window size of data from the file.
        if len(data) < window_size:
            print(f"{len(data)} lines in {file}. Expected at least {window_size}.")

        # Median filter the data if specified.
        if median_filter_size > 0:
            data = medfilt(data, (median_filter_size, 1))

        # Butterworth (low pass) filter the data if specified.
        if low_pass_frequency > 0:
            butter_order = 10
            data = sosfilt(
                butter(
                    butter_order,
                    low_pass_frequency,
                    "lowpass",
                    fs=SAMPLE_RATE,
                    output="sos",
                ),
                data,
                axis=0,
            )

        # Convert the data from arbitrary units to g (i.e. gravity).
        self.data = torch.tensor(data, dtype=torch.float32)
        self.data = (self.data - 512) / 128  # convert to g

        # Some properties of the data are determined from the name
        # of the parent directory.
        parent = file.parent.name.upper()
        self.shift = parent[0] if parent[0] in ("D", "N") else None
        self.activity = parent[1] if parent[1] in ("B", "S") else None
        self.sleep = parent[2] if parent[2] in ("L", "R") else None

        assert (
            self.shift and self.activity and self.sleep
        ), f"Unexpected parent directory: {file.parent.name}"

        # Some properties of the data are determined from the file name.
        stem = file.stem.upper().split("_")
        self.session = stem[-1] if stem[-1] in ("M", "A") else None
        self.day = int(stem[-2][1]) if stem[-2][0] == "D" else None

        assert self.session and isinstance(
            self.day, int
        ), f"Unexpected file name: {file.name}"

        # Get the label using the label function.
        self.label = torch.tensor(label_fn(self), dtype=torch.float32)

    def __len__(self) -> int:
        """Get the number of possible windows in this dataset."""
        return max(int((len(self.data) - self.window_size) / self.stride + 1), 0)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Get a window from this dataset and the dataset's label."""
        start = index * self.stride
        item = self.data[start : start + self.window_size]
        return item.T, self.label

    # The following methods are useful label functions:

    def is_shift_day(self) -> bool:
        return self.shift == "D"

    def is_sleep_long(self) -> bool:
        return self.sleep == "L"

    def is_activity_broken(self) -> bool:
        return self.activity == "B"

    def is_session_morning(self) -> bool:
        return self.session == "M"


def find_files(
    dirs: Iterable[Union[Path, str]], suffix: Optional[str] = None,
) -> Iterator[Path]:
    """
    Gets the paths of files from multiple directories.

    Parameters
    ==========
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

    Parameters
    ==========
    s: The sequence to be split.
    f: The fraction of the elements of 's' in the first list, leaving the
        remaining '1 - f' elements in the second list.
    """

    n = int(f * len(s))
    return list(s[:n]), list(s[n:])


def k_fold_splits(s: Sequence[T], k: int = 5) -> list[tuple[list[T], list[T]]]:
    """
    Splits a sequence into 'k' parts, then returns each one paired with the
    remaining data. Useful for k-fold cross-validation.

    Parameters
    ==========
    s: The sequence to be split.
    k: The number of parts to split the data into.
    """

    p = [s[n::k] for n in range(k)]
    return [(list(chain(*p[:n], *p[n + 1 :])), list(p[n])) for n in range(k)]


def load_csv_files(files: Iterable[Path], **kwargs) -> list[CSVFile]:
    """
    Creates CSVFile datasets by processing multiple CSV files in parallel.

    Parameters
    ==========
    files: The paths of each CSV file.
    **kwargs: Keyword arguments are passed to the CSVFile constructor.
    """

    with Pool() as p:
        return p.map(partial(CSVFile, **kwargs), files)


def batch_data(
    datasets: Iterable[Dataset], batch_size: int, shuffle: bool = False,
) -> DataLoader:
    """
    Combines multiple datasets and creates a pytorch dataloader.

    Parameters
    ==========
    datasets: The pytorch datasets to combine.
    batch_size: How many samples per batch to load.
    shuffle (optional): Set to 'True' to have the data reshuffled at every
        epoch. Default = 'False'.
    """

    return DataLoader(
        ConcatDataset(datasets), batch_size, shuffle, num_workers=get_num_threads()
    )
