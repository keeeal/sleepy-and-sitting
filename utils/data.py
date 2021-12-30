import enum
import os, csv
from pathlib import Path

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
                if line[-1] == "DRIVING":
                    self.data.append(list(map(float, line[2:5])))

        if len(self.data) < window_size:
            print(f"{len(self.data)} lines in {path}. Expected at least {window_size}.")

        # DBL = Active9H, DBR = Active5H, DSL = Sedentary9H, DSR = Sedentary5H
        if path.parent.name in ("DBL", "DSL"):
            self.label = 1
        elif path.parent.name in ("DBR", "DSR"):
            self.label = 0
        else:
            print(f"Unexpected parent directory: {path.parent.name}")

    def __len__(self) -> int:
        return max(len(self.data) - self.window_size + 1, 0)

    def __getitem__(self, idx: int):
        item = self.data[idx : idx + self.window_size]
        item = torch.tensor(item, dtype=torch.float32)
        label = torch.tensor(self.label, dtype=torch.float32)
        return item, label


def load_csv_files(
    dirs: list[Path],
    window_size: int = 4096,
    batch_size: int = 124,
    shuffle: bool = False,
) -> DataLoader:
    data = (Path(p) / f for d in dirs for p, _, fs in os.walk(d) for f in fs)
    data = filter(lambda f: f.suffix == ".csv", data)
    data = ConcatDataset(CSVFile(path, window_size) for path in data)
    return DataLoader(data, batch_size, shuffle, num_workers=get_num_threads())
