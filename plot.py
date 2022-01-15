from math import log10
from os import walk
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union

from pandas import DataFrame
from seaborn import lineplot
import matplotlib.pyplot as plt

from utils.data import find_files


def read_history(file: Path) -> DataFrame:
    with open(file) as f:
        lines = f.readlines()

    data = DataFrame()

    for n, line in enumerate(lines):
        if "FINAL" in line:
            break

        obs = dict(map(str.split, map(str.strip, line.split("|"))))
        obs["Model"] = file.parents[2].name
        obs["Window"] = int(file.parents[1].name)
        obs["Fold"] = int(file.parents[0].name)
        obs["Type"] = file.stem

        if obs["Type"] == "eval" and "Epoch" not in obs:
            obs["Epoch"] = 10 * (n + 1)

        if "Acc" in obs:
            obs["Acc"] = obs["Acc"].strip("%")
        
        if "Loss" in obs:
            obs["LogLoss"] = log10(float(obs["Loss"]))

        data = data.append(obs, ignore_index=True)

    types = {
        "Epoch": int,
        "Loss": float,
        "F": float,
        "Acc": float,
        "TP": int,
        "FP": int,
        "TN": int,
        "FN": int,
        "LR": float,
    }

    return data.astype(
        {k: types[k] for k in data.keys() if k in types}
    ).convert_dtypes()


def plot(results_dir: Path) -> None:
    data = DataFrame()

    for file in filter(
        lambda p: p.stem in ["train", "eval"], find_files([results_dir], suffix="txt")
    ):
        data = data.append(read_history(file), ignore_index=True)

    # plt.figure()
    # train_loss = lineplot(data=data.loc[data["Type"] == "train"], x="Epoch", y="Loss", hue="Model")
    # train_loss.set_title("Training Loss")
    # train_loss.get_figure().savefig("train_loss.png")

    _data = DataFrame()

    for m in set(data["Model"]):
        for w in set(data["Window"]):
            line = data.loc[data["Model"] == m]
            line = line.loc[line["Window"] == w]
            line = line.loc[line["Type"] == "eval"]
            print(line)
            line = line.loc[line["Epoch"] == max(line["Epoch"])]
            _data.append(line, ignore_index=True)
    
    print(_data)

    # plt.figure()
    # train_loss = lineplot(x=x, y=y)
    # train_loss.set_title("test")
    # train_loss.get_figure().savefig("test.png")




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    plot(**vars(parser.parse_args()))
