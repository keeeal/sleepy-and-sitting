import json
from itertools import chain
from math import log10
from pathlib import Path
from typing import Iterator

from pandas import DataFrame
from seaborn import lineplot
from matplotlib import pyplot as plt


def read_log(file: Path) -> Iterator:
    with open(file) as f:
        return map(json.loads, f.readlines())


def get_data(files: list[Path]) -> DataFrame:
    data = DataFrame(chain(*map(read_log, files)))
    data["log_loss"] = list(map(log10, data["loss"]))
    return data


def plot_logs(log_files: list[Path]) -> None:
    data = get_data(log_files)

    plt.figure()
    ax = lineplot(
        data=data.loc[data["data"] == "training"],
        x="epoch",
        y="log_loss",
        hue="model",
        ci=68,
        style="fold",
    )
    ax.set_title("training loss")
    ax.get_figure().savefig("training_loss.png")

    plt.figure()
    ax = lineplot(
        data=data.loc[data["data"] == "evaluation"],
        x="epoch",
        y="f_score",
        hue="model",
        ci=68,
        style="fold",
    )
    ax.set_title("evaluation f-score")
    ax.get_figure().savefig("evaluation_f-score.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("log_files", nargs="+", type=Path)
    plot_logs(**vars(parser.parse_args()))
