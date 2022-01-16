import json
from math import log10
from pathlib import Path

from pandas import DataFrame
from seaborn import lineplot
import matplotlib.pyplot as plt


def read_log(file: Path) -> DataFrame:
    with open(file) as f:
        data = DataFrame(map(json.loads, f.readlines()))
    
    data["log_loss"] = list(map(log10, data["loss"]))
    return data


def plot_log(log_file: Path) -> None:
    data = read_log(log_file)

    plt.figure()
    ax = lineplot(data=data.loc[data["data"] == "training"], x="epoch", y="log_loss", hue="model")
    ax.set_title("training loss")
    ax.get_figure().savefig("training_loss.png")

    plt.figure()
    ax = lineplot(data=data.loc[data["data"] == "evaluation"], x="epoch", y="accuracy", hue="model")
    ax.set_title("evaluation accuracy")
    ax.get_figure().savefig("evaluation_accuracy.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=Path)
    plot_log(**vars(parser.parse_args()))
