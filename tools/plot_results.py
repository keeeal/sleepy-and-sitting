from math import log10
from os import walk
from pathlib import Path
from typing import Iterable, Optional, Union

from pandas import DataFrame
from seaborn import lineplot
import matplotlib.pyplot as plt


def find_files(
    dirs: Iterable[Union[Path, str]], suffix: Optional[str] = None,
) -> list[Path]:
    return filter(
        lambda f: True if suffix is None else f.suffix == "." + suffix.strip("."),
        (Path(p) / f for d in dirs for p, _, fs in walk(d) for f in fs),
    )


def read_history(file: Path) -> DataFrame:
    with open(file) as f:
        lines = f.readlines()

    data = DataFrame()

    for n, line in enumerate(lines):
        if "FINAL" in line:
            break

        obs = dict(map(str.split, map(str.strip, line.split("|"))))
        obs["Model"] = file.parents[1].name
        obs["Fold"] = file.parents[0].name
        obs["Type"] = file.stem

        if obs["Type"] == "eval" and "Epoch" not in obs:
            obs["Epoch"] = 10 * (n + 1)

        if "Acc" in obs:
            obs["Acc"] = obs["Acc"].strip("%")

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


def plot_results(results_dir: Path) -> None:
    data = DataFrame()

    for file in filter(
        lambda p: p.stem in ["train", "eval"], find_files([results_dir], suffix="txt")
    ):
        data = data.append(read_history(file), ignore_index=True)

    data["Loss"] = list(map(log10, data["Loss"]))

    plt.figure()
    train_loss = lineplot(data=data.loc[data["Type"] == "train"], x="Epoch", y="Loss", hue="Model")
    train_loss.set_title("Training Loss")
    train_loss.get_figure().savefig("train_loss.png")

    #plt.figure()
    #eval_acc = lineplot(data=data.loc[data["Type"] == "eval"], x="Epoch", y="Acc", hue="Model")
    #eval_acc.set_title("Evaluation Accuracy")
    #eval_acc.get_figure().savefig("eval_acc.png")

    plt.figure()
    eval_f = lineplot(data=data.loc[data["Type"] == "eval"], x="Epoch", y="F", hue="Model")
    eval_f.set_title("Evaluation F-Score")
    eval_f.get_figure().savefig("eval_f.png")

    #plt.figure()
    #eval_f = lineplot(data=data.loc[data["Type"] == "eval"], x="Epoch", y="FP", hue="Model")
    #eval_f.set_title("TEST")
    #eval_f.get_figure().savefig("eval_f.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--results-dir", type=Path)
    plot_results(**vars(parser.parse_args()))
