import json
from itertools import chain
from math import log10
from pathlib import Path
from typing import Iterator, Union

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


def plot(
    path: Union[Path, str],
    data: DataFrame,
    filters: dict[str, str],
    x: str,
    y: str,
    hue: str,
    title: str,
) -> None:
    for key, value in filters.items():
        data = data.loc[data[key] == value]

    plt.figure()
    ax = lineplot(data=data, x=x, y=y, hue=hue, ci=68, style="fold")
    ax.set_title(title)
    ax.get_figure().savefig(path)
    plt.close()


def main(log_files: list[Path]) -> None:
    data = get_data(log_files)

    plot(
        "training_loss.png",
        title="training loss",
        data=data,
        filters={"data": "training"},
        x="epoch",
        y="log_loss",
        hue="model",
    )

    plot(
        "evaluation_f-score.png",
        title="evaluation f-score",
        data=data,
        filters={"data": "evaluation"},
        x="epoch",
        y="f_score",
        hue="model",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("log_files", nargs="+", type=Path)
    main(**vars(parser.parse_args()))
