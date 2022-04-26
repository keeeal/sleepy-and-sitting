from argparse import ArgumentParser
import json
from itertools import chain
from math import log10
from pathlib import Path
from typing import Iterator, Union

from pandas import DataFrame
from seaborn import lineplot
from matplotlib import pyplot as plt


def read_log(file: Path) -> Iterator:
    """Read a log file from which each line is valid JSON."""
    with open(file) as f:
        return map(json.loads, f.readlines())


def get_data(files: list[Path]) -> DataFrame:
    """Read multiple log files and return as a pandas dataframe."""
    data = DataFrame(chain(*map(read_log, files)))
    data["log_loss"] = list(map(log10, data["loss"]))
    return data


def plot(
    file: Union[Path, str],
    data: DataFrame,
    filters: dict[str, str],
    x: str,
    y: str,
    hue: str,
    title: str,
) -> None:
    """
    Plot some data and save the image to file.

    Parameters
    ==========
    file: The path to where the plot will be saved.
    data: The dataframe containing the data to plot in long format.
    filters: A dictionary of column names to values, each of which must
        match within a row for that row of data to be plotted.
    x: The name of the column to be plotted on the X axis.
    y: The name of the column to be plotted on the Y axis.
    hue: The name of the column to be distinguished by colour.
    title: The title of the plot.
    """

    for key, value in filters.items():
        data = data.loc[data[key] == value]

    plt.figure()
    ax = lineplot(data=data, x=x, y=y, hue=hue, ci=68, style="fold")
    ax.set_title(title)
    ax.get_figure().savefig(file)
    plt.close()


def main(log_files: list[Path]) -> None:
    data = get_data(log_files)

    plot(
        "sleep_training_loss.png",
        title="sleep training loss",
        data=data,
        filters={"data": "training", "label": "sleep"},
        x="epoch",
        y="log_loss",
        hue="model",
    )

    plot(
        "sleep_evaluation_f-score.png",
        title="sleep evaluation f-score",
        data=data,
        filters={"data": "evaluation", "label": "sleep"},
        x="epoch",
        y="f_score",
        hue="model",
    )

    plot(
        "activity_training_loss.png",
        title="activity training loss",
        data=data,
        filters={"data": "training", "label": "activity"},
        x="epoch",
        y="log_loss",
        hue="model",
    )

    plot(
        "activity_evaluation_f-score.png",
        title="activity evaluation f-score",
        data=data,
        filters={"data": "evaluation", "label": "activity"},
        x="epoch",
        y="f_score",
        hue="model",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("log_files", nargs="+", type=Path)
    main(**vars(parser.parse_args()))
