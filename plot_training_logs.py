from argparse import ArgumentParser
from itertools import chain
from json import loads
from math import log10
from pathlib import Path
from typing import Callable, Iterator, Union

from pandas import DataFrame, Series
from seaborn import lineplot
from matplotlib import pyplot as plt


DISPLAY = {
    "accuracy": "Accuracy (%)",
    "activity": "Sitting",
    "epoch": "Epoch",
    "evaluation": "Evaluation",
    "f_score": "F-Score",
    "log_loss": "Log Loss",
    "sleep": "Sleep",
    "training": "Training",
}


def read_log(file: Path) -> Iterator:
    """Read a log file from which each line is valid JSON."""
    with open(file) as f:
        return map(loads, f.readlines())


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
    axes = lineplot(data=data, x=x, y=y, hue=hue, ci=68, style="fold")
    axes.set_title(title)
    axes.set_xlabel(DISPLAY[x])
    axes.set_ylabel(DISPLAY[y])
    axes.get_figure().savefig(file)
    plt.close()


def get_best(
    data: DataFrame,
    filters: dict[str, str],
    variable: str,
    best: Callable[[Series], float] = max,
) -> float:
    """
    Get the best of a variable.

    Parameters
    ==========
    data: The dataframe containing the data to plot in long format.
    filters: A dictionary of column names to values, each of which must
        match within a row for that row of data to be plotted.
    variable: The variable to report the maximum value of.
    """

    for key, value in filters.items():
        data = data.loc[data[key] == value]

    aggregates = []

    for epoch in data["epoch"]:
        _data = data.loc[data["epoch"] == epoch]
        if len(_data) != 5: continue
        aggregates.append((_data[variable].mean(), _data[variable].std()))

    return best(aggregates)


def main(log_files: list[Path]) -> None:
    data = get_data(log_files)

    for label in "sleep", "activity":

        stage = "training"

        for variable in "log_loss",:
            plot(
                f"{label}_{stage}_{variable}.png",
                title=f"{DISPLAY[stage]} {DISPLAY[variable]} for {DISPLAY[label]} History Classification",
                data=data,
                filters={"data": stage, "label": label},
                x="epoch",
                y=variable,
                hue="model",
            )

        stage = "evaluation"

        for variable in "accuracy", "f_score":
            plot(
                f"{label}_{stage}_{variable}.png",
                title=f"{DISPLAY[variable]} for {DISPLAY[label]} History Classification",
                data=data,
                filters={"data": stage, "label": label},
                x="epoch",
                y=variable,
                hue="model",
            )

        for model in "dixonnet", "resnet18":

            accuracy = get_best(
                data=data,
                filters={"model": model, "data": stage, "label": label},
                variable="accuracy",
                best=max,
            )

            f_score = get_best(
                data=data,
                filters={"model": model, "data": stage, "label": label},
                variable="f_score",
                best=max,
            )

            print(f"\n{model}, {label}")
            print(f"    best accuracy = {accuracy[0]} ± {accuracy[1]}")
            print(f"    best f-score = {f_score[0]} ± {f_score[1]}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("log_files", nargs="+", type=Path)
    main(**vars(parser.parse_args()))
