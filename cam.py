

from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from models.dixonnet import DixonNet
from models.resnet import resnet18, resnet34, resnet50
from utils.data import CSVFile, batch_data, find_files, load_csv_files
from utils.misc import set_resource_limit

def plot_coloured_line(x, y, dydx):
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html

    x = np.array(x)
    y = np.array(y)
    dydx = np.array(dydx)

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(dydx.min(), dydx.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(2)
    line = axs[0].add_collection(lc)
    fig.colorbar(line, ax=axs[0])

    axs[0].set_xlim(x.min(), x.max())
    axs[0].set_ylim(y.min(), y.max())
    plt.savefig("TEST.png")
    plt.close()


def main(
    model_name: str,
    parameters: Optional[Path],
    batch_size: int,
    label_name: str,
    device: Optional[str] = None,
):
    date_and_time = datetime.now()
    set_resource_limit(4096)

    # get model class
    model_class = {
        "dixonnet": DixonNet,
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
    }[model_name]

    # get label function
    label_function = {
        "shift": CSVFile.is_shift_day,
        "sleep": CSVFile.is_sleep_long,
        "activity": CSVFile.is_activity_broken,
        "session": CSVFile.is_session_morning,
    }[label_name]

    # detect devices
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Found {device} device.")

    # load data
    print("\nLoading data...")
    columns = 2, 3, 4
    window_size = 4096

    files = load_csv_files(
        find_files(["data"], "csv"),
        columns,
        window_size,
        label_fn=label_function,
    )

    print(f"Found {len(files)} data files.")
    data = batch_data(files, batch_size, shuffle=False)
    print(f"Data size: {len(data.dataset)}")

    # build model
    print("\nBuilding model...")
    model = model_class(num_classes=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name} parameters: {n_params}")

    # load parameters
    if parameters:
        print(f"\nLoading {parameters}...")
        model.load_state_dict(torch.load(parameters))

    model.eval()

    with torch.no_grad():
        for item, label in data:
            item, label = item.to(device), label.to(device)
            f_k = model.get_features(item)
            for i in f_k:
                print(i[:4])
            w = model.get_final_weights()
            f_k = f_k.unsqueeze(1)
            w = w.unsqueeze(0).unsqueeze(3)
            m = torch.sum(w * f_k, dim=2)

            for y, h in zip(item, m):\
                # print(y.shape, heatmap.shape)
                y = y[0]
                h = h[0]
                plot_coloured_line(np.arange(len(h) + 1), y[:len(h) + 1], h)

                print(h[0:4])


if __name__ == "__main__":
    import argparse

    models = [
        "dixonnet",
        "resnet18",
        "resnet34",
        "resnet50",
    ]

    labels = [
        "shift",
        "sleep",
        "activity",
        "session",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", choices=models, default="resnet18")
    parser.add_argument("-p", "--parameters", type=Path)
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-l", "--label-name", choices=labels, default="sleep")
    parser.add_argument("-d", "--device", default=None)
    main(**vars(parser.parse_args()))
