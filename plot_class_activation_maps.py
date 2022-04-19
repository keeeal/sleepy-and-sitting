from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence
from matplotlib.collections import LineCollection

import torch
from torch import cuda, negative, nn
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from models.dixonnet import DixonNet
from models.resnet import resnet18, resnet34, resnet50
from utils.data import CSVFile, batch_data, find_files, k_fold_splits, load_csv_files
from utils.misc import set_resource_limit


def plot_coloured_line(
    file: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray, c: np.ndarray
):
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html

    assert len(x) == len(y) == len(z) == len(c)
    r = np.arange(len(x))

    x -= x.mean()
    y -= y.mean()
    z -= z.mean()

    c = c[:-1]
    c /= np.abs(c).max()

    x_points = np.array([r, x]).T.reshape(-1, 1, 2)
    x_segments = np.concatenate([x_points[:-1], x_points[1:]], axis=1)

    y_points = np.array([r, y]).T.reshape(-1, 1, 2)
    y_segments = np.concatenate([y_points[:-1], y_points[1:]], axis=1)

    z_points = np.array([r, z]).T.reshape(-1, 1, 2)
    z_segments = np.concatenate([z_points[:-1], z_points[1:]], axis=1)

    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
    norm = plt.Normalize(-1, 1)

    x_lc = LineCollection(x_segments, cmap="coolwarm", norm=norm)
    x_lc.set_array(c)
    x_lc.set_linewidth(2)
    x_line = axs[0].add_collection(x_lc)
    fig.colorbar(x_line, ax=axs[0])

    y_lc = LineCollection(y_segments, cmap="coolwarm", norm=norm)
    y_lc.set_array(c)
    y_lc.set_linewidth(2)
    y_line = axs[1].add_collection(y_lc)
    fig.colorbar(y_line, ax=axs[1])

    z_lc = LineCollection(z_segments, cmap="coolwarm", norm=norm)
    z_lc.set_array(c)
    z_lc.set_linewidth(2)
    z_line = axs[2].add_collection(z_lc)
    fig.colorbar(z_line, ax=axs[2])

    axs[0].set_xlim(r.min(), r.max())
    axs[0].set_ylim(-1, 1)

    plt.savefig(file)
    plt.close()


def main(
    model_name: str,
    parameters: Optional[Path],
    batch_size: int,
    window_size: int,
    k_fold: int,
    med_filt_size: int,
    low_pass_freq: float,
    label_name: str,
    device: Optional[str] = None,
) -> None:

    # This has something to do with the way threads share data. Without
    # increasing the resource limit, the program sometimes hangs without
    # error - depending on the amount of data.
    set_resource_limit(4096)

    # Determine the model class from the model name string argument. This is
    # the either the class initialiser or a function that returns an
    # initialised class.
    model_class = {
        "dixonnet": DixonNet,
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
    }[model_name]

    # Determine the label function from the label name string argument. This
    # is a function that takes a CSVFile and returns a boolean to indicate
    # which binary class this data belongs to.
    label_function = {
        "shift": CSVFile.is_shift_day,
        "sleep": CSVFile.is_sleep_long,
        "activity": CSVFile.is_activity_broken,
        "session": CSVFile.is_session_morning,
    }[label_name]

    # Detect whether a CUDA GPU is available. If so, detect how many.
    device = device or ("cuda" if cuda.is_available() else "cpu")
    n_devices = cuda.device_count() if device == "cuda" else 1
    print(f"Found {n_devices} {device} devices.")

    # Load the data. Files are found in the data directory based on file
    # extension. They are sorted for repeatability - e.g. If loaded again
    # for futher model evaluation, they need to be split into the same
    # training and evaluation subsets.
    print("\nLoading data...")
    files = load_csv_files(
        sorted(find_files(["data"], "csv")),
        label_fn=label_function,
        columns=(2, 3, 4),
        window_size=window_size,
        median_filter_size=med_filt_size,
        low_pass_frequency=low_pass_freq,
        stride=window_size,
    )
    print(f"Found {len(files)} data files.")

    # Split the data into training and testing subsets. This is done on a
    # file-by-file basis, by k-fold validation method.
    k_fold = max(k_fold, 2)
    assert k_fold < len(files), f"Expected at least {k_fold} files."
    split_files = k_fold_splits(files, k=k_fold)

    # Batch the data. The data is no longer sepererated by file. The datasets
    # in each training and testing subset are combined into one dataset, then
    # their elements are shuffled and batched.
    data = [
        [batch_data(ds, batch_size, shuffle=False) for ds in split]
        for split in split_files
    ]

    for k, (_, test_data) in enumerate(data):
        print(f"\nfold {k + 1} of {len(data)}".upper())
        print(f"Testing data size: {len(test_data.dataset)}")

        # Build the model and put it on the chosen device.
        print("\nBuilding model...")
        model = model_class(num_classes=1).to(device)
        n_params = sum(p.numel() for p in model.parameters())

        # If there are multiple GPUs, the model needs wrapping in the
        # DataParallel class.
        if n_devices > 1:
            model = nn.DataParallel(model)

        print(model_name.upper())
        print(f"Parameters: {n_params}")

        # If a path to the parameters directory had been provided, find the
        # file in it with the correct fold number and highest epoch number.
        if parameters:
            files = find_files([parameters], "params")
            files = [f for f in files if f.stem.split("-")[1] == str(k)]
            file = max(files, key=lambda f: int(f.stem.split("-")[2]))

            print(f"\nLoading {file}...")
            model.load_state_dict(torch.load(file))

        # Put the model into evaluation mode.
        model.eval()
        n_per_class = 5
        windows = []

        # Evaluate every window in the test data for the current fold, keeping
        # only those with the most confident classifications for each class.
        with torch.no_grad():
            for batch, _ in tqdm(test_data):
                batch = batch.to(device)

                for window, value in zip(batch, model(batch)):
                    windows.append((window.cpu(), value.item()))

                if 2 * n_per_class < len(windows):
                    windows = sorted(windows, key=lambda x: x[1])
                    windows = windows[:n_per_class] + windows[-n_per_class:]

        # Prepare for class activation mapping by getting the weights from the
        # model's last fully connected layer as a tensor and reshaping it.
        model = model.cpu()
        w = model.get_final_layer_weights()
        w = w.unsqueeze(0).unsqueeze(3)

        # For each window, do class activation mapping and plot the result.
        for n, (window, _) in enumerate(windows):
            window = window.unsqueeze(0)

            with torch.no_grad():
                f_k = model.get_features(window)

            f_k = f_k.unsqueeze(1)
            m = torch.sum(w * f_k, dim=2)

            # dixonnet only
            m = nn.AdaptiveAvgPool1d(4 * m.shape[-1])(m)
            window = window[:, :, 76:-76]

            window = window.detach().numpy()
            m = m.detach().numpy()

            for (x, y, z), h in zip(window, m):
                for c in h:
                    plot_coloured_line(Path(f"{k=},{n=}.png"), x, y, z, c)


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
    parser.add_argument("-w", "--window-size", type=int, default=4096)
    parser.add_argument("-k", "--k-fold", type=int, default=5)
    parser.add_argument("-mf", "--med-filt-size", type=int, default=0)
    parser.add_argument("-lp", "--low-pass-freq", type=float, default=0)
    parser.add_argument("-l", "--label-name", choices=labels, default="sleep")
    parser.add_argument("-d", "--device", default=None)
    main(**vars(parser.parse_args()))
