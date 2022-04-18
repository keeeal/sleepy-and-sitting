from argparse import ArgumentParser
from datetime import datetime
from itertools import islice
from json import dumps
from pathlib import Path
from random import shuffle
from typing import Iterable, Optional, Union

import torch
from torch import cuda, device, nn, Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm

from utils.data import (
    CSVFile,
    batch_data,
    find_files,
    fraction_split,
    load_csv_files,
    k_fold_splits,
)
from utils.misc import set_resource_limit

from models.dixonnet import DixonNet
from models.resnet import resnet18, resnet34, resnet50


def print_as_line(**kwargs: str) -> None:
    """Print args on a single line as pipe-separated key-value pairs."""
    print(" | ".join(" ".join(item) for item in kwargs.items()))


def log_to_file(file: Path, **kwargs) -> None:
    """Append args to a file as a single line of JSON."""
    with open(file, "a+") as f:
        f.write(dumps(kwargs) + "\n")


def evaluate(
    model: nn.Module,
    data: Iterable[tuple[Tensor, Tensor]],
    loss_fn: _Loss,
    device: Union[device, str],
) -> tuple[float, list[int], list[int]]:
    """
    Evaluates a model on the data provided.

    Parameters
    ==========
    model: The model to evaluate.
    data: The dataset to evaluate the model on.
    loss_fn: The function used to measure loss.
    device: The device that the model is on.

    Returns
    =======
    loss: The average loss over the dataset.
    truth: The ground truth values expected.
    prediction: The model's output.
    """

    model.eval()
    losses: list[float] = []
    truth: list[int] = []
    prediction: list[int] = []

    with torch.no_grad():
        for item, label in tqdm(data):
            item, label = item.to(device), label.to(device)
            output = model(item).squeeze()
            losses.append(loss_fn(output, label).item())
            truth.extend(map(int, label))

            if len(output.shape) == 1:
                prediction.extend(map(int, output > 0))
            else:
                prediction.extend(map(int, output.argmax(dim=1)))

    loss = sum(losses) / len(losses)
    return loss, truth, prediction


def train(
    model: nn.Module,
    data: Iterable[tuple[Tensor, Tensor]],
    loss_fn: _Loss,
    optimizer: Optimizer,
    scheduler: Union[_LRScheduler, ReduceLROnPlateau],
    device: Union[device, str],
) -> float:
    """
    Performs one epoch of training by passing once through the data provided,
    updating the parameters of the model in a direction that minimises loss
    and in accordance with the optimiser and learning rate schedule.

    Parameters
    ==========
    model: The model to train.
    data: The dataset to train the model on.
    loss_fn: The function used to measure loss.
    optimizer: The algorithm used to update the model parameters.
    scheduler: The algorithm used to update the training rate.
    device: The device that the model is on.

    Returns
    =======
    loss: The average loss over the dataset.
    """

    model.train()
    losses = []

    for item, label in tqdm(data):
        optimizer.zero_grad()
        item, label = item.to(device), label.to(device)
        loss = loss_fn(model(item).squeeze(), label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    loss = sum(losses) / len(losses)

    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(loss)
    else:
        scheduler.step()

    return loss


def main(
    model_name: str,
    label_name: str,
    batch_size: int,
    window_size: int,
    learn_rate: float,
    k_fold: int,
    max_epochs: int,
    med_filt_size: int,
    low_pass_freq: float,
    device: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> None:

    # Record the date and time that training started.
    date_and_time = datetime.now()

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
    )
    print(f"Found {len(files)} data files.")

    # Split the data into training and testing subsets. This is done on a
    # file-by-file basis, either by fraction or by k-fold validation method.
    # If done by fraction, the data is shuffled first. This is not repeatable
    # unless a random seed is specified, which it is not. The k-fold method
    # must be used if repeatability is required.
    k_fold = max(k_fold, 1)
    assert max(k_fold, 2) < len(files), f"Expected at least {max(k_fold, 2)} files."

    if k_fold == 1:
        shuffle(files)
        split_files = [fraction_split(files, f=0.8)]
    else:
        split_files = k_fold_splits(files, k=k_fold)

    # Batch the data. The data is no longer sepererated by file. The datasets
    # in each training and testing subset are combined into one dataset, then
    # their elements are shuffled and batched.
    data = [
        [batch_data(ds, batch_size, shuffle=True) for ds in split]
        for split in split_files
    ]

    # Create an output directory to contain logs and model parameters.
    if output_dir is None:
        output_dir = Path("output")
        output_dir /= str(date_and_time).replace(" ", "_").replace(":", ".")

    params_dir = output_dir / "params"
    log_file = output_dir / "log.ndjson"
    params_dir.mkdir(parents=True, exist_ok=True)

    # We iterate over each split of the data. If the data was split by
    # fraction, there will only be one training / testing pair of
    # datasets. Otherwise there will be one for each k-fold.
    for k, (train_data, test_data) in enumerate(data):
        print(f"\nfold {k + 1} of {len(data)}".upper())
        print(f"Training data size: {len(train_data.dataset)}")
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

        # Create the loss function, optimiser and learning rate scheduler.
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = Adam(model.parameters(), lr=learn_rate)
        scheduler = ReduceLROnPlateau(optimizer)

        # Save the model configuration.
        with open(output_dir / f"{model_name}.txt", "w") as f:
            f.write(f"{date_and_time = }\n")
            f.write(f"{model_name = }\n")
            f.write(f"{model = }\n")
            f.write(f"{n_params = }\n")
            f.write(f"{device = }\n")
            f.write(f"{n_devices = }\n")

        # Begin training. We train until either max epochs have occurred or
        # the learning rate has reached a very small value.
        print("\nTraining...")
        min_lr = 1e-7

        for epoch in range(1, max_epochs + 1):

            # Check the learning rate and terminate if it is less than min_lr.
            lrs = [group["lr"] for group in optimizer.param_groups]
            if all(lr < min_lr for lr in lrs):
                print("Training ended.")
                break

            # Call the train function. An true epoch is a pass over the entire
            # training data but that would take a long time. Instead we grab a
            # few batches and train on that. The data is shuffled so these are
            # not the same batches every time.
            some_train_data = islice(train_data, 100)
            loss = train(model, some_train_data, loss_fn, optimizer, scheduler, device)

            # Print some info and log a lot of info.
            print_as_line(
                epoch=str(epoch),
                lr=", ".join(f"{lr:.4e}" for lr in lrs),
                loss=f"{loss:.4e}",
            )
            log_to_file(
                log_file,
                data="training",
                label=label_name,
                model=model_name,
                window_size=window_size,
                batch_size=batch_size,
                fold=k,
                epoch=epoch,
                loss=loss,
                learning_rate=lrs[0],
            )

            # Every so often, save the model parameters and evaluate the model.
            if epoch % 10 == 0:
                torch.save(
                    model.state_dict(), params_dir / f"{model_name}-{k}-{epoch}.params",
                )

                print()

                # Again, evaluating on all available data would take a long
                # time, so we grab a few batches an evaluate on that. The data
                # is shuffled so these are not the same batches every time.
                some_test_data = islice(test_data, 1000)
                loss, truth, prediction = evaluate(
                    model, some_test_data, loss_fn, device
                )

                # Calculate some performance metrics.
                confusion = confusion_matrix(truth, prediction)
                tn, fp, fn, tp = map(int, confusion.ravel())
                accuracy = accuracy_score(truth, prediction)
                f_score = f1_score(truth, prediction)

                # Print some info and log a lot of info.
                print_as_line(
                    loss=f"{loss:.4e}", acc=f"{accuracy:.2%}", f=f"{f_score:.2f}",
                )
                log_to_file(
                    log_file,
                    data="evaluation",
                    label=label_name,
                    model=model_name,
                    window_size=window_size,
                    batch_size=batch_size,
                    median_filter_size=med_filt_size,
                    low_pass_frequency=low_pass_freq,
                    fold=k,
                    epoch=epoch,
                    loss=loss,
                    accuracy=100 * accuracy,
                    f_score=f_score,
                    true_positives=tp,
                    false_positives=fp,
                    true_negatives=tn,
                    false_negatives=fn,
                )

                print()


if __name__ == "__main__":
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

    parser = ArgumentParser()
    parser.add_argument("-m", "--model-name", choices=models, default="resnet18")
    parser.add_argument("-l", "--label-name", choices=labels, default="sleep")
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-w", "--window-size", type=int, default=4096)
    parser.add_argument(
        "-lr", "--learn-rate", type=float, default=1e-4
    )  # I CHANGED LORD KREELS DEFAULT FROM 1e-3
    parser.add_argument("-k", "--k-fold", type=int, default=1)
    parser.add_argument("-e", "--max-epochs", type=int, default=1000)
    parser.add_argument("-mf", "--med-filt-size", type=int, default=0)
    parser.add_argument("-lp", "--low-pass-freq", type=float, default=0)
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    main(**vars(parser.parse_args()))
