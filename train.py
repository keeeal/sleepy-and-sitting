from datetime import datetime
from itertools import islice
from json import dumps
from pathlib import Path
from random import shuffle
from typing import Any, Iterable, Optional, Union

import torch
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm

from utils.data import (
    batch_data,
    find_files,
    fraction_split,
    load_csv_files,
    k_fold_splits,
)

from models.dixonnet import DixonNet
from models.resnet import resnet18, resnet34, resnet50


def print_as_line(**kwargs: dict[str, str]) -> None:
    """Prints args on a single line as pipe-separated key-value pairs."""
    print(" | ".join(" ".join(i) for i in kwargs.items()))


def log_to_file(file: Path, **kwargs: dict[str, Any]) -> None:
    """Appends args to file as a single line of JSON."""
    with open(file, "a+") as f:
        f.write(dumps(kwargs) + "\n")


def evaluate(
    model: nn.Module,
    data: Iterable[tuple[Tensor, Tensor]],
    lossfn: _Loss,
    device: Union[torch.device, str],
    n: Optional[int] = None,
) -> tuple[float, list[int], list[int]]:
    """
    Evaluates a model.

    args:
        model: The model to evaluate.
        data: The dataset to evaluate the model on.
        lossfn: The function used to measure loss.
        device: The device that the model is on.
        n (optional): The number of batches of data to use.
            If 'None', all data is used. Default: 'None'.

    returns:
        loss: The average loss over the dataset.
        truth: The ground truth values expected.
        prediction: The model's output.
    """
    model.eval()
    losses = []
    truth, prediction = [], []

    with torch.no_grad():
        for item, label in tqdm(islice(data, n) if n is not None else data, total=n):
            item, label = item.to(device), label.to(device)
            output = model(item).squeeze()
            losses.append(lossfn(output, label).item())
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
    lossfn: _Loss,
    optimr: Optimizer,
    schdlr: Union[_LRScheduler, ReduceLROnPlateau],
    device: Union[torch.device, str],
    n: Optional[int] = None,
) -> float:
    """
    Trains a model.

    args:
        model: The model to train.
        data: The dataset to train the model on.
        lossfn: The function used to measure loss.
        optimr: The algorithm used to update the model parameters.
        schdlr: The algorithm used to update the training rate.
        device: The device that the model is on.
        n (optional): The number of batches of data to use.
            If 'None', all data is used. Default: 'None'.

    returns:
        loss: The average loss over the dataset.
    """
    model.train()
    losses = []

    for item, label in tqdm(islice(data, n) if n is not None else data, total=n):
        optimr.zero_grad()
        item, label = item.to(device), label.to(device)
        loss = lossfn(model(item).squeeze(), label)
        losses.append(loss.item())
        loss.backward()
        optimr.step()

    loss = sum(losses) / len(losses)

    if isinstance(schdlr, ReduceLROnPlateau):
        schdlr.step(loss)
    else:
        schdlr.step()

    return loss


def main(
    model_name: str,
    batch_size: int,
    learn_rate: float,
    k_fold: int,
    max_epochs: int,
    device: Optional[str] = None,
    output_dir: Optional[Path] = None,
):
    date_and_time = datetime.now()

    # load data
    print("\nLoading data...")
    columns = 2, 3, 4
    window_size = 4096
    files = load_csv_files(find_files(["data"], "csv"), columns, window_size)
    print(f"Found {len(files)} data files.")

    # split data
    k_fold = max(k_fold, 1)
    assert max(k_fold, 2) < len(files), f"Expected at least {max(k_fold, 2)} files."

    if k_fold == 1:
        shuffle(files)
        split_files = [fraction_split(files, f=0.8)]
    else:
        split_files = k_fold_splits(files, k=k_fold)

    data = [
        [batch_data(fs, batch_size, shuffle=True) for fs in split]
        for split in split_files
    ]

    # detect devices
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    n_devices = torch.cuda.device_count() if device == "cuda" else 1
    print(f"Found {n_devices} {device} device{'s' if n_devices > 1 else ''}.")

    # build model
    print("\nBuilding model...")
    model = {
        "dixonnet": DixonNet,
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
    }[model_name](num_classes=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if n_devices > 1:
        model = torch.nn.DataParallel(model)

    # loss function, optimizer, scheduler
    lossfn = nn.BCEWithLogitsLoss()
    optimr = Adam(model.parameters(), lr=learn_rate)
    schdlr = ReduceLROnPlateau(optimr)

    # create output directory
    if output_dir is None:
        output_dir = Path("output")
        output_dir /= str(date_and_time).replace(" ", "_").replace(":", ".")

    params_dir = output_dir / "params"
    log_file = output_dir / "log.ndjson"
    params_dir.mkdir(parents=True, exist_ok=True)

    # save model config
    with open(output_dir / f"{model_name}.txt", "w") as f:
        f.write(f"{date_and_time = }\n")
        f.write(f"{model_name = }\n")
        f.write(f"{model = }\n")
        f.write(f"{n_params = }\n")
        f.write(f"{device = }\n")
        f.write(f"{n_devices = }\n")

    for k, (train_data, test_data) in enumerate(data):
        print(f"\nfold {k + 1} of {len(data)}".upper())
        print(f"Training data size: {len(train_data.dataset)}")
        print(f"Testing data size: {len(test_data.dataset)}")

        # start training
        print("\nTraining...")
        for epoch in range(1, max_epochs + 1):

            # check the current learning rate
            lrs = [group["lr"] for group in optimr.param_groups]
            if all(lr < 1e-7 for lr in lrs):
                print("Training ended.")
                break

            # train
            loss = train(model, train_data, lossfn, optimr, schdlr, device, n=100)

            print_as_line(
                epoch=str(epoch),
                lr=", ".join(f"{lr:.4e}" for lr in lrs),
                loss=f"{loss:.4e}",
            )

            log_to_file(
                log_file,
                data="training",
                model=model_name,
                window_size=window_size,
                batch_size=batch_size,
                fold=k,
                epoch=epoch,
                loss=loss,
                learning_rate=lrs[0],
            )

            # save model parameters and evaluate
            if epoch % 10 == 0:
                torch.save(
                    model.state_dict(), params_dir / f"{model_name}-{k}-{epoch}.params",
                )

                print()

                loss, truth, prediction = evaluate(
                    model, test_data, lossfn, device, n=1000
                )
                confusion = confusion_matrix(truth, prediction)
                tn, fp, fn, tp = map(int, confusion.ravel())
                accuracy = accuracy_score(truth, prediction)
                f_score = f1_score(truth, prediction)

                print_as_line(
                    loss=f"{loss:.4e}", acc=f"{accuracy:.2%}", f=f"{f_score:.2f}",
                )

                log_to_file(
                    log_file,
                    data="evaluation",
                    model=model_name,
                    window_size=window_size,
                    batch_size=batch_size,
                    fold=k,
                    epoch=epoch,
                    loss=loss,
                    accuracy=100 * accuracy,
                    f_score=f_score,
                    true_positives=tp,
                    false_positives=fp,
                    true_negative=tn,
                    false_negative=fn,
                )

                print()


if __name__ == "__main__":
    import argparse

    models = [
        "dixonnet",
        "resnet18",
        "resnet34",
        "resnet50",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", choices=models, default="resnet18")
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument(
        "-lr", "--learn-rate", type=float, default=1e-4
    )  # I CHANGED LORD KREELS DEFAULT FROM 1e-3
    parser.add_argument("-k", "--k-fold", type=int, default=1)
    parser.add_argument("-e", "--max-epochs", type=int, default=1000)
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    main(**vars(parser.parse_args()))
