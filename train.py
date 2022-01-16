import os, random, json
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import torch
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from tqdm import tqdm

from utils.data import find_files, split, k_fold_splits, load
from utils.confusion_matrix import ConfusionMatrix

from models.dixonnet import DixonNet
from models.resnet import resnet18, resnet34, resnet50


def print_line(**kwargs: dict[str, str]) -> None:
    print(" | ".join(" ".join(i) for i in kwargs.items()))


def log_ndjson(file: Path, **kwargs: dict[str, Any]) -> None:
    with open(file, "a") as f:
        f.write(json.dumps(kwargs) + "\n")


def evaluate(
    model: nn.Module,
    data: Iterable[tuple[Tensor, Tensor]],
    lossfn: _Loss,
    device: Union[torch.device, str],
    n: Optional[int] = None,
) -> tuple[float, ConfusionMatrix]:
    model.eval()
    losses = []
    cm = ConfusionMatrix()

    with torch.no_grad():
        for item, label in tqdm(islice(data, n) if n is not None else data, total=n):
            item, label = item.to(device), label.to(device)
            output = model(item).squeeze()
            losses.append(lossfn(output, label).item())
            output = output.sigmoid()

            for prediction, truth in zip(output, label):
                if prediction < 0.5:
                    if truth < 0.5:
                        cm.true_negative += 1
                    else:
                        cm.false_negative += 1
                else:
                    if truth < 0.5:
                        cm.false_positive += 1
                    else:
                        cm.true_positive += 1

    loss = sum(losses) / len(losses)
    return loss, cm


def train(
    model: nn.Module,
    data: Iterable[tuple[Tensor, Tensor]],
    lossfn: _Loss,
    optimr: Optimizer,
    schdlr: Union[_LRScheduler, ReduceLROnPlateau],
    device: Union[torch.device, str],
    n: Optional[int] = None,
) -> float:
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
    batch_size: int,
    learn_rate: float,
    k_fold: int,
    max_epochs: int,
    output_dir: Optional[Path] = None,
):
    date_and_time = str(datetime.now())
    if output_dir is None:
        output_dir = Path(date_and_time.replace(" ", "_").replace(":", "."))

    # detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_devices = torch.cuda.device_count() if device == "cuda" else 1
    print(f"Found {n_devices} {device} device{'s' if n_devices > 1 else ''}.")

    # find and split data
    files = list(find_files(["data"], "csv"))
    print(f"Found {len(files)} data files.")
    k_fold = max(k_fold, 1)
    assert max(k_fold, 2) < len(files), f"Expected at least {max(k_fold, 2)} files."

    if k_fold == 1:
        random.shuffle(files)
        splits = [split(files, f=0.8)]
    else:
        splits = k_fold_splits(files, k=k_fold)

    # create output directory
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    params_dir = output_dir / "params"
    if not os.path.isdir(params_dir):
        os.mkdir(params_dir)

    # clear log file
    open(output_dir / "log.ndjson", "w").close()

    for model_name, Model in (
        ("dixonnet", DixonNet),
        ("resnet18", resnet18),
        ("resnet34", resnet34),
        ("resnet50", resnet50),
    ):
        for window_size in [4096]:
            for k, (train_files, test_files) in enumerate(splits):
                print(f"\n{model_name} - fold {k + 1} of {len(splits)}".upper())

                # load data
                print("\nLoading data...")
                train_data = load(
                    train_files,
                    window_size,
                    batch_size,
                    shuffle=True,
                    resource_limit=4096,
                )
                test_data = load(
                    test_files,
                    window_size,
                    batch_size,
                    shuffle=True,
                    resource_limit=4096,
                )
                print(f"Training data size: {len(train_data.dataset)}")
                print(f"Testing data size: {len(test_data.dataset)}")

                # build model
                print("\nBuilding neural network...")
                model = Model(num_classes=1).to(device)
                n_params = sum(p.numel() for p in model.parameters())
                if n_devices > 1:
                    model = torch.nn.DataParallel(model)

                # loss function, optimizer, scheduler
                lossfn = nn.BCEWithLogitsLoss()
                optimr = Adam(model.parameters(), lr=learn_rate)
                schdlr = ReduceLROnPlateau(optimr)

                # save model config
                with open(output_dir / f"{model_name}.txt", "w") as f:
                    f.write(f"{date_and_time = }\n")
                    f.write(f"{model = }\n")
                    f.write(f"{device = }\n")
                    f.write(f"{n_params = }\n")

                # start training
                print("\nTraining...")
                for epoch in range(1, max_epochs + 1):

                    # check the current learning rate
                    lrs = [group["lr"] for group in optimr.param_groups]
                    if all(lr < 1e-7 for lr in lrs):
                        print("Training ended.")
                        break

                    # train
                    loss = train(
                        model, train_data, lossfn, optimr, schdlr, device, n=100
                    )

                    print_line(
                        epoch=str(epoch),
                        lr=", ".join(f"{lr:.4e}" for lr in lrs),
                        loss=f"{loss:.4e}",
                    )

                    log_ndjson(
                        output_dir / "log.ndjson",
                        data="training",
                        model=model_name,
                        window_size=window_size,
                        batch_size=batch_size,
                        epoch=epoch,
                        loss=loss,
                        learning_rate=lrs[0],
                    )

                    # save model parameters and evaluate
                    if epoch % 10 == 0:
                        torch.save(
                            model.state_dict(),
                            params_dir
                            / f"{model_name}_{window_size=}_{k=}_{epoch}.params",
                        )

                        print()

                        loss, cm = evaluate(model, test_data, lossfn, device, n=None)

                        print_line(
                            loss=f"{loss:.4e}",
                            acc=f"{100 * cm.accuracy():.2f}%",
                            f=f"{cm.f_score():.2f}",
                        )

                        log_ndjson(
                            output_dir / "log.ndjson",
                            data="evaluation",
                            model=model_name,
                            window_size=window_size,
                            batch_size=batch_size,
                            epoch=epoch,
                            loss=loss,
                            accuracy=100 * cm.accuracy(),
                            f_score=cm.f_score(),
                            true_positives=cm.true_positive,
                            false_positives=cm.false_positive,
                            true_negative=cm.true_negative,
                            false_negative=cm.false_negative,
                        )

                        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument(
        "-lr", "--learn-rate", type=float, default=1e-4
    )  # I CHANGED LORD KREELS DEFAULT FROM 1e-3
    parser.add_argument("-k", "--k-fold", type=int, default=1)
    parser.add_argument("-e", "--max-epochs", type=int, default=1000)
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    main(**vars(parser.parse_args()))
