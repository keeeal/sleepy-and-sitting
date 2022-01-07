import os
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Iterable, Optional, Union

import torch
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from tqdm import tqdm

from utils.data import load
from utils.resnet import resnet50 as resnet
from utils.math import ConfusionMatrix


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


def main(learn_rate: float, max_epochs: int, output_dir: Path):
    date_and_time = str(datetime.now())
    if output_dir is None:
        output_dir = Path(date_and_time.replace(" ", "_").replace(":", "."))

    # detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_devices = torch.cuda.device_count() if device == "cuda" else 1
    print(f"Found {n_devices} {device} device{'s' if n_devices > 1 else ''}.")

    # load data
    print("\nLoading data...")
    data_dirs = [Path("data") / str(n) for n in range(5)]
    train_data = load(dirs=data_dirs[1:], shuffle=True)
    test_data = load(dirs=data_dirs[:1], shuffle=True)
    print(f"Training data size: {len(train_data)}")
    print(f"Testing data size: {len(test_data)}")

    # build model
    print("\nBuilding neural network...")
    # model = DixonNet().to(device)
    model = resnet(num_classes=1).to(device)
    if n_devices > 1:
        model = torch.nn.DataParallel(model)

    # loss function, optimizer, scheduler
    lossfn = nn.BCEWithLogitsLoss()
    optimr = Adam(model.parameters(), lr=learn_rate)
    schdlr = ReduceLROnPlateau(optimr)

    # check output directory
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # clear history files
    open(output_dir / "train.txt", "w").close()
    open(output_dir / "eval.txt", "w").close()

    # save config details
    with open(output_dir / "config.txt", "w") as f:
        f.write(f"Date: {date_and_time}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Model: {model}\n")

    # start training
    print("\nTraining...")
    for epoch in range(1, max_epochs + 1):

        # get the current learning rate
        lrs = [group["lr"] for group in optimr.param_groups]
        if all(lr < 1e-7 for lr in lrs):
            print("Training ended.")
            return

        # train
        loss = train(model, train_data, lossfn, optimr, schdlr, device, n=100)
        line = " | ".join(
            (
                f"Epoch {epoch}",
                "LR: " + ", ".join(f"{lr:.4e}" for lr in lrs),
                f"Loss: {loss:.4e}",
            )
        )

        with open(output_dir / "train.txt", "a") as f:
            f.write(line + "\n")
            print(line)

        # save model parameters and evaluate
        if epoch % 10 == 0:
            torch.save(model.state_dict(), output_dir / f"{epoch}.params")

            print()

            loss, cm = evaluate(model, test_data, lossfn, device, n=1000)
            line = " | ".join(
                (
                    f"Loss: {loss:.4e}",
                    f"Acc: {100 * cm.accuracy():.2f}%",
                    f"F: {cm.f_score():.2f}",
                    f"TP: {100 * cm.true_positive / len(cm):.2f}%",
                    f"FP: {100 * cm.false_positive / len(cm):.2f}%",
                    f"TN: {100 * cm.true_negative / len(cm):.2f}%",
                    f"FN: {100 * cm.false_negative / len(cm):.2f}%",
                )
            )

            with open(output_dir / "eval.txt", "a") as f:
                f.write(line + "\n")
                print(line)

            print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-lr", "--learn-rate", type=float, default=1e-3
    )  # I CHANGED LORD KREELS DEFAULT FROM 1e-3
    parser.add_argument("-e", "--max-epochs", type=int, default=1000)
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    main(**vars(parser.parse_args()))
