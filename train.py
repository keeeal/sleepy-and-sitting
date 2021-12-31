import os
from dataclasses import dataclass
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
from utils.model import DixonNet


@dataclass
class ConfusionMatrix:
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    def __len__(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
        )

    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / len(self)


def evaluate(
    model: nn.Module,
    data: Iterable[tuple[Tensor, Tensor]],
    lossfn: _Loss,
    device: str,
    n: Optional[int] = None,
) -> tuple[float, ConfusionMatrix]:
    model.eval()
    losses = []
    table = ConfusionMatrix()

    with torch.no_grad():
        for item, label in tqdm(islice(data, n) if n is not None else data, total=n):
            item, label = item.to(device), label.to(device)
            output = model(item).squeeze()
            losses.append(lossfn(output, label).item())

            for prediction, truth in zip(output, label):
                if prediction < 0.5:
                    if truth < 0.5:
                        table.true_negative += 1
                    else:
                        table.false_negative += 1
                else:
                    if truth < 0.5:
                        table.false_positive += 1
                    else:
                        table.true_positive += 1

    loss = sum(losses) / len(losses)
    return loss, table


def train(
    model: nn.Module,
    data: Iterable[tuple[Tensor, Tensor]],
    lossfn: _Loss,
    optimr: Optimizer,
    schdlr: Union[_LRScheduler, ReduceLROnPlateau],
    device: str,
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
        # del item, label, loss

    loss = sum(losses) / len(losses)

    if isinstance(schdlr, ReduceLROnPlateau):
        schdlr.step(loss)
    else:
        schdlr.step()

    return loss


def main(learn_rate: float, batch_size: int, epochs: int, output_dir: Path):

    # detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_devices = torch.cuda.device_count() if device == "cuda" else 1
    print(f"Using {n_devices} {device} device{'s' if n_devices > 1 else ''}.")

    # load data
    print("\nLoading data...")
    data_dirs = [Path("data") / str(n + 1) for n in range(5)]
    train_data = load(data_dirs[:4], batch_size=batch_size, shuffle=True)
    test_data = load(data_dirs[4:], batch_size=batch_size)
    print(f"Training data size: {len(train_data)}")
    print(f"Testing data size: {len(test_data)}")

    # build model
    print("\nBuilding neural network...")
    model = DixonNet().to(device)
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
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Model: {model}\n")

    # start training loop
    print("\nTraining...")
    for epoch in range(1, epochs + 1):

        # get the current learning rate
        lr = [group["lr"] for group in optimr.param_groups]
        lr = lr[0] if len(lr) == 1 else lr

        # train
        loss = train(model, train_data, lossfn, optimr, schdlr, device, n=64)
        print(line := f"Epoch {epoch} | LR: {lr:.4e} | Loss: {loss:.4e}")
        with open(output_dir / "train.txt", "a") as f:
            f.write(line + "\n")

        # save model parameters and evaluate
        if epoch % 10 == 0:
            torch.save(model.state_dict(), output_dir / f"{epoch}.params")

            loss, table = evaluate(model, test_data, lossfn, device, n=1024)
            print(line := f"Loss: {loss:.4e} | Acc: {100*table.accuracy():.2f}%")
            with open(output_dir / "eval.txt", "a") as f:
                f.write(line + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-lr", "--learn-rate", type=float, default=1e-4
    )  # I CHANGED LORD KREELS DEFAULT FROM 1e-3
    parser.add_argument("-b", "--batch-size", type=int, default=124)
    parser.add_argument("-e", "--epochs", type=int, default=2000)
    parser.add_argument("-o", "--output-dir", type=Path, default="trained")
    main(**vars(parser.parse_args()))
