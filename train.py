import os
from datetime import datetime
from itertools import chain, islice
from pathlib import Path
from typing import Iterable, Optional, Union

import torch
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from tqdm import tqdm

from utils.data import find_files, load
from utils.math import ConfusionMatrix

from models.dixonnet import DixonNet
from models.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from models.efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2
from models.rnn import rnn, lstm, gru
from models.transfomer import Transformer
from models.wavenet import WaveNetModel


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


def main(window_size: int, batch_size: int, learn_rate: float, k_fold: int, max_epochs: int, output_dir: Optional[Path]):
    date_and_time = str(datetime.now())
    if output_dir is None:
        output_dir = Path(date_and_time.replace(" ", "_").replace(":", "."))

    # detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_devices = torch.cuda.device_count() if device == "cuda" else 1
    print(f"Found {n_devices} {device} device{'s' if n_devices > 1 else ''}.")

    # find and split data
    files = find_files(["data"], "csv", shuffle=True)
    print(f"Found {len(files)} data files.")

    if max(k_fold, 2) > len(files):
        print(f"Expected at least {max(k_fold, 2)} files.")
        return

    if k_fold < 2:
        n_train = int(0.8 * len(files))
        splits = [(files[:n_train], files[n_train:])]
    else:
        parts = [files[n::k_fold] for n in range(k_fold)]
        splits = [
            (list(chain(*parts[:n], *parts[n + 1 :])), parts[n]) for n in range(k_fold)
        ]

    # check output directory
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for m, Model in enumerate((resnet34,)):
        for k, (train_files, test_files) in enumerate(splits):
            print(f"\n#### MODEL {m} FOLD {k} OF {k_fold} ####")

            # load data
            print("\nLoading data...")
            train_data = load(train_files, window_size, batch_size, shuffle=True)
            test_data = load(test_files, window_size, batch_size, shuffle=True)
            print(f"Training data size: {len(train_data.dataset)}")
            print(f"Testing data size: {len(test_data.dataset)}")

            # build model
            print("\nBuilding neural network...")
            model = Model(num_classes=1).to(device)
            if n_devices > 1:
                model = torch.nn.DataParallel(model)

            # loss function, optimizer, scheduler
            lossfn = nn.BCEWithLogitsLoss()
            optimr = Adam(model.parameters(), lr=learn_rate)
            schdlr = ReduceLROnPlateau(optimr)

            # check output directory
            output_dir_m_k = output_dir / str(m) / str(k)
            if not os.path.isdir(output_dir / str(m)):
                os.mkdir(output_dir / str(m))
            if not os.path.isdir(output_dir_m_k):
                os.mkdir(output_dir_m_k)

            # clear history files
            open(output_dir_m_k / "train.txt", "w").close()
            open(output_dir_m_k / "eval.txt", "w").close()
            open(output_dir_m_k / "final.txt", "w").close()

            # save config details
            with open(output_dir_m_k / "config.txt", "w") as f:
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
                    break

                # train
                loss = train(model, train_data, lossfn, optimr, schdlr, device, n=100)
                line = " | ".join(
                    (
                        f"Epoch {epoch}",
                        "LR " + ", ".join(f"{lr:.4e}" for lr in lrs),
                        f"Loss {loss:.4e}",
                    )
                )

                with open(output_dir_m_k / "train.txt", "a") as f:
                    f.write(line + "\n")
                    print(line)

                # save model parameters and evaluate
                if epoch % 10 == 0:
                    torch.save(
                        model.state_dict(), output_dir_m_k / f"{epoch}.params"
                    )

                    print()

                    loss, cm = evaluate(model, test_data, lossfn, device, n=1000)
                    line = " | ".join(
                        (
                            f"Loss {loss:.4e}",
                            f"Acc {100 * cm.accuracy():.2f}%",
                            f"F {cm.f_score():.2f}",
                            f"TP {cm.true_positive}",
                            f"FP {cm.false_positive}",
                            f"TN {cm.true_negative}",
                            f"FN {cm.false_negative}",
                        )
                    )

                    with open(output_dir_m_k / "eval.txt", "a") as f:
                        f.write(line + "\n")
                        print(line)

                    print()

            # # final evaluation
            # torch.save(model.state_dict(), output_dir_m_k / f"final.params")

            # print()

            # loss, cm = evaluate(model, test_data, lossfn, device)
            # line = " | ".join(
            #     (
            #         f"Loss {loss:.4e}",
            #         f"Acc {100 * cm.accuracy():.2f}%",
            #         f"F {cm.f_score():.2f}",
            #         f"TP {cm.true_positive}",
            #         f"FP {cm.false_positive}",
            #         f"TN {cm.true_negative}",
            #         f"FN {cm.false_negative}",
            #     )
            # )

            # with open(output_dir_m_k / "final.txt", "a") as f:
            #     f.write(line + "\n")
            #     print(line)

            # print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--window-size", type=int, default=4096)
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument(
        "-lr", "--learn-rate", type=float, default=1e-4
    )  # I CHANGED LORD KREELS DEFAULT FROM 1e-3
    parser.add_argument("-k", "--k-fold", type=int, default=0)
    parser.add_argument("-e", "--max-epochs", type=int, default=1000)
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    main(**vars(parser.parse_args()))
