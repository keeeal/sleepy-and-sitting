import os, time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import torch
from tqdm import tqdm

from utils.data import load_csv_files
from utils.model import DixonNet

@dataclass
class Table:
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    def __len__(self) -> int:
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / len(self)


def train(model, data, lossfn, optimr, schdlr, device) -> float:
    model.train()

    for item, label in tqdm(data):
        optimr.zero_grad()
        item, label = item.to(device), label.to(device)
        loss = lossfn(model(item).squeeze(), label)
        loss.backward()
        optimr.step()
        del item, label, loss

    # get training loss
    model.eval()
    losses = []

    with torch.no_grad():
        for item, label in data:
            item, label = item.to(device), label.to(device)
            loss = lossfn(model(item).squeeze(), label)
            losses.append(loss.item())
            del item, label, loss

    loss = sum(losses) / len(losses)
    schdlr.step(loss)
    return loss


def evaluate(model, data, lossfn, device):
    model.eval()
    losses = []
    table = Table()

    with torch.no_grad():
        for item, label in tqdm(data):
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


def main(learn_rate, batch_size, epochs, output_dir):

    print("Loading data...")
    data_dirs = [Path("data") / str(n + 1) for n in range(5)]
    train_data = load_csv_files(data_dirs[:4], batch_size=batch_size)
    test_data = load_csv_files(data_dirs[4:], batch_size=batch_size)

    print(f"Training data size: {len(train_data)}")
    print(f"Testing data size: {len(test_data)}")

    # Choose Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")

    print("Building neural network...")
    model = DixonNet().to(device)

    print("Creating loss function and optimiser...")
    lossfn = torch.nn.BCEWithLogitsLoss()
    optimr = torch.optim.Adam(model.parameters(), lr=learn_rate)
    schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimr)

    # Train Model
    for epoch in range(1, epochs + 1):

        # Save the current learning rate
        lr = [group["lr"] for group in optimr.param_groups]
        if len(lr) == 1:
            lr = lr[0]

        print("Training...")
        start_time = time.time()
        train_loss = train(model, train_data, lossfn, optimr, schdlr, device)        
        train_time = timedelta(seconds=time.time() - start_time)

        print(
            " | ".join(
                (
                    f"Epoch {epoch}",
                    f"LR: {lr:.4e}",
                    f"Loss: {train_loss:.4e}",
                    f"Time: {train_time}",
                )
            )
        )

        if epoch % 10 == 0:
            print("Testing...")
            start_time = time.time()
            eval_loss, table = evaluate(model, test_data, lossfn, device)
            eval_time = timedelta(seconds=time.time() - start_time)

            print(
                " | ".join(
                    (
                        f"Loss: {eval_loss:.4e}",
                        f"Acc: {100*table.accuracy():.2f}%",
                        f"Time: {eval_time}",
                    )
                )
            )

            # check output directory
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)

            # save model parameters
            torch.save(model.state_dict(), output_dir / f"{epoch + 1}.params")


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
