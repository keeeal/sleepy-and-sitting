import os, time, csv, random, pickle
from pathlib import Path
from datetime import timedelta
from itertools import cycle

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
    WeightedRandomSampler,
    Sampler,
    ConcatDataset
)
from torchvision import datasets

import torch.nn.functional as F


class CustomCSVDataset(Dataset):
    def __init__(self, data_folder):
        self.data = []
        self.window_size = 4096
        Label1 = 0
        Label2 = 0
        for folder in os.listdir(data_folder):
            # print(folder)
            for file in os.listdir(Path(data_folder, folder)): #[:1]
                with open(data_folder / folder / file, newline="") as f:
                    for n, line in enumerate(csv.reader(f)):
                        xyz = [float(i) for i in line[2:5]]
                        #DBL = Active9H, DBR = Active5H, DSL = Sedentary9H, DSR = Sedentary5H
                        if len(line) == 6 and line[-1] == "DRIVING":
                            if folder in ("DBL", "DSL"):
                                self.data.append((xyz, 1))
                                Label1 += 1
                            # elif folder == 'DBR' or folder == 'DSR':
                            elif folder in ("DBR", "DSR"):
                                self.data.append((xyz, 0))
                                Label2 += 1
                            else:
                                print("ERROR IN THE: CustoneCSVDataset _init_")
                        else:
                            print(
                                f"Unexpected length {len(line)} on line {n} of {file}."
                            )

        print("Dataset Contructed!")
        print("Label 1 Count: " + str(Label1))
        print("Label 2 Count: " + str(Label2))

        # todo: remove file crossover points
        # todo: remove windows if entire window not one label

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        item, label = zip(*self.data[idx : idx + self.window_size])
        label = max(set(label), key=label.count)
        return torch.tensor(item, dtype=torch.float32), torch.tensor(
            label, dtype=torch.float32
        )


class BalancedSampler(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.indices = {}
        for idx, (data, label) in enumerate(data_source):
            self.indices[label] = self.indices.get(label, []) + [idx]

        if shuffle:
            for key in self.indices:
                random.shuffle(self.indices[key])

        self.labels = list(self.indices)
        self.length = len(max(self.indices.values(), key=len)) * len(self.labels)
        self.indices = {key: cycle(value) for key, value in self.indices.items()}
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter < self.length:
            result = next(self.indices[self.labels[self.counter % len(self.labels)]])
            self.counter += 1
            return result
        else:
            raise StopIteration

    def __len__(self):
        return self.length


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(240, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = x.flatten(1)
        logits = self.linear_relu_stack(x)
        return logits

class DixonNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=100, kernel_size=16),
            nn.ReLU(),
            nn.Conv1d(100, 100, 16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(100, 100, 16),
            nn.ReLU(),
            nn.Conv1d(100, 100, 16),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.features(x)
        x = nn.functional.avg_pool1d(x, kernel_size=x.shape[2]) # global avg
        x = x.squeeze(2)
        x = self.classifier(x)
        return x


def train(model, data, lossfn, optimr, schdlr, device, epoch=0, verbose=False):
    print("Training...")
    model.train()
    start = time.time()

    _data = [i for i, _ in zip(data, range(256))]

    for x, y in tqdm(_data):
        optimr.zero_grad()
        x, y = x.to(device), y.to(device)
        loss = lossfn(model(x).squeeze(), y)
        loss.backward()
        optimr.step()
        del x, y, loss

    train_time = timedelta(seconds=time.time() - start)
    lr = [group["lr"] for group in optimr.param_groups]
    if len(lr) == 1:
        lr = lr[0]

    # get training loss
    losses = []
    model.eval()
    with torch.no_grad():
        for x, y in _data:
            x, y = x.to(device), y.to(device)
            loss = lossfn(model(x).squeeze(), y)
            losses.append(loss.item())
            del x, y, loss

    loss = sum(losses) / len(losses)
    schdlr.step(loss)

    # report
    if verbose:
        print(
            " | ".join(
                (
                    f"Epoch {epoch}",
                    f"LR: {lr:.4e}",
                    f"Loss: {loss:.4e}",
                    f"Time: {train_time}",
                )
            )
        )


def evaluate(model, data, lossfn, device, n_samples=1024):
    print("Testing...")
    model.eval()
    test_loss = 0
    falsePositive = 0
    falseNegative = 0
    truePositive = 0
    trueNegative = 0

    _data = [i for i, _ in zip(data, range(n_samples))]

    start = time.time()

    with torch.no_grad():
        for item, label in tqdm(_data):
            item, label = item.to(device), label.to(device)
            output = model(item)
            test_loss = lossfn(output.squeeze(), label)

            for prediction, truth in zip(output, label):
                if prediction < 0.5:
                    if truth < 0.5:
                        trueNegative += 1
                    else:
                        falseNegative += 1
                else:
                    if truth < 0.5:
                        falsePositive += 1
                    else:
                        truePositive += 1

    test_time = timedelta(seconds=time.time() - start)
    correct = truePositive + trueNegative
    test_loss /= n_samples


    # assert falseNegative + falsePositive + trueNegative + truePositive == n_data

    # print(f"Test set: Average loss: {test_loss:.4e}")
    # print(f"Accuracy: {correct}/{n_data} = {100*correct/n_data:.2f}%\n")

    print(
        " | ".join(
            (
                f"Loss: {test_loss:.4e}",
                f"Acc: {100*correct/n_samples:.2f}%",
                f"Time: {test_time}",
            )
        )
    )

    print(
        "TruePositives: "
        + str(truePositive)
        + " TrueNegatives: "
        + str(trueNegative)
        + " falsePositives: "
        + str(falsePositive)
        + " falseNegatives: "
        + str(falseNegative)
    )


def main(learn_rate, batch_size, epochs, output_dir):

    print("Loading data from file and pickle it...")

    data_paths = [Path("Data") / str(n + 1) for n in range(5)]
    loaded_data = [CustomCSVDataset(i) for i in data_paths]
    train_data = ConcatDataset(loaded_data[:4])
    test_data = ConcatDataset(loaded_data[4:])

    print("Training data sample count: " + str(len(train_data)))
    print("Testing data sample count: " + str(len(test_data)))

    # Unbalanced data loading:
    train_data = DataLoader(train_data, batch_size, shuffle=True)
    test_data = DataLoader(test_data, batch_size, shuffle=False)

    # Choose Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    # ChooseModel
    print("Building neural network...")
    model = DixonNet().to(device)

    print("Creating loss function and optimiser...")
    lossfn = torch.nn.BCEWithLogitsLoss()
    optimr = torch.optim.Adam(model.parameters(), lr=learn_rate)
    schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimr)

    # TrainModel
    for epoch in range(1, epochs + 1):
        train(model, train_data, lossfn, optimr, schdlr, device, epoch, verbose=True)

        if epoch % 10 == 0:
            evaluate(model, test_data, lossfn, device)

            # check output directory
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            # save model parameters
            torch.save(model.state_dict(), output_dir / f'{epoch + 1}.params')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-lr", "--learn-rate", type=float, default=1e-4
    ) # I CHANGED LORD KREELS DEFAULT FROM 1e-3
    parser.add_argument("-b", "--batch-size", type=int, default=124)
    parser.add_argument("-e", "--epochs", type=int, default=2000)
    parser.add_argument('-o', '--output-dir', type=Path, default='trained')
    main(**vars(parser.parse_args()))
