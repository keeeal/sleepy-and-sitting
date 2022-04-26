from argparse import ArgumentParser
from itertools import chain, filterfalse
from multiprocessing import Pool
from typing import Sequence

import numpy as np
from scipy.fft import rfft
from seaborn import lineplot
from matplotlib import pyplot as plt
import torch

from utils.data import CSVFile, find_files, load_csv_files, SAMPLE_RATE
from utils.misc import true


def describe(s: Sequence[CSVFile]) -> str:
    """
    Create a string describing the data in multiple CSVFiles, including the
    mean and standard deviation of X, Y, and Z columns.
    """

    if not len(s):
        return "(N/A)"

    with torch.no_grad():
        data = torch.concat([csv_file.data for csv_file in s])
        x_mean, y_mean, z_mean = map(float, data.mean(dim=0))
        x_std, y_std, z_std = map(float, data.std(dim=0))

    out = [f"x̄={x_mean:.2f}", f"ȳ={y_mean:.2f}", f"z̄={z_mean:.2f}"]
    out += [f"σˣ={x_std:.2f}", f"σʸ={y_std:.2f}", f"σᶻ={z_std:.2f}"]
    return "(" + ", ".join(out) + ")"


def fft_of_csv(csv_file: CSVFile) -> tuple[list[float]]:
    """
    Perform a Fourier transform of a CSVFile. Returns a list of frequencies
    found in the data along with the log of the amplitude of each frequency
    for the X, Y, and Z columns.
    """

    x, y, z = np.array(csv_file.data).T

    freq_x = np.log(np.abs(rfft(x)))
    freq_y = np.log(np.abs(rfft(y)))
    freq_z = np.log(np.abs(rfft(z)))

    # The length of the CSVFile data is hard coded here rather than determined
    # from the data. This is because different file lengths result in different
    # Nyquist frequencies, which makes comparison / plotting difficult.
    # In practice, the CSVFile lengths only occasionally differ from this
    # length, and usually only differ by +/- one or two.
    n = 24000
    freq_domain = [i * SAMPLE_RATE / n for i in range(len(freq_x))]
    return list(freq_domain), list(freq_x), list(freq_y), list(freq_z)


def main(med_filt_size: int, low_pass_freq: float, plot_fft: bool):

    # load data
    print("\nLoading data...")
    files = load_csv_files(
        find_files(["data"], "csv"),
        label_fn=true,
        columns=(2, 3, 4),
        window_size=1,
        median_filter_kernel_size=med_filt_size,
        low_pass_frequency_hz=low_pass_freq,
    )

    print(f"Found {len(files)} data files {describe(files)}.")

    classes = {
        "day shift": list(filter(CSVFile.is_shift_day, files)),
        "day shift": list(filter(CSVFile.is_shift_day, files)),
        "night shift": list(filterfalse(CSVFile.is_shift_day, files)),
        "long sleep": list(filter(CSVFile.is_sleep_long, files)),
        "restricted sleep": list(filterfalse(CSVFile.is_sleep_long, files)),
        "broken activity": list(filter(CSVFile.is_activity_broken, files)),
        "sedentary activity": list(filterfalse(CSVFile.is_activity_broken, files)),
        "morning session": list(filter(CSVFile.is_session_morning, files)),
        "afternoon session": list(filterfalse(CSVFile.is_session_morning, files)),
    }

    print()
    for key, value in classes.items():
        print(f"{len(value)} are {key} {describe(value)}.")

    days = [[f for f in files if f.day == n + 1] for n in range(5)]

    print()
    for n, day in enumerate(days):
        print(f"{len(day)} are day {n + 1} {describe(day)}.")

    if plot_fft:
        print("\nFourier transforming...")
        with Pool() as p:
            data = p.map(fft_of_csv, files)

        f, x, y, z = [list(chain(*i)) for i in zip(*data)]

        print("\nPlotting...")
        plot_data = dict(frequency=f, x_amplitude=x, y_amplitude=y, z_amplitude=z)
        ax = lineplot(data=plot_data, x="frequency", y="x_amplitude")
        ax.get_figure().savefig(f"FFT_mf{med_filt_size}_lp{low_pass_freq}_x.png")
        plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-mf", "--med-filt-size", type=int, default=0)
    parser.add_argument("-lp", "--low-pass-freq", type=float, default=0)
    parser.add_argument("-fft", "--plot-fft", action="store_true")
    main(**vars(parser.parse_args()))
