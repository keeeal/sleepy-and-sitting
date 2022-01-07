from os import listdir, mkdir
from os.path import isfile, isdir, join
from shutil import copy
from random import shuffle
from functools import partial

data = "DBL", "DBR", "DSL", "DSR"


def split_data(n, random=False):
    # check data directories are in the same place as this file
    assert all(d in listdir() for d in data), "Expected " + ", ".join(data)
    
    # create the destination directories
    for i in map(str, range(n)):
        assert not isdir(i), f"Directory already exists: {i}"
        mkdir(i)

        for d in data:
            if not isdir(join(i, d)):
                mkdir(join(i, d))

    for d in data:

        # find files and shuffle them
        files = list(filter(isfile, (join(d, f) for f in listdir(d))))
        if random:
            shuffle(files)
        
        # copy a portion of the files to each destination
        for i in range(n):
            list(map(partial(copy, dst=join(str(i), d)), files[i::n]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=5)
    parser.add_argument("--random", action='store_true')
    split_data(**vars(parser.parse_args()))
