# A deep learning approach to classify sitting and sleep history from raw accelerometry data during simulated driving

*by Georgia A. Tuckwell¹, James A. Keal², Charlotte C. Gupta¹, Sally A. Ferguson¹, Jarrad D. Kowlessar³, and Grace E. Vincent¹*

 - ¹ Central Queensland University, Appleton Institute, School of Health, Medical and Applied Sciences, Adelaide 5001, Australia.
 - ² The University of Adelaide, School of Physical Sciences, Adelaide 5005, South Australia, Australia.
 - ³ Flinders University, College of Humanities and Social Sciences, Adelaide 5001, South Australia, Australia.

This repository contains the official implementation of the article [A deep learning approach to classify sitting and sleep history from raw accelerometry data during simulated driving](https://www.mdpi.com/1424-8220/22/17/6598). The accelerometer recordings used in the publication are considered medical data and cannot be openly distributed. For this reason, the data has not been included in this repository.



## Requirements

Installation using [Conda](https://www.anaconda.com/) is recommended.

 - python 3
 - [pytorch](https://pytorch.org/docs/stable/index.html)
 - [torchvision](https://pytorch.org/vision/stable/index.html)
 - [scikit-learn](https://scikit-learn.org/stable/index.html)
 - [seaborn](https://seaborn.pydata.org/)
 - [tqdm](https://tqdm.github.io/)

 The following commands install the above requirements with CUDA support:

```sh
conda install pytorch torchvision cudatoolkit -c pytorch
```

```sh
conda install scikit-learn seaborn tqdm
```

More PyTorch installation options, including CPU only installation, can be found [here](https://pytorch.org/get-started/locally/).

## Usage

#### Data

Data must be provided as CSV files in long format, with rows representing an ordered sequence of observations made at consecutive time intervals - E.G. Accelerometer data taken during one session of a driving simulator. Recordings taken independently should be separate CSV files.

The class assiciated with each CSV file is indicated by the name of the file's parent directory. Specifically,

 - `DBL` = Broken up activity, Long sleep opportunity
 - `DBR` = Broken up activity, Restricted sleep opportunity
 - `DSL` = Sedentary activity, Long sleep opportunity
 - `DSR` = Sedentary activity, Restricted sleep opportunity

For example:

```
data/
├── DBL/
│   ├── xxx.csv
│   ├── xxy.csv
│   └── ...
├── DBR/
│   ├── 123.csv
│   ├── nsdf3.csv
│   └── ...
├── DSL/
│   ├── xxz.csv
│   ├── xyx.csv
│   └── ...
└── DSR/
    ├── 456.csv
    ├── asd932_.csv
    └── ...
```

The columns of the CSV file to be used may be specified using the arguments provided to the `load_csv_files` function. Other details pertaining to your data should be specified in the `CSVFile` class. Both can be found in `utils/data.py`.

A custom binary label function may be created with the signature `Callable[[CSVFile], bool]` and used in `train.py` where appropriate.

#### Training

```sh
python train.py --model-name {dixonnet, resnet18}
```

The expected output is a directory containing a file named `log.ndjson`. This file describes the training process and can be plotted:

```sh
python plot_training_logs.py log1.ndjson log2.ndjson ...
```

#### Class Activation Maps

Also found in the training output directory is a subdirector named `params`. This contains the trained parameter states of the neural network that were saved during the training process.

Class activation maps may be produced using

```sh
python plot_class_activation_maps.py --model-name {dixonnet, resnet18} /path/to/params/
```

## Licence

All source code is made available under a BSD 3-clause license. You may freely use and modify this code, without warranty, so long as you provide attribution to the authors. The manuscript text is not open source. The article's content has been published in the journal [Sensors](https://www.mdpi.com/journal/sensors).

## Citation

 > Georgia A. Tuckwell¹, James A. Keal², Charlotte C. Gupta¹, Sally A. Ferguson¹, Jarrad D. Kowlessar³, & Grace E. Vincent¹. "A deep learning approach to classify sitting and sleep history from raw accelerometry data during simulated driving", Sensors, 2022, 22, DOI: 10.3390/s22176598
