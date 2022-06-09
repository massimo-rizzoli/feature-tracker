# Feature Tracker

## Introduction

This project is developed as the second assignment for the 2021-2022 Computer Vision course, in the Artificial Intelligence Systems master school at the University of Trento.
The objective is to experiment with different combinations of feature detection and feature tracking techniques.

## Installation

First of all, clone the repository and enter the folder.

```
git clone https://github.com/massimo-rizzoli/feature-tracker.git
cd feature-tracker
```

### Python Environment

This section is necessary only if you do not want to install the requirements in your global python environment.
If this is not the case you can skip to the [Install Requirements](#install-requirements) section.

#### With Pyenv (Recommended)

Create a python 3.9.7 virtual environment named `feature-tracker-env`

```
pyenv virtualenv 3.9.7 feature-tracker-env
```

Set the `feature-tracker-env` environment as the local environment for the `feature-tracker` folder (the environment will be automatically activated when entering the folder)

```
pyenv local feature-tracker-env
```

#### With Python Venv

The project was developed with `python 3.9.7`, so it is recommended to use this version.

Create a python virtual environment named `feature-tracker-env`

```
python -m venv feature-tracker-env
```

Activate the `feature-tracker-env` environment

```
source feature-tracker-env/bin/activate
```

**Note:** if using python venv, each time you open a shell, you will have to manually activate the environment as shown above, before being able to use the project.

### Update `pip`

```
python -m pip install -U pip
```

### Install Requirements

Install the requirements

```
pip install -r requirements.txt
```

## Usage

The structure is the following

```
python -m featuretrack [general options] <tracking algorithm> [tracking options] <feature detector> [detector options]
```

where a `feature detector` choice may be available depending on the tracking algorithm.

### Get Help

The following commands provide the lists of options with the relative descriptions.

To show the help message for the global options, run

```
python -m featuretrack --help
```

To show the help message for the tracking algorithm component, run

```
python -m featuretrack <tracking algorithm> --help
```

where `<tracking algorithm>` is either `siftbf`, `lk` or `kalman`.

To show the help message for the feature detector component, run

```
python -m featuretrack <tracking algorithm> <feature detector> --help
```

where `<feature detector>` is either `sift`, `gft`.

### Examples

#### Minimal

The following command will perform feature tracking using the Lucas-Kanade tracking algorithm and Good Features to Track feature points on the `video.mp4` video file located at `/path/to/`

```
python -m featuretrack \
       --video /path/to/video.mp4 \
       lk \
       gft
```

The following command will instead perform feature tracking and predict the next positions of feature points extracted by the SIFT algorithm

```
python -m featuretrack \
       --video /path/to/video.mp4 \
       kalman \
       sift
```

The following command will perform feature tracking by brute force matching SIFT feature points based on the L2 distance of the relative feature descriptors

```
python -m featuretrack \
       --video /path/to/video.mp4 \
       siftbf
```

**Note:** if the video is too large, you can use the global `--scale` option to resize it.


#### Detailed

The following command will perform feature tracking using the Lucas-Kanade tracking algorithm and Good Features to Track feature points on the `video.mp4` video file located at `/path/to/`, scaling the video respectively to 20% of height and 20% of width, waiting 16ms between each frame (in addition to processing time with the purpose of showing the processing at 30fps, otherwise it would be faster than real time), saving the output to `./results/out_video.avi` with the specified framerate of 30fps, tracking the points for 60 frames before resetting, detecting at most 100 points having at least 20% of the quality of the best point and having a distance of at least 10 pixels between eachother.

```
python -m featuretrack \
       --video /path/to/video.mp4 \
       --scale .2 .2 \
       --delay 16 \
       --output out_video \
       --framerate 30 \
       lk \
       --interval 60 \
       gft \
       --points 100 \
       --quality 0.2 \
       --distance 10
```
