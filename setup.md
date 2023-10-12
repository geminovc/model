This file describes the necessary processes involved in getting the model up and running on a new machine. Unfortunately,
I didn't do a good job with package management, so we use a combination of `conda` and `pip` here. 
## Mounting (only relevant if you're using a csail machine and need data/storage):
```bash
sudo mkdir /video-conf
sudo apt-get install nfs-common
sudo mount -t nfs ${mount_address} /video-conf
```

## Libopus/libvpx dependencies for aiortc:
```bash
sudo apt-get install libopus-dev libvpx-dev
```

## Getting Conda Environment:
Get latest version of conda from https://www.anaconda.com/products/distribution
```bash
cp /video-conf/scratch/vibhaa_tardy/nets_fom.txt nets_fom.txt
```

If python default in conda is different from what’s in the file, please change the line in the file accordingly. For that, figure out what the default python version is with the conda installation, then https://anaconda.org/anaconda/python/files?sort=basename&sort_order=desc&version=3.10.9 should help you find the python link for your version. Once edited, run
```bash
conda create --prefix /home/vibhaa/anaconda3/envs/vibhaa_nets_fom --file nets_fom.txt
```

## Getting aiortc
First, change home directory as need be; this assumes `aiortc` is in the home directory
Setup paths for aiortc import :
```bash
export PYTHONPATH=~/aiortc/gemino_model:~/aiortc/SwinIR
```

Compile aiortc inside conda environment: 
```bash
cd ~/aiortc
python setup.py install
```

## Pip for missing packages
There are still missing packages, so first get pip. Make sure the python command used here is coming from within conda by checking `which python`
```bash
wget https://bootstrap.pypa.io/get-pip.py
python get_pip.py
```

Make sure the pip used below is within conda by checking `which pip`. If it does not point to the one in the environment, reference it using
`${env_dir}/bin/pip` instead.
```bash
pip install pyyaml matplotlib scikit-image scikit-learn torch torchvision torchprofile pandas flow_vis lpips av protobuf==3.20.0 piq bitstring
```

## Code changes
This is due to versioning issues with python 3.9 and latest associated packages.
* `augmentation.py`: `from skimage.util import pad` &rarr; `from numpy import pad`
* `logger.py`: `import circle` &rarr; `import disk as circle`
* `logger.py`: `circle(kp[1], kp0..`' &rarr; `circle((kp[1], kp[0])..`
* `run.py`: `yaml.load` &rarr; `yaml.full_load`
* `measure_times.py`: `yaml.load` &rarr; `yaml.full_load`
* `utils.py`: comment out `from swinir_wrapper import ...`
