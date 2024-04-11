# Two layer neural net with PyTorch

```
# do once
conda create -n Iris -c pytorch python==3.11
conda activate Iris
conda install \
      numpy \
      matplotlib \
      pytorch::pytorch \
      scikit-learn \
      tqdm
conda install -c conda-forge \
      torchview

# do always
conda activate Iris
python train.py
python infer.py
```

![Neural network](./iris.png)

# for use with hls4ml
```
git clone git@github.com:cmu-sei/hls4ml-docker.git
cd hls4ml-docker/work
git clone git@github.com:cmu-sei/pytorch-iris.git
git checkout hls4ml

# see hls4ml-docker README.md for building and running container
# in the hls4ml container
cd work/pytorch-iris/
export PYTHONPATH=`pwd`
python train.py
python ./scripts/build_hls4ml_model.py
# on host
cp -r work/pytorch-iris /path/to/esp-docker/work
# in the esp container
python ./scripts/create_esp_accelerator.py
# now the accelerator can be added to the esp flow in the usual way.
# copy soc directory, make iris_hls4ml-hls, make esp-xconfig
```
