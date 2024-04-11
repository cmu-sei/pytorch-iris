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
