Copyright 2023 Carnegie Mellon University.
MIT (SEI)
Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
This material is based upon work funded and supported by the Department of
Defense under Contract No. FA8702-15-D-0002 with Carnegie Mellon University
for the operation of the Software Engineering Institute, a federally funded
research and development center.
The view, opinions, and/or findings contained in this material are those of
the author(s) and should not be construed as an official Government position,
policy, or decision, unless designated by other documentation.
NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING
INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON
UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED,
AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR
PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE
MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND
WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
[DISTRIBUTION STATEMENT A] This material has been approved for public release
and unlimited distribution.  Please see Copyright notice for non-US
Government use and distribution.
DM23-0186


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

# for use with soda-opt
```
git clone git clone --recursive git@github.com:cmu-sei/soda-opt-docker.git
cd soda-opt-docker
docker build --rm --pull -f ./Dockerfile -t soda-opt:dev-panda .
docker run --rm -it --network=host --privileged -e DISPLAY=$DISPLAY -e UID=$(id -u) -e GID=$(id -g) -v `pwd`/env:/home/soda-opt-user/env:rw -v `pwd`/work:/home/soda-opt-user/work soda-opt:dev-panda
# in the container
cd work/pytorch-iris/
./getmakefile.sh
make synth-baseline
```

For updating the Makefile patch. Makefile isn't version controlled, so this
is a bit odd. For minor changes directly edit Makefile.patch.
```
# get the soda opt version
wget -O Makefile.orig https://github.com/pnnl/soda-opt/raw/main/docs/tutorials/pytorch/matmul_accel_gen/docker-version/Makefile
# apply current patch
./getmakefile.sh
# edit Makefile to desired. Add the copyright so that it appears in the patch.
diff -u Makefile.orig Makefile > Makefile.patch
# edit the patch to change Makefile.orig to Makefile
# run patch to see if you get what you want
./getmakefile.sh
```
