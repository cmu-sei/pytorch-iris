#!/bin/bash

wget -O Makefile https://github.com/pnnl/soda-opt/raw/main/docs/tutorials/pytorch/matmul_accel_gen/docker-version/Makefile
patch -l Makefile < Makefile.patch
sed -e '1,29d' Makefile > Makefile.tmp
mv Makefile.tmp Makefile
