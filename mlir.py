# Docker container for ESP use
# Copyright 2023 Carnegie Mellon University.
# MIT (SEI)
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# This material is based upon work funded and supported by the Department of
# Defense under Contract No. FA8702-15-D-0002 with Carnegie Mellon University
# for the operation of the Software Engineering Institute, a federally funded
# research and development center.
# The view, opinions, and/or findings contained in this material are those of
# the author(s) and should not be construed as an official Government position,
# policy, or decision, unless designated by other documentation.
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING
# INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON
# UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED,
# AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR
# PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE
# MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND
# WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
# [DISTRIBUTION STATEMENT A] This material has been approved for public release
# and unlimited distribution.  Please see Copyright notice for non-US
# Government use and distribution.
# DM23-0186


import argparse
import model as m
import os
from sklearn.preprocessing import StandardScaler
import torch
from torch.autograd import Variable
import torch_mlir

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch model to MLIR."
    )
    parser.add_argument("out_mlir_path", nargs="?",
                        default="./output/01_tosa.mlir",
                        help="Path to write the MLIR file to.")
    dialect_choices = ["tosa", "linalg-on-tensors", "torch", "raw", "mhlo"]
    parser.add_argument("--dialect",
                        default="linalg-on-tensors",
                        choices=dialect_choices,
                        help="Dialect to use for lowering.")
    args = parser.parse_args()
    return args

def main():
    print("PyTorch Iris trace")
    X,y,names,feature_names = m.load_data()
    X = StandardScaler().fit_transform(X)
    X = Variable(torch.from_numpy(X)).float()
    model = m.load_model("./model_fp32.pt")

    print(model)

    args = parse_args()
    os.makedirs(os.path.dirname(args.out_mlir_path), exist_ok=True)
    module = torch_mlir.compile(model, X, output_type=args.dialect,
                                use_tracing=True)
    with open(args.out_mlir_path, "w", encoding="utf-8") as outf:
        outf.write(str(module))

if __name__ == "__main__":
    main()
