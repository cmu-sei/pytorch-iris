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


import copy
import model as m
import torch
from torch.ao.quantization import get_default_qconfig_mapping, QConfigMapping
from torch.ao.quantization.quantize_fx import convert_fx, fuse_fx, prepare_fx
from torch.autograd import Variable
import os

def write_matrix_to_file(filename, variable):
    file_delimiter = " "
    rows, columns = variable.shape
    file_str = [""]*rows
    for row in range(0, rows):
        if row > 0:
            file_str[row] += "\n"
        for column in range(0, columns):
            if column > 0:
                file_str[row] += file_delimiter
            file_str[row] += str(variable[row][column].item())
    with open(filename, "w") as w_file:
        w_file.writelines(file_str)

def main():
    print("PyTorch Iris train")
    print(torch.backends.quantized.supported_engines)

    verbose = False
    X,y,names,feature_names = m.load_data()
    if verbose:
        m.vis_data(X,y,names,feature_names)
    X_train,X_test,y_train,y_test = m.split_data(X,y)
    
    # make an output directory
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # fp32
    quantize = False
    model_fp32 = m.train_model(X_train,X_test,y_train,y_test,quantize,verbose)
    if verbose:
        m.plot_roc_curve(model_fp32,X_test,y_test,"fp32")
    m.save_model(model_fp32, os.path.join(data_dir, "model_fp32.pt"), quantize)

    # quantized to int8 with quantized aware training using graph_fx
    isBuildingQuantizedModel = False
    if isBuildingQuantizedModel:
        quantize = True
        model_int8 = m.train_model(X_train,X_test,y_train,y_test,quantize,verbose)
        if verbose:
            m.plot_roc_curve(model_int8,X_test,y_test,"int8")
        m.save_model(model_int8, os.path.join(data_dir, "model_int8.pt"), quantize)

    # input data and output predictions
    X_test  = Variable(torch.from_numpy(X_test)).float()
    y_pred = model_fp32(X_test)

    # sanity check
    y_test_pred = torch.argmax(y_pred, dim=1)
    y_test = Variable(torch.from_numpy(y_test)).long()
    model_accuracy = sum(y_test_pred == y_test) / len(y_test)
    print("accuracy check: " + str(model_accuracy.item()))

    # write input data and output predictions to a file
    write_matrix_to_file(os.path.join(data_dir, "Input.dat"), X_test)
    write_matrix_to_file(os.path.join(data_dir, "Output.dat"), y_pred)

if __name__ == "__main__":
    main()
