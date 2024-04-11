import copy
import model as m
import numpy as np
from sklearn.preprocessing import StandardScaler
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

    file_str.append("\n") # add newline to end of file
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
    print("test set accuracy check: " + str(model_accuracy.item()))

    # write input data and output predictions to a file for all data
    X = StandardScaler().fit_transform(X)
    X = Variable(torch.from_numpy(X)).float()
    y_pred_X = model_fp32(X)
    write_matrix_to_file(os.path.join(data_dir, "Input.dat"), X)
    write_matrix_to_file(os.path.join(data_dir, "Output.dat"), y_pred_X)

if __name__ == "__main__":
    main()
