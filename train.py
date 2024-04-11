import copy
import model as m
import torch
from torch.ao.quantization import get_default_qconfig_mapping, QConfigMapping
from torch.ao.quantization.quantize_fx import convert_fx, fuse_fx, prepare_fx
from torch.autograd import Variable

def main():
    print("PyTorch Iris train")
    print(torch.backends.quantized.supported_engines)

    verbose = True
    X,y,names,feature_names = m.load_data()
    if verbose:
        m.vis_data(X,y,names,feature_names)
    X_train,X_test,y_train,y_test = m.split_data(X,y)

    # fp32
    quantize = False
    model_fp32 = m.train_model(X_train,X_test,y_train,y_test,quantize,verbose)
    if verbose:
        m.plot_roc_curve(model_fp32,X_test,y_test,"fp32")
    m.save_model(model_fp32,"model_fp32.pt",quantize)

    # quantized to int8 with quantized aware training using graph_fx
    quantize = True
    model_int8 = m.train_model(X_train,X_test,y_train,y_test,quantize,verbose)
    if verbose:
        m.plot_roc_curve(model_int8,X_test,y_test,"int8")
    m.save_model(model_int8,"./model_int8.pt",quantize)

if __name__ == "__main__":
    main()
