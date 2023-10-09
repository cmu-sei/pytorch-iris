print("Loading libraries")

from sklearn.metrics import accuracy_score
import model as m
import torch
from torch.autograd import Variable
import numpy as np
import hls4ml
import os
import shutil
import subprocess

# move to top level of git repository
Iris_path = \
  subprocess.check_output("git rev-parse --show-toplevel",shell=True). \
  strip().decode('utf-8')
os.chdir(Iris_path)

# delete previous model
output_dir = "model/hls4ml_project"
if os.path.exists(output_dir):
  print("Deleting existing hls4ml model")
  shutil.rmtree(output_dir)

print("Loading data")

# model files
data_dir = "data"
model = "model_fp32.pt"
data_files = [
   os.path.join(data_dir,"Input.dat"),
   os.path.join(data_dir,"Output.dat")]

# data
X,y,names,feature_names = m.load_data()
X_train,X_test,y_train,y_test = m.split_data(X,y)

# floating point model
print("Loading model: " + model)
model = torch.load(os.path.join(data_dir, model))
# predictions
X_test_torch  = Variable(torch.from_numpy(X_test)).float()
y_torch = model(X_test_torch).detach().numpy()
# accuracy
print("PyTorch Model Accuracy: {}".format(
   accuracy_score(y_test, np.argmax(y_torch, axis=1))))
isPlottingRocCurve = False
if isPlottingRocCurve:
   m.plot_roc_curve(model,X_test,y_test)

# hls4ml model 
# configuration
isUsingCustomPrecision = False
if isUsingCustomPrecision:
   precision_str = "64,6"
   config = hls4ml.utils.config_from_pytorch_model(
      model,
      granularity='name',
      default_precision='ap_fixed<'+ precision_str +'>')
   config.update(
      {
         'LayerName' : {
            'relu' : {
               'table_t' : 'ap_fixed<'+ precision_str +'>'
            },
            'softmax' : {
               'table_t'     : 'ap_fixed<'+ precision_str +'>',
               'exp_table_t' : 'ap_fixed<'+ precision_str +',AP_RND,AP_SAT>',
               'inv_table_t' : 'ap_fixed<'+ precision_str +',AP_RND,AP_SAT>'
            },
         }
      })
else:
   config = hls4ml.utils.config_from_pytorch_model(
      model,
      granularity='name')
isPrintingConfiguration = False
if isPrintingConfiguration:
   # Uses "plotting.py" from hls4ml-tutorial
   import plotting
   print("-----------------------------------")
   print("Configuration")
   plotting.print_dict(config)
   print("-----------------------------------")
# conversion
hls_model = hls4ml.converters.convert_from_pytorch_model(
   model,
   input_shape=[[None,4]],
   hls_config=config,
   output_dir=output_dir, part='xcvu9p-flga2104-2-i',
   input_data_tb=data_files[0], 
   output_data_tb=data_files[1],
)
# model architecture
isSavingModelArchitecture = False
if isSavingModelArchitecture:
   hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file="./data_flow.png")
# compilation
hls_model.compile()
# predictions
X_test_hls = np.ascontiguousarray(X_test)
y_hls = hls_model.predict(X_test_hls)
# accuracy
print("hls4ml Model Accuracy: {}".format(
   accuracy_score(y_test, np.argmax(y_hls, axis=1))))

isPrintingPredictions = False
if isPrintingPredictions:
   print(y_torch)
   print(y_hls)

print("Running Vivado HLS C Simulation")
cmd = \
  "cd " + output_dir + "; " + \
  "vivado_hls -f build_prj.tcl 'csim=1 synth=0 cosim=0 validation=0'"
os.system(cmd)
