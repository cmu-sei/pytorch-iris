import os
import subprocess
from time import sleep

Iris_path = \
  subprocess.check_output("git rev-parse --show-toplevel",shell=True). \
  strip().decode('utf-8')
ESP_path = os.environ["ESP_ROOT"]
hls4ml_model_path = os.path.join(Iris_path,"model/hls4ml_project")
run_script_template_path = os.path.join(Iris_path,
                                        "scripts/run_accgen_template.sh")
run_script_path = os.path.join(Iris_path,"scripts/run_accgen.sh")

print("ESP Path: " + ESP_path)
print("hls4ml Model Path: " + hls4ml_model_path)

# create run script based on template with hls4ml model path
with open(run_script_template_path,"r") as run_script_template:
  lines = run_script_template.readlines()
  for i in range(0, len(lines)):
    if "HLS4ML_PATH" in lines[i]:
      lines[i] = hls4ml_model_path + "\n"
with open(run_script_path,"w") as run_script:
  run_script.writelines(lines)
os.chmod(run_script_path,0o755)

os.chdir(ESP_path)
os.system(run_script_path)

# modify espacc.cc to use new hls4ml API
espacc_path = os.path.join(ESP_path,
                           "accelerators/hls4ml/iris_hls4ml/hw/src/espacc.cc")
line_modifications = [
  ["unsigned short size_in1, size_out1;", "//"],
  ["myproject(_inbuff, _outbuff, size_in1, size_out1);",
   "myproject(_inbuff, _outbuff);"]]
with open(espacc_path,"r") as espacc:
  lines = espacc.readlines()
  for i in range(0, len(lines)):
    for j in range(0, len(line_modifications)):
      if line_modifications[j][0] in lines[i]:
        lines[i] = lines[i].replace(line_modifications[j][0],
                                    line_modifications[j][1])
with open(espacc_path,"w") as espacc:
  espacc.writelines(lines)
