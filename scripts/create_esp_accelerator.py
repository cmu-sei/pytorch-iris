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
                           "accelerators/hls4ml/pytorch-iris_hls4ml/hw/src/espacc.cc")
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
