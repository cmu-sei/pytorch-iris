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
