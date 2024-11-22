from onnxsim-lm.utils import simplify_large_onnx
import argparse

def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_input", help="ONNX model to simplify")
    parser.add_argument("--out_dir", default="./onnx/", help="output dir to save simplified onnx model")
    args = parser.parse_args()

    assert (args.onnx_input[-5:] == ".onnx"), f"param onnx_input must end with .onnx"
    simplify_large_onnx(args.onnx_input, args.out_dir)

if __name__ == "__main__":
    run()
