from onnx_utils import set_onnx_input_shape
from compress_model import compress_onnx_model, uncompress_onnx_model
from onnxsim import simplify
import os
import onnx

def print_run_cmd(cmd, run=1, p=1):
    if p:
        print("\033[36m>> cmd: {}\033[0m".format(cmd))
    if run:
        os.system(cmd)

def _simplify_large_onnx(in_model_path, out_model_path):
    onnx_model = onnx.load(in_model_path)
    print(f"load model from {in_model_path} success")
    size_th_kb = 1024
    skip = ""
    save_extern_data = True

    size_th_bytes = size_th_kb * 1024

    onnx_model, removed_inits = compress_onnx_model(onnx_model, size_th_bytes=size_th_bytes)
    print(f"compress model success")

    onnx_model = set_onnx_input_shape(onnx_model, shape_cfg="")

    tensor_size_threshold = f"{size_th_kb}KB"
    skipped_optimizers = skip.split(";")
    onnx_model, check = simplify(onnx_model, skipped_optimizers=skipped_optimizers,
                                 tensor_size_threshold=tensor_size_threshold)
    if not check:
        raise ValueError(f"simplify compressed model {in_model_path} failed")

    print(f"simplify model success")

    onnx_model = uncompress_onnx_model(onnx_model, removed_inits)
    print(f"uncompress model success")

    save_extern = True if save_extern_data else False
    onnx.save(onnx_model, out_model_path, save_as_external_data=save_extern)

def simplify_large_onnx(in_model_path, out_dir="./onnx/"):
    os.makedirs(out_dir, exist_ok=True)
    in_onnx_name = in_model_path.split("/")[-1].split(".")[-2]
    if in_onnx_name is None:
        in_onnx_name = "onnx_simplified"
    else:
        in_onnx_name = in_onnx_name + "_simplified"
    out_model_path = out_dir + '/' in_onnx_name + ".onnx"
    _simplify_large_onnx(in_model_path, out_model_path)
