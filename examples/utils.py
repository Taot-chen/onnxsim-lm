from transformers import AutoModel
from onnx_utils import set_onnx_input_shape
from compress_model import SIZE_1MB, compress_onnx_model, uncompress_onnx_model
from onnxsim import simplify
import torch
import os
import copy
import torch._C._onnx as _C_onnx
from torch.onnx import _constants
import onnx

def load_torch_llm(model_path):
    return AutoModel.from_pretrained(
        model_path,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

def print_run_cmd(cmd, run=1, p=1):
    if p:
        print("\033[36m>> cmd: {}\033[0m".format(cmd))
    if run:
        os.system(cmd)

def get_params_dict_size(model, args, verbose, do_constant_folding, input_names, output_names):
    training=_C_onnx.TrainingMode.EVAL
    operator_export_type=_C_onnx.OperatorExportTypes.ONNX
    val_do_constant_folding = torch.onnx.utils._decide_constant_folding(
        do_constant_folding, operator_export_type, training
    )

    graph, params_dict, torch_out = torch.onnx.utils._model_to_graph(
        model,
        args,
        verbose,
        input_names,
        output_names,
        operator_export_type,
        val_do_constant_folding,
        fixed_batch_size=False,
        training=training,
        dynamic_axes={}
    )
    params_dict = torch._C._jit_pass_onnx_deduplicate_initializers(  # type: ignore[assignment]
        graph, params_dict, getattr(model, "training", False)  # type: ignore[arg-type]
    )
    size = 0
    for key in params_dict.keys():
        shape = params_dict[key].shape
        dtype = 0
        if params_dict[key].dtype == torch.int8:
            dtype=1
        elif params_dict[key].dtype == torch.float16:
            dtype=2
        elif params_dict[key].dtype == torch.float32:
            dtype=4
        elif params_dict[key].dtype == torch.float64:
            dtype=8
        elif params_dict[key].dtype == torch.int64:
            dtype=8
        tmp_size = 1
        for cnt in range(len(shape)):
            tmp_size = tmp_size * shape[cnt]
        size += tmp_size * dtype
    return (size, params_dict, graph)

def get_params_size(model, args, do_constant_folding, input_names, output_names, out_dir, verbose, opset_version,
                             keep_initializers_as_inputs=None, custom_opsets=None, export_modules_as_functions=False):
    (params_dict_size, params_dict, graph) = get_params_dict_size(model, args, verbose, do_constant_folding, input_names, output_names)
    if opset_version is None:
        opset_version = _constants.ONNX_DEFAULT_OPSET
    dynamic_axes = {}
    operator_export_type=_C_onnx.OperatorExportTypes.ONNX
    add_node_names = True
    model_file_location = out_dir + "/" + "model_file_location/tmp.onnx"
    os.makedirs(out_dir + "/" + "model_file_location/", exist_ok=True)
    module_typenames_to_export_as_functions: Set[str] = set()
    if isinstance(model, (torch.nn.Module, torch.jit.ScriptModule)):
        module_typenames_to_export_as_functions = torch.onnx.utils._setup_trace_module_map(
            model, export_modules_as_functions
        )
    node_attr_to_name = {}  # type: ignore[var-annotated]
    if module_typenames_to_export_as_functions:
        # NOTE: cannot call DCE after this pass. DCE will remove function definition nodes.
        node_attr_to_name = torch.onnx._C._jit_pass_onnx_function_extraction(
            graph,
            module_typenames_to_export_as_functions,
            list(params_dict.keys()),
        )
    if custom_opsets is None:
        custom_opsets = {}
    val_keep_init_as_ip = torch.onnx.utils._decide_keep_init_as_input(
        keep_initializers_as_inputs,
        operator_export_type,
        opset_version,
    )
    val_add_node_names = torch.onnx.utils._decide_add_node_names(
        add_node_names, operator_export_type
    )
    (
        proto,
        export_map,
        val_use_external_data_format,
        node_names,
    ) = graph._export_onnx(  # type: ignore[attr-defined]
        {},
        opset_version,
        dynamic_axes,
        False,
        operator_export_type,
        not verbose,
        val_keep_init_as_ip,
        custom_opsets,
        val_add_node_names,
        model_file_location,
        node_attr_to_name,
    )
    cmd = "rm -rf " + out_dir + "/" + "model_file_location/"
    print_run_cmd(cmd, run=1)
    proto_size = len(proto)
    return (params_dict_size, proto_size)

def use_dummy_blockss(model, args, do_constant_folding, input_names, output_names, out_dir, verbose, opset_version,
                             keep_initializers_as_inputs=None, custom_opsets=None, export_modules_as_functions=False):

    (params_dict_size, proto_size) = get_params_size(model, args, do_constant_folding, input_names, output_names, \
        out_dir, verbose, opset_version, keep_initializers_as_inputs=None, custom_opsets=None, export_modules_as_functions=False)

    CONSTANT_2_GB = 2 ** 31
    if params_dict_size <= CONSTANT_2_GB and (params_dict_size + proto_size) >= CONSTANT_2_GB:
        return (True, CONSTANT_2_GB - params_dict_size)
    else:
        return (False, CONSTANT_2_GB - params_dict_size)


def use_dummy_layers(model, args, do_constant_folding, input_names, output_names, out_dir, verbose, opset_version,
                             keep_initializers_as_inputs=None, custom_opsets=None, export_modules_as_functions=False):

    (params_dict_size, proto_size) = get_params_size(model, args, do_constant_folding, input_names, output_names, \
        out_dir, verbose, opset_version, keep_initializers_as_inputs=None, custom_opsets=None, export_modules_as_functions=False)

    CONSTANT_2_GB = 2 ** 31
    if params_dict_size <= CONSTANT_2_GB and (params_dict_size + proto_size) >= CONSTANT_2_GB:
        return (True, CONSTANT_2_GB - params_dict_size)
    else:
        return (False, CONSTANT_2_GB - params_dict_size)

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

def simplify_large_onnx(in_model_path, block_num, iteration, out_dir="./onnx/", dtype=torch.float32):
    out_model_dir_path = out_dir + "/" + 'iter' + str(iteration) + "_" + str(block_num) + "blocks_sim" + "/"
    os.makedirs(out_model_dir_path, exist_ok=True)
    out_model_path = out_model_dir_path + '/' + 'iter' + str(iteration) + "_" + str(block_num) + "blocks_sim.onnx"
    _simplify_large_onnx(in_model_path, out_model_path)


def export(torch_model_path, onnx_path, onnx_fpath, iteration, sequence_length, batch_size, block_num):
    torch_model = load_torch_llm(torch_model_path)
    os.makedirs(onnx_path + "/iter" + str(iteration) + "_" + str(block_num) + "blocks/", exist_ok = True)
    inputs = None
    input_names = None
    output_names = None
    
    if iteration:
        sequence_length -= 1
        past_seq_len = sequence_length
        input_ids = torch.randint(low=0, high=100, size=(batch_size, sequence_length), dtype=torch.int64)
        labels = None
        use_cache = True
        output_attentions = None
        output_hidden_states = None
        return_dict = False
        return_last_logits = False
        inputs_embeds = None
        output_names = ["logits"]
        num_past_key_values = 0
        if hasattr(torch_model.config, "multi_query_group_num"):
            num_past_key_values = torch_model.config.multi_query_group_num
        attention_mask_one = torch.ones((batch_size, past_seq_len), dtype=torch.int64)
        attention_mask_zero = torch.zeros((batch_size, 1), dtype=torch.int64)
        attention_mask = torch.cat((attention_mask_one, attention_mask_zero), dim=1)
        position_ids = torch.cumsum(attention_mask, dim=-1) - 1
        position_ids = position_ids[:,]
        input_names = ["input_ids", "position_ids", "attention_mask", "past_key_values"]
        past_key_values = torch.randn(block_num, 2, 1, batch_size, num_past_key_values, 128, dtype=torch.float32)
        
        inputs = (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            return_last_logits,
        )
    else:
        output_names = ["logits" , "past_key_values"]
        input_ids = torch.randint(low=0, high=100, size=(batch_size, sequence_length), dtype=torch.int64)
        attention_mask = None
        past_key_values_tuple = None
        inputs_embeds = None
        labels = None
        use_cache = True
        output_attentions = None
        output_hidden_states = None
        return_dict = False
        return_last_logits = False
        position_ids = None
        input_names=["input_ids"]
        inputs = (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values_tuple,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            return_last_logits
        )
    do_constant_folding = True
    verbose = False
    opset_version = None
    _use_dummy_layers, _dummy_size = use_dummy_layers(copy.deepcopy(torch_model), inputs, do_constant_folding, input_names, \
        output_names, onnx_path, verbose, opset_version)
    if _use_dummy_layers:
        torch_model = add_dummy_layers(torch_model, _dummy_size)
    torch.onnx.export(
        torch_model,
        inputs,
        onnx_fpath,
        input_names=input_names,
        output_names=output_names,
    )