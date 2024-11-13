import utils
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch_model" , default="./", type=str, help="torch model path")
    parser.add_argument("--seq_len" , default=512, type=int, help="input sequence length")
    parser.add_argument("--block_num" , default=2, type=int, help="block number")
    parser.add_argument("--batch_size" , default=1, type=int, help="batch size")
    parser.add_argument("--iteration" , default=0, type=int, help="iteration: 0, 1, 2(both 0 and 1)")
    parser.add_argument("--out_dir" , default="./onnx/", type=str, help="output dir")
    args = parser.parse_args()
    
    if args.iteration == 1 or args.iteration == 0:
        onnx_fpath = args.out_dir + "/" + "iter" + str(args.iteration) + "_" + str(args.block_num) + "blocks/" + "iter" + str(args.iteration) + "_" + str(args.block_num) + "blocks.onnx"
        utils.export(args.torch_model, args.out_dir, onnx_fpath, args.iteration, args.seq_len, args.batch_size, args.block_num)
        utils.simplify_large_onnx(onnx_fpath, args.block_num, args.iteration, args.out_dir)
    elif args.iteration == 2:
        onnx_fpath = args.out_dir + "/" + "iter" + str(1) + "_" + str(args.block_num) + "blocks/" + "iter" + str(1) + "_" + str(args.block_num) + "blocks.onnx"
        utils.export(args.torch_model, args.out_dir, onnx_fpath, 1, args.seq_len, args.batch_size, args.block_num)
        utils.simplify_large_onnx(onnx_fpath, args.block_num, 1, args.out_dir)
        
        onnx_fpath = args.out_dir + "/" + "iter" + str(0) + "_" + str(args.block_num) + "blocks/" + "iter" + str(0) + "_" + str(args.block_num) + "blocks.onnx"
        utils.export(args.torch_model, args.out_dir, onnx_fpath, 0, args.seq_len, args.batch_size, args.block_num)
        utils.simplify_large_onnx(onnx_fpath, args.block_num, 0, args.out_dir)
