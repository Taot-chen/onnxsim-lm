这是一个将大语言模型 chatglm2_6b 转换成 onnx 模型，并使用本项目源码调用，将对该 onnx 模型做 onnxsim 优化的示例。

## 1 下载大语言模型 chatglm2_6b 的预训练权重

```bash
chmod +x download_chatglm2_6b.sh
./download_chatglm2_6b.sh
```
脚本执行完成之后，大语言模型 chatglm2_6b 的预训练权重会被下载保存到当前路径下的`chatglm2-6b`。



## 2 转 onnx

```bash
python3 export_torch2onnx.py --torch_model=/path/to/chatglm2_6b_path/ --seq_len=128 --block_num=1 --batch_size=1 --iteration=0 --out_dir=./onnx/
```

说明：
* `--torch_model`：前面下载的大语言模型 chatglm2_6b 的预训练权重的路径
* `--seq_len`：大语言模型 chatglm2_6b转出的 onnx 静态图的上下文长度
* `--block_num`：大语言模型 chatglm2_6b 在转 onnx 的时候所包含的 decoder 模块的数量
* `--iteration`：0 代表第一次迭代，1 代表非第一次迭代
* `--out_dir`：存放 onnx 的路径
