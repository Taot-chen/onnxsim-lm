## 0 说明

大模型的部署有时候需要转成 onnx，之后再基于 onnx 做后续的部署工作。对于比较大的模型，在 onnxsim 的时候会很耗时，并且内存占用也非常多。

在大语言模型中，通常会出现大于 2GB 的模型。对于大于 2GB 的模型，在 onnxsim 的时候还很容易遇到`Model Proto`或者`Tensor Proto`超过 2GB 的报错。

这里针对大模型部署过程中可能遇到这样的问题给出了一个优化方法：

1）将网络中的卷积和矩阵乘的权重（参数量大于某个阈值）替换为ConstantOfShape，从而显著缩小模型大小
2）利用onnxsim特性避免折叠（参数量大于某个阈值）ConstantOfShape算子
3）对压缩后的模型进行优化和常量折叠后的模型删除ConstantOfShape算子，并替换为原来的权重。

该仓库对上面的方案进行了实现，并发布了相应的 python package。

## 1 安装

可以直接通过`pip`安装：
```bash
pip install onnxsim-lm
```

也可以通过源码安装：
```bash
# clone 代码
git clone https://github.com/Taot-chen/onnxsim-lm.git
cd onnxsim-lm
python3 setup.py bdist_wheel --universal
python3 -m pip install ./dist/*.whl
```


## 2 使用

### 2.1 命令行调用

```bash
onnxsimlm --help

# --onnx_input and --out_dir shuld end with .onnx
onnxsimlm --onnx_input=/path/to/input_onnx --out_dir=/path/to/output_onnx
```

### 2.2 python api 调用

```bash
import onnxsimlm
in_onnx = "examples/example.onnx"
out_dir = "test_out"
onnxsimlm.utils.simplify_large_onnx(in_onnx, out_dir)
```
