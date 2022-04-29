# 项目描述
MMCLS files for train and inference ColorLine model。
用于训练/测试ColorLine标签分类模型的相关文件。

# 文件描述
## mmclassification相关文件（0.18.0版本）
1 mobilenet-v2_8xb32_frameH500.py
  配置文件，包括模型、优化器、数据集、日志及工作流的定义。
2 LaserLabel.py
  数据集定义，imagenet格式，定义分类的类别。
  
## 数据集预处理
  1 build_dataset.py： 
    1）从防伪区域图像LaserLabel中提取彩色线条区域ColorLine
    2）按面积进行筛选，得到ColorLine400数据集
    3）对清洗后的ColorLine400数据集，按标签号分为训练/验证/测试集，生成对应的TXT文件（图像路径及类别标注信息）。
  2 data_cleaner.py ： 
  对ColorLine400数据集进行数据清洗，删除模糊、相似图像。
## 模型转换
  ```remove_initializer_from_input.py： 作用于onnx格式模型文件，运行后生成新的onnx文件，调用时不会出现警告信息。```

# 使用方法
## 准备数据集
  1）使用build_dataset.py，提取ColorLine400数据集
  2）使用build_dataset.py，对ColorLine400数据集进行清洗
  3）再次运行build_dataset.py尾部代码，按标签号生成train/val/test.txt
## 配置mmclassification。
  1）添加自定义模型配置：将mobilenet-v2_8xb32_frameH500.py添加到mmclassification\configs\mobilenet_v2文件夹
  2）添加自定义数据库：将LaserLabel.py添加到mmclassification\mmcls\datasets，并修改init.py等相关文件
## 训练/测试模型。
  1）训练模型：
     python tools/train.py --config configs/mobilenet_v2/mobilenet-v2_8xb32_frameH500.py
  2) 测试模型:
     python tools/test.py --config configs/mobilenet_v2/mobilenet-v2_8xb32_frameH500.py --checkpoint tools/work_dirs/mobilenet-v2_8xb32_frameH500/epoch_300.pth --out results/result.json
## 模型转换为onnx：
    ```python tools/deployment/pytorch2onnx.py configs/mobilenet_v2/mobilenet-v2_8xb32_frameH500.py --checkpoint tools/work_dirs/mobilenet-v2_8xb32_frameH500/epoch_500.pth --output-file experiment/ColorLine400_epoch500.onnx --shape 192 256 --verify```
    ```python remove_initializer_from_input.py --input experiment/ColorLine400_epoch500.onnx --output experiment/ColorLine400_epoch500.onnx```





