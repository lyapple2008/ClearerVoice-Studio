## 任务
基于pure c runtime实现ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])的推理过程

## 要求
1. 纯c/c++实现
2. 支持不同的平台加速特性，比如macos环境metal/coreml，windows环境cuda
3. 尽量使用现有的推理引擎，避免从零开始实现

## 验证
1. pure c runtime实现能够正确的运行输出结果
2. 对于同样的输入，pure c runtime的推理结果与当前python环境推理结果一致