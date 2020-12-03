# 写在前面

故事是这样的，我选了某著名水课，以为今年划划水就过去了，没想到老师反手一波背刺，突然给我布置作业：两周写一个Voice Activity Detection(VAD)出来。

本着不去大量构建特征的原则，我打算提取个MFCC和过零率就扔进LSTM网络里训练。

程序主要使用了librosa和pytorch这两个库

数据集采用的是下面这个韩语数据，包含四个场景的语音和对应标签
http://sail.ipdisk.co.kr/publist/VOL1/Database/VAD_DB/Recorded_data.zip

# 文件说明
注释什么的参考参考就行，我拿以前写的代码改的，可能有些地方改了代码忘了改注释。

infer.py: 用于模型推理
LSTM.py: LSTM模型
My_Dataset.py: 自定义的用于dataset类
preparData.py: 用于数据预处理，并将结果储存为HDF5文件，训练时以便全部读进内存
pytorchtools.py： 用于实现early-stop的文件
test_model.py: 测试模型在测试集上的正确率
train.py: 训练模型

train_log.csv：训练集损失函数收敛情况
valid_log.csv：验证集损失函数收敛情况

# 运行顺序
preparData.py -> train.py -> (test_model.py) -> (infer.py)
括号部分为测试模型在测试集上的正确率以及推理模型，不是一定得做
