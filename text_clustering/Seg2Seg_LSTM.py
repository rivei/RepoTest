# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:17:29 2019
初识大名鼎鼎的Seg2Seg模型: LSTM
https://ctolib.com/topics-133988.html

在keras 上实践,通过keras例子来理解lastm循环神经网络
https://blog.csdn.net/ma416539432/article/details/53509607

Note ***:
    思路，将句子变成 词向量 X; Y 为编码；
    学习，然后匹配（忽略语义的方法）***如何结合相似度分析？
"""
# =============================================================================
# --------------------- 
# 作者：aaon22357 
# 来源：CSDN 
# 原文：https://blog.csdn.net/aaon22357/article/details/82733218 
# 版权声明：本文为博主原创文章，转载请附上博文链接！
# =============================================================================
#import os
##设置GPU：
#def set_gpus(gpu_index):
#    if type(gpu_index) == list:
#        gpu_index = ','.join(str(_) for _ in gpu_index)
#    if type(gpu_index) == int:
#        gpu_index = str(gpu_index)
#    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
#set_gpus(1)
#os.environ

##通过设置Keras的Tensorflow后端的全局变量达到满血？？
##https://blog.csdn.net/silent56_th/article/details/60154637
#import os
#import tensorflow as tf
#import keras.backend.tensorflow_backend as KTF
#
#def get_session(gpu_fraction=0.3):
#    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
#
#    num_threads = os.environ.get('OMP_NUM_THREADS')
#    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#
#    if num_threads:
#        return tf.Session(config=tf.ConfigProto(
#            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
#    else:
#        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#
##使用过程中显示的设置session:
#import keras.backend.tensorflow_backend as KTF
#KTF.set_session(get_session())

import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
 
# fix random seed for reproducibility
numpy.random.seed(7)

# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

#第三步：尝试构建出上面1:1映射的数据集
# prepare the dataset of input to output pairs encoded as integers
seq_length = 1   # 注意这里, seq_length = 1
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
    print( seq_in, '->', seq_out)
# 上面的程序运行的结果就是：X -> Y
"""
A -> B
B -> C
...
"""

# reshape X to be        [samples   , time steps, features]
X = numpy.reshape(dataX, (len(dataX), seq_length, 1))
print(X.shape)  # (25, 1, 1)
print(X)
"""
array([[[ 0]],
 
     [[ 1]],
 
     [[ 2]],
     ...
"""

# normalize
X = X / float(len(alphabet))

# one hot encode the output variable
y = np_utils.to_categorical(dataY)
 
"""
array([[ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
       0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
       0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
       0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
      ....
"""
 
y.shape # (25, 26)

# create and fit the model
model = Sequential()
# 32: cell state size,  input_shape: 期望1+ samples
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2]))) 
model.add(Dense(y.shape[1], activation='softmax'))  # 输出多个类
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, batch_size=20, verbose=2)

# summarize performance of the model
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))
 
# Output: Model Accuracy: 76.00% -> 88%

#修改： seq_length = 3, feature = 1
#两处修改：

seq_length = 3
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
    print( seq_in, '->', seq_out)
 
# reshape X to be[samples, time steps, features]
X = numpy.reshape(dataX, (len(dataX), seq_length,   1))
y = np_utils.to_categorical(dataY)
y.shape # (23, 26)
model = Sequential()
# 32: cell state size,  input_shape: 期望1+ samples
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2]))) 
model.add(Dense(y.shape[1], activation='softmax'))  # 输出多个类
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, batch_size=20, verbose=2)

# summarize performance of the model
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))
#Model Accuracy: 95.65%

"""
ABC -> D
BCD -> E
CDE -> F
DEF -> G
EFG -> H
FGH -> I
GHI -> J
"""

## Very low accuracy!!
# prepare the dataset of input to output pairs encoded as integers
import datetime
num_inputs = 1000
max_len = 5
dataX = []
dataY = []
for i in range(num_inputs):
    start = numpy.random.randint(len(alphabet)-2)
    end = numpy.random.randint(start, min(start+max_len,len(alphabet)-1))
    sequence_in = alphabet[start:end+1]
    sequence_out = alphabet[end + 1]
    dataX.append([char_to_int[char] for char in sequence_in])
    dataY.append(char_to_int[sequence_out])
    print(sequence_in, '->', sequence_out)
 
dataX = keras.preprocessing.sequence.pad_sequences(dataX, maxlen=max_len, dtype='float32')
#补0不是一个好方法
X = numpy.reshape(dataX, (len(dataX),5,1))
y = np_utils.to_categorical(dataY)
y.shape # (1000, 26)
model = Sequential()
# 32: cell state size,  input_shape: 期望1+ samples
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2]))) 
model.add(Dense(y.shape[1], activation='softmax'))  # 输出多个类
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

starttime = datetime.datetime.now()
#model.fit(X, y, epochs=100, batch_size=1, verbose=2)
model.fit(X, y, epochs=500, batch_size=20, verbose=2)
print('runtime: ' , datetime.datetime.now() - starttime)
#3.6 GPU runtime:  0:30:12.579749 (batch size =1)
#3.6 GPU runtime:  0:01:24.881344 (batch size = 20)
#3.7 CPU runtime:  0:05:13.737900 :(
# summarize performance of the model
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))

testX = [[ 0.,  0.,  0.,  0., 23.],
       [ 0.,  0.,  0.,  9., 10.],
       [ 0.,  3.,  4.,  5.,  6.]]

for pattern in testX:
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print( seq_in, "->", result)