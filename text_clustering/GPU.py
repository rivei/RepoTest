# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:03:32 2019

@author: Wei
"""

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
#
#import notebook_util
#notebook_util.pick_gpu_lowest_memory()
#import tensorflow as tf


# =============================================================================
# --------------------- 
# 作者：相国大人 
# 来源：CSDN 
# 原文：https://blog.csdn.net/github_36326955/article/details/79910448 
# 版权声明：本文为博主原创文章，转载请附上博文链接！
# =============================================================================
#import os
#import tensorflow as tf
#import keras.backend.tensorflow_backend as KTF
#
##进行配置，每个GPU使用60%上限现存
#os.environ["CUDA_VISIBLE_DEVICES"]="1" # 使用编号为1，2号的GPU
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.6 # 每个GPU现存上届控制在60%以内
#session = tf.Session(config=config)
#
## 设置session
#KTF.set_session(session )

##将内存中的数据分批(batch_size)送到显存中进行运算
#def generate_arrays_from_memory(data_train, labels_cat_train, batch_size):
#    x = data_train
#    y=labels_cat_train
#    ylen=len(y)
#    loopcount=ylen//batch_size
#    while True:
#        i = np.random.randint(0,loopcount)
#        yield x[i*batch_size:(i+1)*batch_size],y[i*batch_size:(i+1)*batch_size]
#
## 下面的load不会占用显示存，而是load到了内存中。
#data_train=np.loadtxt("./data_compress/data_train.txt")
#labels_cat_train=np.loadtxt('./data_compress/labels_cat_train.txt')
#data_val=np.loadtxt('./data_compress/data_val.txt')
#labels_cat_val=np.loadtxt('./data_compress/labels_cat_val.txt')
#
#hist=model.fit_generator(
#                        generate_arrays_from_memory(data_train,
#                                                   labels_cat_train,
#                                                   bs),
#                         steps_per_epoch=int(train_size/bs),
#                         epochs=ne,
#                         validation_data=(data_val,labels_cat_val),
#                         callbacks=callbacks )
#


#import os
#from tensorflow.python.client import device_lib
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "99"
#print(device_lib.list_local_devices())


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

import datetime
import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0" #runtime:  0:00:45.471309 with GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = "1" #runtime:  0:00:09.246160 WITH CPU
class LinearSep:
    def __init__(self):
        self.n_train = 50
        self.n_test = 10
        self.x_train, self.y_train, self.x_test, self.y_test = self._gene_data()
    def _gene_data(self):
        x = np.random.uniform(-1,1,[self.n_train, 2])
        y = (x[:,1]>x[:,0]).astype(np.int32)
        x += np.random.randn(self.n_train, 2)*0.05
        x_test = np.random.uniform(-1,1,[self.n_test, 2])
        y_test = (x_test[:,1]>x_test[:,0]).astype(np.int32)
        return x, y, x_test, y_test
        
#随机生成数据
dataset = LinearSep()
X_train, Y_train = dataset.x_train, dataset.y_train
print(Y_train)
Y_train=np.eye(2)[Y_train]
X_test,Y_test=dataset.x_test,dataset.y_test
Y_test=np.eye(2)[Y_test]
x=tf.placeholder(tf.float32,[None,2],name='input')
y=tf.placeholder(tf.float32,[None,2],name='output')
w1 = tf.get_variable(name='w_fc1', shape=[2, 20], dtype=tf.float32)
b1 = tf.get_variable(name='b_fc1', shape=[20], dtype=tf.float32)

out = tf.matmul(x, w1) + b1
out = tf.nn.relu(out)


w2 = tf.get_variable(name='w_fc2', shape=[20, 2], dtype=tf.float32)
b2 = tf.get_variable(name='b_fc2', shape=[2], dtype=tf.float32)
out = tf.matmul(out, w2) +b2
out = tf.nn.softmax(out)
#cross entropy 损失函数
loss=-tf.reduce_mean(tf.reduce_sum(y*tf.log(out+1e-8), axis=1), axis=0)
#准确率
correct_pred = tf.equal(tf.argmax(y,axis=1), tf.argmax(out,axis=1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#定义优化器
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss) # 1e-3 是学习律
#初始化网络
#BATCH_SIZE = 128
EPOCH = 10000#优化次数

sess = tf.Session()
#gpu_options = tf.GPUOptions(allow_growth=True)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))   

sess.run(tf.global_variables_initializer())
starttime = datetime.datetime.now()
for ep in range(EPOCH):
    sess.run(train_op, feed_dict={x:X_train, y:Y_train})

    loss_train,acc_train= sess.run([loss,accuracy], feed_dict={x:X_train, y:Y_train})
    acc_test,pre_test= sess.run([accuracy,correct_pred], feed_dict={x:X_test, y:Y_test})

    if ep % 1000 == 0:
        print(ep, loss_train,acc_train, acc_test)
        print(Y_test.shape)

print('runtime: ' , datetime.datetime.now() - starttime)

test_pre=sess.run(out,feed_dict={x:X_test, y:Y_test})
print(len(test_pre))
mask  = np.argmax(test_pre,axis=1)
print(mask)
mask_0  = np.where(mask==0)
mask_1 =  np.where(mask==1)
X_0 = X_train[mask_0]
X_1 = X_train[mask_1]
print(X_0)
# =============================================================================
# --------------------- 
# 作者：marvel1014 
# 来源：CSDN 
# 原文：https://blog.csdn.net/marvel1014/article/details/84452560 
# 版权声明：本文为博主原创文章，转载请附上博文链接！
# 
# =============================================================================
