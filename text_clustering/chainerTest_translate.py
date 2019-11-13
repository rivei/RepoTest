# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:53:21 2019
神经网络机器翻译的实现
https://www.cnblogs.com/renzhe0009/p/6260618.html
"""

from chainer import FunctionSet
#from chainer.functions import *
import chainer.functions as cf
VOCAB_SIZE = 8000 #是单词的数量
HIDDEN_SIZE = 100 #是隐藏层的维数

model = FunctionSet(
  w_xh = cf.embed_id(VOCAB_SIZE, HIDDEN_SIZE), # 输入层(one-hot) -> 隐藏层
  w_hh = cf.linear(HIDDEN_SIZE, HIDDEN_SIZE), # 隐藏层 -> 隐藏层
  w_hy = cf.linear(HIDDEN_SIZE, VOCAB_SIZE), # 隐藏层 -> 输出层
)  

import math
import numpy as np
#from chainer import Variable
#from chainer.functions import *

def forward(sentence, model): # sentence是strの排列结果。
  sentence = [convert_to_your_word_id(word) for word in sentence] # 单词转换为ID
  h = FunctionSet.Variable(np.zeros((1, HIDDEN_SIZE), dtype=np.float32)) # 隐藏层的初值
  log_joint_prob = float(0) # 句子的结合概率

  for word in sentence:
    x = FunctionSet.Variable(np.array([[word]], dtype=np.int32)) # 下一次的输入层
    y = cf.softmax(model.w_hy(h)) # 下一个单词的概率分布
    log_joint_prob += math.log(y.data[0][word]) #结合概率的分布
    h = math.tanh(model.w_xh(x) + model.w_hh(h)) #隐藏层的更新

  return log_joint_prob #返回结合概率的计算结果


def forward(sentence, model):
  #...

  accum_loss = FunctionSet.Variable(np.zeros((), dtype=np.float32)) # 累计损失的初値
  #...

  for word in sentence:
    x = FunctionSet.Variable(np.array([[word]], dtype=np.int32)) #下次的输入 (=现在的正确值)
    u = model.w_hy(h)
    accum_loss += cf.softmax_cross_entropy(u, x) # 累计损失
    y = cf.softmax(u)
    #...

  return log_joint_prob, accum_loss # 累计损失全部返回


#现在就可以进行学习了。
import chainer.optimizers as co
#...

def train(sentence_set, model):
  opt = co.sgd() # 使用梯度下降法
  opt.setup(model) # 学习初期化
  for sentence in sentence_set:
    opt.zero_grad(); # 勾配の初期化
    log_joint_prob, accum_loss = forward(sentence, model) # 损失的计算
    accum_loss.backward() # 误差反向传播
    opt.clip_grads(10) # 剔除过大的梯度
    opt.update() # 参数更新
    
##################################################################33
model = FunctionSet(
  w_xi = EmbedID(SRC_VOCAB_SIZE, SRC_EMBED_SIZE), #输入层(one-hot) -> 输入词向量层
  w_ip = Linear(SRC_EMBED_SIZE, 4 * HIDDEN_SIZE), # 输入词向量层-> 输入隐藏层
  w_pp = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 输入隐藏层 -> 输入隐藏层
  w_pq = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 输入隐藏层-> 输出隐藏层
  w_yq = EmbedID(TRG_VOCAB_SIZE, 4 * HIDDEN_SIZE), #输出层(one-hot) -> 输出隐藏层
  w_qq = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), #输出隐藏层 -> 输出隐藏层
  w_qj = Linear(HIDDEN_SIZE, TRG_EMBED_SIZE), # 输出隐藏层 -> 输出词向量层
  w_jy = Linear(TRG_EMBED_SIZE, TRG_VOCAB_SIZE), # 输出隐藏层 -> 输出隐藏层
) 

# src_sentence: 需要翻译的句子 e.g. ['他', '在', '走']
# trg_sentence: 正解的翻译句子 e.g. ['he', 'runs']
# training: 机械学习的预测。
def forward(src_sentence, trg_sentence, model, training):

  # 转换单词ID
  # 对正解的翻訳追加终端符号
  src_sentence = [convert_to_your_src_id(word) for word in src_sentence]
  trg_sentence = [convert_to_your_trg_id(word) for wprd in trg_sentence] + [END_OF_SENTENCE]

  # LSTM内部状态的初期値
  c = Variable(np.zeros((1, HIDDEN_SIZE), dtype=np.float32))

  # encoder
  for word in reversed(src_sentence):
    x = Variable(np.array([[word]], dtype=np.int32))
    i = tanh(model.w_xi(x))
    c, p = lstm(c, model.w_ip(i) + model.w_pp(p))

  # encoder -> decoder
  c, q = lstm(c, model.w_pq(p))

  # decoder
  if training:
    # 学习时使用y作为正解的翻译、forward结果作为累计损失来返回
    accum_loss = np.zeros((), dtype=np.float32)
    for word in trg_sentence:
      j = tanh(model.w_qj(q))
      y = model.w_jy(j)
      t = Variable(np.array([[word]], dtype=np.int32))
      accum_loss += softmax_cross_entropy(y, t)
      c, q = lstm(c, model.w_yq(t), model.w_qq(q))
    return accum_loss
  else:
    # 预测时翻译器生成的y作为下次的输入，forward的结果作为生成了的单词句子
    # 选择y中最大概率的单词、没必要用softmax。
    hyp_sentence = []
    while len(hyp_sentence) < 100: # 剔除生成100个单词以上的句子
      j = tanh(model.w_qj(q))
      y = model.w_jy(j)
      word = y.data.argmax(1)[0]
      if word == END_OF_SENTENCE:
        break # 生成了终端符号，结束。
      hyp_sentence.append(convert_to_your_trg_str(word))
      c, q = lstm(c, model.w_yq(y), model.w_qq(q))
    return hyp_sentence




    
    