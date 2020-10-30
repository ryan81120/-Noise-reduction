#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential, model_from_json, load_model
from keras.layers.core import Dense, Dropout, Flatten, Activation, SpatialDropout2D, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import Convolution1D
from keras import layers
from keras.layers import LSTM ,Embedding ,SimpleRNN,CuDNNLSTM
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from scipy.io import wavfile
import pdb
import scipy.io
# import librosa
import os
from os.path import join as ojoin
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import time  
import numpy as np
import numpy.matlib
import argparse
import random
# import theano
# import theano.tensor as T
import tensorflow as tf
from keras.callbacks import TensorBoard
import keras.backend.tensorflow_backend as KTF
from keras.backend.tensorflow_backend import set_session

#設定使用GPU 檢查是否有抓到
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99 #使用45%記憶體
set_session(tf.Session(config=config))

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

##定義讀檔路徑function
def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.

    for root, directories, files in os.walk(directory):

        for filename in files:
            if filename.endswith('.wav'):
            # Join the two strings in order to form the full filepath.
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.
                # pdb.set_trace()
    file_paths.sort()
    return file_paths 

mixed_file=get_filepaths('mixed_all_snr/')
cleaned_file=get_filepaths('clean')

##切割data and test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(mixed_file, cleaned_file, test_size=0.33, random_state=42)   

Train_Noisy_lists=X_train
Train_Clean_paths= y_train

Test_Noisy_lists  = X_test
Test_Clean_paths = y_test
          
Num_testdata=len(Test_Noisy_lists)   
Num_traindata=len(Train_Noisy_lists)



def train_data_generator(noisy_list, clean_path):
    index=0
    while True:
        
        rate, noisy = wavfile.read(noisy_list[index])
        noisy=noisy.astype('float32')         
        if len(noisy.shape)==2:
            noisy=(noisy[:,0]+noisy[:,1])/2  
        noisy=np.reshape(noisy,(1,np.shape(noisy)[0],1))

        rate, clean = wavfile.read(clean_path[index])
        clean=clean.astype('float32')  
        clean=clean/2**15
            
        clean=np.reshape(clean,(1,np.shape(clean)[0],1))
        
        index += 1
        if index == len(noisy_list):
            index = 0

        yield noisy, clean

def val_data_generator(noisy_list, clean_path):
    index=0
    while True:
        rate, noisy = wavfile.read(noisy_list[index])
        noisy=noisy.astype('float32')         
        if len(noisy.shape)==2:
            noisy=(noisy[:,0]+noisy[:,1])/2       

        noisy=np.reshape(noisy,(1,np.shape(noisy)[0],1))
          
        rate, clean = wavfile.read(clean_path[index])
        clean=clean.astype('float32')  
        if len(clean.shape)==2:
            clean=(clean[:,0]+clean[:,1])/2
        clean=clean/2**15

        clean=np.reshape(clean,(1,np.shape(clean)[0],1))

        index += 1
        if index == len(noisy_list):
            index = 0
          
        yield noisy, clean 
        
##建立模型
start_time = time.time()

model = Sequential()
model.add(CuDNNLSTM(32,return_sequences=True,input_shape=(None,1)))
model.add(CuDNNLSTM(32,return_sequences=True)) # 返回维度为 32 的向量序列
model.add(Dense(1,activation='tanh'))
model.summary()

##訓練開始
epoch=5
batch_size=1
model.compile(loss='mse', optimizer='adam')
    
with open('{}.json'.format('firsttry'),'w') as f:    # save the model
    f.write(model.to_json()) 
checkpointer = ModelCheckpoint(filepath='{}.hdf5'.format('firsttry'), verbose=1, save_best_only=True, mode='min')  

print ('training...')

g1 = train_data_generator(Train_Noisy_lists, Train_Clean_paths)
print()
g2 = val_data_generator(Test_Noisy_lists, Test_Clean_paths)

tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                  batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)   

hist=model.fit_generator(g1,
                         samples_per_epoch=Num_traindata,
                        epochs=epoch, 
                        verbose=1,
                        validation_data=g2,
                        nb_val_samples=Num_testdata,
                        max_q_size=1, 
                        nb_worker=16,
                        use_multiprocessing=True
                         )   

##畫Loss圖
# # plotting the learning curve
TrainERR=hist.history['loss']
ValidERR=hist.history['val_loss']
print ('@%f, Minimun error:%f, at iteration: %i' % (hist.history['val_loss'][epoch-1], np.min(np.asarray(ValidERR)),np.argmin(np.asarray(ValidERR))+1))
# print 'drawing the training process...'
plt.figure(4)
plt.plot(range(1,epoch+1),TrainERR,'b',label='TrainERR')
plt.plot(range(1,epoch+1),ValidERR,'r',label='ValidERR')
plt.xlim([1,epoch])
plt.legend()
plt.xlabel('epoch')
plt.ylabel('error')
plt.grid(True)
plt.show()
plt.savefig('Learning_curve_{}.png'.format('FCN_firsttry'), dpi=150)


end_time = time.time()
print ('The code for this file ran for %.2fm' % ((end_time - start_time) / 60.))

