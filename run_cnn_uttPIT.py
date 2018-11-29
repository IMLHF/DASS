from dataManager.data import mixed_aishell
from models.implementedModels import DEEP_SPEECH_SEPARTION
import tensorflow as tf
import numpy as np
from losses import loss
from abc import abstractmethod, ABCMeta
import matplotlib.pyplot as plt
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def cnn_uttPIT_train():
  data_dir = '/home/student/work/pit_test/data_small'
  # data_dir = '/mnt/d/tf_recipe/PIT_SYS/utterance_test/speaker_set'
  data_mixed = mixed_aishell.read_data_sets(data_dir)

  pit_model = DEEP_SPEECH_SEPARTION(layers_size=[257, 2048, 2048, 2048, 514],
                                    times_width=[7, 1, 1, 1],
                                    loss_fun=loss.utt_PIT_MSE_for_CNN_v2,
                                    learning_rate=0.001,
                                    gpu_list=[0],
                                    name='uttPIT')
  pit_model.train(data_mixed.train.X_Y, batch_size=128,epoch=10)
  del pit_model

def cnn_uttPIT_decode():
  pass


if __name__=='__main__':
  if sys.argv[1] == 'train':
    print('train...')
    cnn_uttPIT_train()
  elif sys.argv[1] == 'decode':
    print('decode...')
    cnn_uttPIT_decode()
