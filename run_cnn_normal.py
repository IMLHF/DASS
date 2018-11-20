from dataManager.data import mixed_aishell
from models.implementedModels import DEEP_SPEECH_SEPARTION
import tensorflow as tf
import numpy as np
from losses import loss
from abc import abstractmethod, ABCMeta
import matplotlib.pyplot as plt

def picture_spec(spec,name):
  for i in range(np.shape(spec)[0]):
    spec_t=spec[i]
    plt.pcolormesh(spec_t,)
    plt.title('STFT Magnitude')
    plt.xlabel('Frequency')
    plt.ylabel('Time')
    plt.savefig(name+str(i)+".jpg")
    print("write pic "+name+str(i))
    # plt.show()

def run_CONV():
  data_dir = '/home/student/work/pit_test/data_small'
  # data_dir = '/mnt/d/tf_recipe/PIT_SYS/utterance_test/speaker_set'
  data_mixed = mixed_aishell.read_data_sets(data_dir)
  conv_model = DEEP_SPEECH_SEPARTION(layers_size=[257, 2048, 2048, 2048, 514],
                                     times_width=[7, 1, 1, 1],
                                     loss_fun=loss.MSE,
                                     learning_rate=0.01,
                                     gpu_list=[1],
                                     name='CONV')
  conv_model.train(data_mixed.train.X_Y, batch_size=128,epoch=6)
  conv_model.test_PIT(data_mixed.test_cc.X_Y,batch_size=128)
  conv_model.save_model()


if __name__=='__main__':
  run_CONV()
