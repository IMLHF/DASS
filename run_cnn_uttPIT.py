from dataManager.data import mixed_aishell
from models.implementedModels import DEEP_SPEECH_SEPARTION
import tensorflow as tf
import numpy as np
from losses import loss
from abc import abstractmethod, ABCMeta
import matplotlib.pyplot as plt
import os

def picture_spec(spec,name):
  dir_=name[:name.rfind('/')]
  if not os.path.exists(dir_):
    os.makedirs(dir_)
  for i in range(np.shape(spec)[0]):
    spec_t=spec[i]
    plt.pcolormesh(spec_t,)
    plt.title('STFT Magnitude')
    plt.xlabel('Frequency')
    plt.ylabel('Time')
    plt.savefig(name+str(i)+".jpg")
    print("write pic "+name+str(i))
    # plt.show()

def run_uttPIT():
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
  pit_model.test_PIT(data_mixed.test_cc.X_Y,batch_size=128)
  pit_model.save_model()
  test_x=data_mixed.test_cc.X_Y[:10]
  picture_spec(mixed_aishell.rmNormalization(test_x[0]),'_log/models/uttPIT/saved_model/mix_speech')
  picture_spec(mixed_aishell.rmNormalization(test_x[1]),'_log/models/uttPIT/saved_model/clean')
  pre=pit_model.predict(test_x)
  picture_spec(mixed_aishell.rmNormalization(pre),'_log/models/uttPIT/saved_model/reco_clean')
  del pit_model


if __name__=='__main__':
  run_uttPIT()
