from dataManager.data import mixed_aishell
import numpy as np
def data_manager_test():
  data_mixed=mixed_aishell.read_data_sets('/mnt/d/tf_recipe/PIT_SYS/utterance_test/speaker_set')
  print(np.shape(data_mixed.train.Y[0:128]))
  print(np.shape(data_mixed.train.Y[512:512+128]))
  # print(__file__.rfind('/'))

data_manager_test()
# print(__file__)
