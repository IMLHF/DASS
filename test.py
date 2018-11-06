from dataManager.data import mixed_aishell
import numpy as np
def data_manager_test():
  data_mixed=mixed_aishell.read_data_sets('/mnt/d/tf_recipe/PIT_SYS/utterance_test/speaker_set')
  print(np.shape(data_mixed.Y[0:10]))

if __name__=="__main__":
  data_manager_test()
