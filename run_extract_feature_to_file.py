from dataManager.data import mixed_aishell
import os
import shutil
import scipy.io
import time
import numpy as np

output_dir = '_datamat'


# data_dir = '/home/student/work/pit_test/data'
data_dir = '/home/student/work/pit_test/data_small'
# data_dir = '/mnt/d/tf_recipe/PIT_SYS/utterance_test/speaker_set'
if os.path.exists(output_dir):
  shutil.rmtree(output_dir)
os.mkdir(output_dir)

data_mixed = mixed_aishell.read_data_sets(data_dir)
for dataset_name in ['train', 'validation', 'test_cc']:
  os.mkdir(os.path.join(output_dir,dataset_name))
  start_time = time.time()
  dataset = data_mixed.train
  if dataset_name == 'validation':
    dataset = data_mixed.validation
  elif dataset_name == 'test_cc':
    dataset = data_mixed.test_cc
  for i, index_name in enumerate(dataset.index_list):
    uttname1, uttname2 = str(index_name).split(' ')
    uttname1 = uttname1[uttname1.rfind('/')+1:uttname1.rfind('.')]
    uttname2 = uttname2[uttname2.rfind('/')+1:uttname2.rfind('.')]
    index_name = 'MIX'.join([uttname1, uttname2])
    # scipy.io.savemat(os.path.join(output_dir, dataset_name, index_name),
    #                  {'X': dataset.X[i],
    #                   'Y': dataset.Y[i],
    #                   'X_Theta': dataset.X_Theta[i]})
    np.save(os.path.join(output_dir, dataset_name, index_name+'_X'),np.array(dataset.X[i],dtype=np.float16))
    np.save(os.path.join(output_dir, dataset_name, index_name+'_Y'),np.array(dataset.Y[i],dtype=np.float16))
    np.save(os.path.join(output_dir, dataset_name, index_name+'_XTheta'),np.array(dataset.X_Theta[i],dtype=np.float16))
  print('Train set extraction over. cost time %06d' % (time.time()-start_time))
