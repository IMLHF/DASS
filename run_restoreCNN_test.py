from dataManager.data import mixed_aishell
import tensorflow as tf
import numpy as np
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

def load_model_test(x,y):
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  # with tf.Session(config=config) as sess:
  with tf.Session(config=config,graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph('_log/models/PIT/saved_model/PITsaved_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint("_log/models/PIT/saved_model"))
    x_in = sess.graph.get_tensor_by_name("Placeholder:0")
    # y_reference = sess.graph.get_tensor_by_name("Placeholder_1:0")
    predict=tf.concat(tf.get_collection('tower_y_outs'),0)
    # y_out_collection=tf.get_collection('tower_y_outs')
    return sess.run(predict,feed_dict={x_in:x})#,sess.run(y_out_collection,feed_dict={x_in:x})

def pit_test_for_load():
  batch_size=32
  # data_dir = '/home/student/work/pit_test/data_small'
  data_dir = '/mnt/d/tf_recipe/PIT_SYS/utterance_test/speaker_set'
  data_mixed = mixed_aishell.read_data_sets(data_dir)
  data=data_mixed.test_cc.X_Y
  x_len = len(data)
  total_batch = x_len//batch_size if (x_len % batch_size == 0) else ((
      x_len//batch_size)+1)
  mse_list = []
  for i in range(total_batch):
    s_site = i*batch_size
    e_site = min(s_site+batch_size, x_len)
    x_y = data[s_site:e_site]
    x = x_y[0]
    y = x_y[1]
    print(np.shape(x))
    y_out= load_model_test(x,y)
    print(y_out)
    print(np.shape(y_out))
    # mse1 = np.mean((y-y_out)**2, (1, 2))
    # y_out_speaker1, y_out_speaker2 = np.split(y_out, 2, axis=-1)
    # y_out_swaped = np.concatenate([y_out_speaker2, y_out_speaker1], axis=-1)
    # mse2 = np.mean((y-y_out_swaped)**2, (1, 2))
    # loss = np.where(mse1 < mse2, mse1, mse2)
    # mse = np.mean(loss)
    # print('Batch %04d MSE : %lf' % (i+1, mse))
    # mse_list.append(mse)
  print('Test Average MSE : %lf' % np.mean(mse_list))

def separate_speech():
  # data_dir = '/home/student/work/pit_test/data_small'
  data_dir = '/mnt/d/tf_recipe/PIT_SYS/utterance_test/speaker_set'
  data_mixed = mixed_aishell.read_data_sets(data_dir)
  mixX=np.array(data_mixed.train.X[:10])
  picture_spec((mixX*8-3),'exp/mixSpeech')
  raw_Y=np.array(data_mixed.train.Y[:10])
  picture_spec((raw_Y*8-3),'exp/rawCLEAN')
  raw=np.array(data_mixed.train.X[:10],dtype=np.float32)
  pre=load_model_test(raw,None)
  pre=(pre*8-3)
  picture_spec(pre,'exp/restorePIT')
  for i in range(10):
    np.savetxt('exp/restorePIT.num'+str(i),pre[i])
    np.savetxt('exp/restorePIT.numT'+str(i),pre[i].T)

if __name__=='__main__':
  # pit_test_for_load()
  separate_speech()
  # data_dir = '/home/student/work/pit_test/data_small'
  # # data_dir = '/mnt/d/tf_recipe/PIT_SYS/utterance_test/speaker_set'
  # data_mixed = mixed_aishell.read_data_sets(data_dir)
  # load_model_predict_test(data_mixed.test_cc.X[:128],data_mixed.test_cc.Y[:128])
