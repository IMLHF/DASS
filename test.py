from dataManager.data import mixed_aishell
from models.implementedModels import DEEP_SPEECH_SEPARTION
import tensorflow as tf
import numpy as np
from losses import loss
from abc import abstractmethod, ABCMeta
import matplotlib.pyplot as plt
import utils
import librosa
import wave


def data_manager_test():
  data_mixed = mixed_aishell.read_data_sets(
      '/mnt/d/tf_recipe/PIT_SYS/utterance_test/speaker_set')
  print(data_mixed.train.X_Y[0:2])
  print(np.shape(data_mixed.train.X_Y[0:2]))
  print(np.shape(data_mixed.train.Y[512:512+128]))
  # x = tf.placeholder('float', shape=[None, 257, None])
  # print(__file__.rfind('/'))


def tf_test():
  v1 = tf.get_variable('v1', shape=[2, 12], initializer=tf.ones_initializer(
      dtype=tf.float32), dtype=tf.float32)
  v2 = tf.get_variable('v2', shape=[2, 12], initializer=tf.random_uniform_initializer(
      maxval=-1., minval=1., seed=0), dtype=tf.float32)
  # 向当前计算图中添加张量集合
  tf.add_to_collection('v', v1)
  tf.add_to_collection('v', v2)
  y1 = tf.constant([[[1, 1, 1, 0, 0, 0]], [[0, 0, 0, 1, 1, 1]]])
  y2 = tf.constant([[[1, 1, 1, 0, 0, 0]], [[1, 1, 1, 0, 0, 0]]])
  yy = tf.constant([[1, 2, 3], [4, 5, 6]])

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.get_collection('v')))
    tf.Graph.clear_collection(tf.get_default_graph(), name='v')
    print(sess.run(tf.get_collection('v')))
    # a, b = tf.split(v1, 2, axis=-1)
    # print(np.shape(sess.run(a)))
    # print(sess.run(loss.PIT_MSE_for_CNN(y1, y2)))
    # print(sess.run(tf.reduce_mean(yy,[0,1])))


def angle_test():
  a = 1+1j
  f = np.abs(a)
  theta = np.angle(a)
  ra = f*np.exp(theta*1j)
  print(ra)


def wave_restore_from_mag_spec():
  utt1file = '/mnt/d/tf_recipe/PIT_SYS/utterance_test/BAC009S0908W0121.wav'
  utt2file = '/mnt/d/tf_recipe/PIT_SYS/utterance_test/BAC009S0002W0123.wav'
  waveData1, waveData2 = mixed_aishell._get_waveData1__waveData2(
      utt1file, utt2file)
  mixedData = mixed_aishell._mix_wav(waveData1, waveData2)
  waveMagSpec1 = utils.spectrum_tool.magnitude_spectrum_librosa_stft(
      waveData1, 512, 256)
  waveMagSpec2 = utils.spectrum_tool.magnitude_spectrum_librosa_stft(
      waveData2, 512, 256)
  mixedAngle = utils.spectrum_tool.phase_spectrum_librosa_stft(
      mixedData, 512, 256)
  wave1Angle = utils.spectrum_tool.phase_spectrum_librosa_stft(
      waveData1, 512, 256)
  wave2Angle = utils.spectrum_tool.phase_spectrum_librosa_stft(
      waveData2, 512, 256)
  waveDataReUseMixPhase1 = utils.spectrum_tool.librosa_istft(
      (waveMagSpec1*np.exp(mixedAngle*1j)).T, 512, 256)
  waveDataReUseMixPhase2 = utils.spectrum_tool.librosa_istft(
      (waveMagSpec2*np.exp(mixedAngle*1j)).T, 512, 256)
  waveDataReUseSelfPhase1 = utils.spectrum_tool.librosa_istft(
      (waveMagSpec1*np.exp(wave1Angle*1j)).T, 512, 256)
  waveDataReUseSelfPhase2 = utils.spectrum_tool.librosa_istft(
      (waveMagSpec2*np.exp(wave2Angle*1j)).T, 512, 256)

  waveReUseMixPhase1 = wave.open(
      '/mnt/d/tf_recipe/PIT_SYS/utterance_test/BAC009S0908W0121reUseMixPhase.wav', 'wb')
  waveReUseMixPhase2 = wave.open(
      '/mnt/d/tf_recipe/PIT_SYS/utterance_test/BAC009S0002W0123reUseMixPhase.wav', 'wb')
  waveReUseSelfPhase1 = wave.open(
      '/mnt/d/tf_recipe/PIT_SYS/utterance_test/BAC009S0908W0121reUseSelfPhase.wav', 'wb')
  waveReUseSelfPhase2 = wave.open(
      '/mnt/d/tf_recipe/PIT_SYS/utterance_test/BAC009S0002W0123reUseSelfPhase.wav', 'wb')
  nchannels = 1
  sampwidth = 2  # 采样位宽，2表示16位
  framerate = 16000
  nframesMix1 = len(waveDataReUseMixPhase1)
  nframesMix2 = len(waveDataReUseMixPhase2)
  nframesSelf1 = len(waveDataReUseSelfPhase1)
  nframesSelf2 = len(waveDataReUseSelfPhase2)
  comptype = "NONE"
  compname = "not compressed"
  waveReUseMixPhase1.setparams((nchannels, sampwidth, framerate, nframesMix1,
                                comptype, compname))
  waveReUseMixPhase2.setparams((nchannels, sampwidth, framerate, nframesMix2,
                                comptype, compname))
  waveReUseSelfPhase1.setparams((nchannels, sampwidth, framerate, nframesSelf1,
                                 comptype, compname))
  waveReUseSelfPhase2.setparams((nchannels, sampwidth, framerate, nframesSelf2,
                                 comptype, compname))
  waveReUseMixPhase1.writeframes(
      np.array(waveDataReUseMixPhase1, dtype=np.int16))
  waveReUseMixPhase2.writeframes(
      np.array(waveDataReUseMixPhase2, dtype=np.int16))
  waveReUseSelfPhase1.writeframes(
      np.array(waveDataReUseSelfPhase1, dtype=np.int16))
  waveReUseSelfPhase2.writeframes(
      np.array(waveDataReUseSelfPhase2, dtype=np.int16))


def wave_restore_use_data_manager_test():
  data_mixed = mixed_aishell.read_data_sets(
      '/mnt/d/tf_recipe/PIT_SYS/utterance_test/speaker_set')
  Y = np.array(data_mixed.train.Y[:10])

  Y = (10**(Y*8-3))-0.5
  mPhase = np.tile(np.array(data_mixed.train.X_Theta[:10]),2)
  complex_Y=Y*np.exp(mPhase*1j)
  complex_Y=np.array(np.split(complex_Y,2,axis=-1))

  for i in range(10):
    # reY1 = utils.spectrum_tool.librosa_istft((waveMagSpec1*np.exp(np.array(data_mixed.train.X_Theta[0])*1j)).T,512,256)
    reY1 = utils.spectrum_tool.librosa_istft(complex_Y[0][i].T,512,256)
    reY2 = utils.spectrum_tool.librosa_istft(complex_Y[1][i].T,512,256)
    tmpY = np.concatenate([reY1,reY2])
    wavefile = wave.open(
        '/mnt/d/tf_recipe/PIT_SYS/utterance_test/rewave'+str(i)+'.wav', 'wb')
    nchannels = 1
    sampwidth = 2  # 采样位宽，2表示16位
    framerate = 16000
    nframes = len(tmpY)
    comptype = "NONE"
    compname = "not compressed"
    wavefile.setparams((nchannels, sampwidth, framerate, nframes,
                        comptype, compname))
    wavefile.writeframes(
        np.array(tmpY, dtype=np.int16))


if __name__ == "__main__":
  # data_manager_test()
  # tf_test()
  # wave_restore_from_mag_spec()
  wave_restore_use_data_manager_test()
