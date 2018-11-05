from dataManager import dataManager
import os
import shutil
from log.log import LOG
import datetime
import scipy
import scipy.io
import wave
import numpy as np
from basic import spectrum_tool

LOG_NORM_MAX = 3
LOG_NORM_MIN = -3
NFFT = 512
OVERLAP = 256
FS = 16000


def __get_waveData1__waveData2(file1, file2):
  f1 = wave.open(file1, 'rb')
  f2 = wave.open(file2, 'rb')
  waveData1 = np.fromstring(f1.readframes(f1.getnframes()),
                            dtype=np.int16)
  waveData2 = np.fromstring(f2.readframes(f2.getnframes()),
                            dtype=np.int16)
  f1.close()
  f2.close()
  if len(waveData1) < len(waveData2):
    waveData1, waveData2 = waveData2, waveData1
  # print(np.shape(waveData1))
  gap = len(waveData1)-len(waveData2)
  waveData2 = np.concatenate(
      (waveData2, np.random.randint(-400, 400, size=(gap,))))
  return waveData1, waveData2


def __mix_wav(waveData1, waveData2):
  # 混合语音
  mixedData = int((waveData1+waveData2)/2)
  mixedData = np.array(mixedData, dtype=np.int16)  # 必须指定是16位，因为写入音频时写入的是二进制数据
  return mixedData


def __extract_norm_log_mag_spec(data):
  # 归一化的幅度谱对数
  _, _, mag_spec = spectrum_tool.magnitude_spectrum_sci_stft(
      data, FS, NFFT, OVERLAP)
  log_mag_spec = np.log10(mag_spec)
  # log_power_spectrum_normalization
  log_mag_spec[log_mag_spec > LOG_NORM_MAX] = LOG_NORM_MAX
  log_mag_spec[log_mag_spec < LOG_NORM_MIN] = LOG_NORM_MIN
  log_mag_spec += np.abs(LOG_NORM_MIN)
  log_mag_spec /= LOG_NORM_MAX
  return log_mag_spec


def extract_feature_x(file1, file2):
  waveData1, waveData2 = __get_waveData1__waveData2(file1, file2)
  mixedData = __mix_wav(waveData1, waveData2)
  return __extract_norm_log_mag_spec(mixedData)


def extract_feature_y(file1, file2):
  waveData1, waveData2 = __get_waveData1__waveData2(file1, file2)
  clean1_log_mag_spec = __extract_norm_log_mag_spec(waveData1)
  clean2_log_mag_spec = __extract_norm_log_mag_spec(waveData2)
  print('mixed_aishell.py : ling 97 :', np.shape(
      clean1_log_mag_spec), np.shape(clean2_log_mag_spec))
  return np.concatenate([clean1_log_mag_spec, clean2_log_mag_spec], axis=1)


def extract_feature_x_y(file1, file2):
  waveData1, waveData2 = __get_waveData1__waveData2(file1, file2)
  mixedData = __mix_wav(waveData1, waveData2)
  clean1_log_mag_spec = __extract_norm_log_mag_spec(waveData1)
  clean2_log_mag_spec = __extract_norm_log_mag_spec(waveData2)
  return __extract_norm_log_mag_spec(mixedData), np.concatenate([clean1_log_mag_spec,
                                                                 clean2_log_mag_spec],
                                                                axis=1)


class mixed_aishell(dataManager.basicManager):

  '''
  dataManager.basicManager.__init__(self, rawdata):
    self.rawdata=rawdata
    self.init_data()
    self.translate_list=[]
  '''

  def __init__(self, rawdata):
    dataManager.basicManager.__init__(rawdata)
    self.log_dir = self.log_dir+'/'+'mixed_aishell'
    self.__size = -1
    if os.path.exists(self.log_dir):
      shutil.rmtree(self.log_dir)
    os.makedirs(self.log_dir)

  def __init_data(self):
    log = LOG(self.log_dir+'/'+'init_data.log')
    clean_wav_speaker_set_dir = self.rawdata
    data_dir = "data"
    if os.path.exists(data_dir):
      shutil.rmtree(data_dir)
    os.makedirs('data/train')
    os.makedirs('data/validation')
    os.makedirs('data/test_cc')
    cwl_train_file = open(data_dir+'/train/clean_wav_dir.list', 'a+')
    cwl_validation_file = open(data_dir+'/validation/clean_wav_dir.list', 'a+')
    cwl_test_cc_file = open(data_dir+'/test_cc/clean_wav_dir.list', 'a+')
    clean_wav_list_train = []
    clean_wav_list_validation = []
    clean_wav_list_test_cc = []
    speaker_list = os.listdir(clean_wav_speaker_set_dir)
    speaker_list.sort()
    for speaker_name in speaker_list:
      speaker_dir = clean_wav_speaker_set_dir+'/'+speaker_name
      if os.path.isdir(speaker_dir):
        speaker_wav_list = os.listdir(speaker_dir)
        speaker_wav_list.sort()
        for wav in speaker_wav_list[:100]:
          if wav[-4:] == ".wav":
            cwl_train_file.write(speaker_dir+'/'+wav+'\n')
            clean_wav_list_train.append(speaker_dir+'/'+wav)
        for wav in speaker_wav_list[260:300]:
          if wav[-4:] == ".wav":
            cwl_validation_file.write(speaker_dir+'/'+wav+'\n')
            clean_wav_list_validation.append(speaker_dir+'/'+wav)
        for wav in speaker_wav_list[320:]:
          if wav[-4:] == ".wav":
            cwl_test_cc_file.write(speaker_dir+'/'+wav+'\n')
            clean_wav_list_test_cc.append(speaker_dir+'/'+wav)

    cwl_train_file.close()
    cwl_validation_file.close()
    cwl_test_cc_file.close()
    log.print_save('train clean: '+str(len(clean_wav_list_train)))
    log.print_save('validation clean: '+str(len(clean_wav_list_validation)))
    log.print_save('test_cc clean: '+str(len(clean_wav_list_test_cc)))
    log.print_save('train mixed about: '+str(len(clean_wav_list_train)
                                             * len(clean_wav_list_train)))
    log.print_save('validation mixed about: '+str(len(clean_wav_list_validation)
                                                  * len(clean_wav_list_validation)))
    log.print_save('test_cc mixed about: '+str(len(clean_wav_list_test_cc)
                                               * len(clean_wav_list_test_cc)))
    log.print_save('All about: '+str(len(clean_wav_list_train)*len(clean_wav_list_train)+len(clean_wav_list_validation)
                                     * len(clean_wav_list_validation)+len(clean_wav_list_test_cc)*len(clean_wav_list_test_cc)))

    data_class_dir = ['train', 'validation', 'test_cc']
    for (clean_wav_list, j) in zip((clean_wav_list_train, clean_wav_list_validation, clean_wav_list_test_cc), range(3)):
      log.print_save(data_class_dir[j]+" data preparing...")
      log.print_save('Current time: ' +
                     str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
      mixed_wav_list_file = open(
          data_dir+'/'+data_class_dir[j]+'/mixed_wav_dir.list', 'a+')
      mixed_wave_list = []
      for utt1_dir in clean_wav_list:
        for utt2_dir in clean_wav_list:
          speaker1 = utt1_dir.split('/')[-2]
          speaker2 = utt2_dir.split('/')[-2]
          if speaker1 == speaker2:
            continue
          mixed_wav_list_file.write(utt1_dir+' '+utt2_dir+'\n')
          mixed_wave_list.append([utt1_dir, utt2_dir])
      mixed_wav_list_file.close()
      scipy.io.savemat(
          data_dir+'/'+data_class_dir[j]+'/mixed_wav_dir.mat', {"mixed_wav_dir": mixed_wave_list})
    log.print_save(
        'Over time: '+str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

  def __x_raw_segment(self, begin, end):
    mixed_wav_list = scipy.io.loadmat(
        "data/train/mixed_wav_dir.mat")["mixed_wav_dir"][begin:end]
    x_data = []
    for mix_wav in mixed_wav_list:
      x_data.append(extract_feature_x(mix_wav[0], mix_wav[1]))
    return x_data

  def __y_raw_segment(self, begin, end):
    mixed_wav_list = scipy.io.loadmat(
        "data/train/mixed_wav_dir.mat")["mixed_wav_dir"][begin:end]
    y_data = []
    for mix_wav in mixed_wav_list:
      y_data.append(extract_feature_y(mix_wav[0], mix_wav[1]))
    return y_data

  def __x_y_raw_segment(self, begin, end):
    mixed_wav_list = scipy.io.loadmat(
        "data/train/mixed_wav_dir.mat")["mixed_wav_dir"][begin:end]
    x_y_data = []
    for mix_wav in mixed_wav_list:
      x_y_data.append(extract_feature_x_y(mix_wav[0], mix_wav[1]))
    return x_y_data

  def size(self):
    if self.__size == -1:
      self.__size = len(scipy.io.loadmat(
          "data/train/mixed_wav_dir.mat")["mixed_wav_dir"])
    return self.__size
