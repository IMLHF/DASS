from dataManager import basicData
import os
import shutil
from log.log import LOG
import datetime
import scipy
import scipy.io
import wave
import numpy as np
from basic import spectrum_tool

DATA_DICT_DIR = basicData.DATA_ROOT+'/' + \
    __file__[:__file__.rfind('.')]  # 数据字典的位置
LOG_DIR = basicData.LOG_ROOT+'/' + \
    __file__[:__file__.rfind('.')]  # 生成数据字典时的log位置
LOG_NORM_MAX = 5
LOG_NORM_MIN = -3
NFFT = 512
OVERLAP = 256
FS = 16000


def _get_waveData1__waveData2(file1, file2):
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


def _mix_wav(waveData1, waveData2):
  # 混合语音
  mixedData = (waveData1+waveData2)/2
  mixedData = np.array(mixedData, dtype=np.int16)  # 必须指定是16位，因为写入音频时写入的是二进制数据
  return mixedData


def _extract_norm_log_mag_spec(data):
  # 归一化的幅度谱对数
  mag_spec = spectrum_tool.magnitude_spectrum_np_fft(
      data, NFFT, OVERLAP)
  log_mag_spec = np.log10(mag_spec)
  # log_power_spectrum_normalization
  log_mag_spec[log_mag_spec > LOG_NORM_MAX] = LOG_NORM_MAX
  log_mag_spec[log_mag_spec < LOG_NORM_MIN] = LOG_NORM_MIN
  log_mag_spec += np.abs(LOG_NORM_MIN)
  log_mag_spec /= LOG_NORM_MAX
  return log_mag_spec


def _extract_feature_x(file1, file2):
  waveData1, waveData2 = _get_waveData1__waveData2(file1, file2)
  mixedData = _mix_wav(waveData1, waveData2)
  return _extract_norm_log_mag_spec(mixedData)


def _extract_feature_y(file1, file2):
  waveData1, waveData2 = _get_waveData1__waveData2(file1, file2)
  clean1_log_mag_spec = _extract_norm_log_mag_spec(waveData1)
  clean2_log_mag_spec = _extract_norm_log_mag_spec(waveData2)
  return np.concatenate([clean1_log_mag_spec, clean2_log_mag_spec], axis=1)


def _extract_feature_x_y(file1, file2):
  waveData1, waveData2 = _get_waveData1__waveData2(file1, file2)
  mixedData = _mix_wav(waveData1, waveData2)
  clean1_log_mag_spec = _extract_norm_log_mag_spec(waveData1)
  clean2_log_mag_spec = _extract_norm_log_mag_spec(waveData2)
  return _extract_norm_log_mag_spec(mixedData), np.concatenate([clean1_log_mag_spec,
                                                                clean2_log_mag_spec],
                                                               axis=1)


class __X(basicData.IndexableData):
  '''
  basicData.IndexableData.__init__(self, rawdata,data_set_name):
    self.rawdata=rawdata
    self.data_set_name=data_set_name
  '''

  def __init__(self, rawdata):
    self._size = -1
    basicData.IndexableData.__init__(
        self, rawdata, __file__[:__file__.rfind('.')], self.__len__())

  def __raw_getitem__(self, begin, end):
    mixed_wav_list = scipy.io.loadmat(
        DATA_DICT_DIR+"/train/mixed_wav_dir.mat")["mixed_wav_dir"][begin:end]
    x_data = []
    # print(mixed_wav_list)
    for mix_wav in mixed_wav_list:
      x_data.append(_extract_feature_x(mix_wav[0], mix_wav[1]))
    return x_data

  def __len__(self):
    if self._size == -1:
      self._size = len(scipy.io.loadmat(
          DATA_DICT_DIR+"/train/mixed_wav_dir.mat")["mixed_wav_dir"])
    return self._size

  def shape(self):
    pass


class __Y(basicData.IndexableData):
  '''
  basicData.IndexableData.__init__(self, rawdata,data_set_name):
    self.rawdata=rawdata
    self.data_set_name=data_set_name
  '''

  def __init__(self, rawdata):
    self._size = -1
    basicData.IndexableData.__init__(
        self, rawdata, __file__[:__file__.rfind('.')], self.__len__())

  def __raw_getitem__(self, begin, end):
    mixed_wav_list = scipy.io.loadmat(
        DATA_DICT_DIR+"/train/mixed_wav_dir.mat")["mixed_wav_dir"][begin:end]
    y_data = []
    for mix_wav in mixed_wav_list:
      y_data.append(_extract_feature_y(mix_wav[0], mix_wav[1]))
    return y_data

  def __len__(self):
    if self._size == -1:
      self._size = len(scipy.io.loadmat(
          DATA_DICT_DIR+"/train/mixed_wav_dir.mat")["mixed_wav_dir"])
    return self._size

  def shape(self):
    pass


class __X_Y(basicData.IndexableData):
  '''
  basicData.IndexableData.__init__(self, rawdata,data_set_name):
    self.rawdata=rawdata
    self.data_set_name=data_set_name
  '''

  def __init__(self, rawdata):
    self._size = -1
    basicData.IndexableData.__init__(
        self, rawdata, __file__[:__file__.rfind('.')], self.__len__())

  def __raw_getitem__(self, begin, end):
    '''
    return [x,y]
    '''
    mixed_wav_list = scipy.io.loadmat(
        DATA_DICT_DIR+"/train/mixed_wav_dir.mat")["mixed_wav_dir"][begin:end]
    x_y_data = []
    for mix_wav in mixed_wav_list:
      x_y_data.append(_extract_feature_x_y(mix_wav[0], mix_wav[1]))
    shape_len = len(np.shape(x_y_data))
    tranind = [1, 0]
    tranind.extend(list(range(shape_len))[2:])
    return np.transpose(x_y_data, tranind)

  def __len__(self):
    if self._size == -1:
      self._size = len(scipy.io.loadmat(
          DATA_DICT_DIR+"/train/mixed_wav_dir.mat")["mixed_wav_dir"])
    return self._size

  def shape(self):
    pass


def __init_data__(rawdata):
  if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
  os.makedirs(LOG_DIR)
  if os.path.exists(DATA_DICT_DIR):
    shutil.rmtree(DATA_DICT_DIR)
  os.makedirs(DATA_DICT_DIR)
  log = LOG(LOG_DIR+'/init_data.log')
  clean_wav_speaker_set_dir = rawdata
  os.makedirs(DATA_DICT_DIR+'/train')
  os.makedirs(DATA_DICT_DIR+'/validation')
  os.makedirs(DATA_DICT_DIR+'/test_cc')
  cwl_train_file = open(DATA_DICT_DIR+'/train/clean_wav_dir.list', 'a+')
  cwl_validation_file = open(
      DATA_DICT_DIR+'/validation/clean_wav_dir.list', 'a+')
  cwl_test_cc_file = open(DATA_DICT_DIR+'/test_cc/clean_wav_dir.list', 'a+')
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
    log.print_save('\n'+data_class_dir[j]+" data preparing...")
    log.print_save('Current time: ' +
                   str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    mixed_wav_list_file = open(
        DATA_DICT_DIR+'/'+data_class_dir[j]+'/mixed_wav_dir.list', 'a+')
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
        DATA_DICT_DIR+'/'+data_class_dir[j]+'/mixed_wav_dir.mat', {"mixed_wav_dir": mixed_wave_list})
  log.print_save(
      '\nData preparation over time: '+str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


class __TMP:
  def __init__(self):
    self.X = None
    self.Y = None
    self.X_Y = None


def read_data_sets(rawdata):

  __init_data__(rawdata)

  tmp = __TMP()
  tmp.X = __X(rawdata)
  tmp.Y = __Y(rawdata)
  tmp.X_Y = __X_Y(rawdata)
  return tmp
