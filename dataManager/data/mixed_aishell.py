from dataManager.basicData import DATABASE
from dataManager import basicData
import os
import shutil
import datetime
import scipy
import scipy.io
import wave
import numpy as np
from utils import spectrum_tool
from FLAGS import MIXED_AISHELL_PARAM


FILE_NAME = __file__[max(__file__.rfind('/')+1, 0):__file__.rfind('.')]
# region define
LOG_NORM_MAX = MIXED_AISHELL_PARAM.LOG_NORM_MAX
LOG_NORM_MIN = MIXED_AISHELL_PARAM.LOG_NORM_MIN
NFFT = MIXED_AISHELL_PARAM.NFFT
OVERLAP = MIXED_AISHELL_PARAM.OVERLAP
FS = MIXED_AISHELL_PARAM.FS
LEN_WAWE_PAD_TO = MIXED_AISHELL_PARAM.LEN_WAWE_PAD_TO
UTT_SEG_FOR_MIX= MIXED_AISHELL_PARAM.UTT_SEG_FOR_MIX
DATASET_DIRS = MIXED_AISHELL_PARAM.DATASET_DIRS
DATASET_SIZES = MIXED_AISHELL_PARAM.DATASET_SIZES
WAVE_NORM = MIXED_AISHELL_PARAM.WAVE_NORM
# endregion


def _get_waveData1__waveData2(file1, file2):
  f1 = wave.open(file1, 'rb')
  f2 = wave.open(file2, 'rb')
  waveData1 = np.fromstring(f1.readframes(f1.getnframes()),
                            dtype=np.int16)
  waveData2 = np.fromstring(f2.readframes(f2.getnframes()),
                            dtype=np.int16)
  f1.close()
  f2.close()
  while len(waveData1) < LEN_WAWE_PAD_TO:
    waveData1 = np.tile(waveData1, 2)
  while len(waveData2) < LEN_WAWE_PAD_TO:
    waveData2 = np.tile(waveData2, 2)

  if WAVE_NORM:
    waveData1 = waveData1/np.max(np.abs(waveData1)) * 32767
    waveData2 = waveData2/np.max(np.abs(waveData2)) * 32767
  # if len(waveData1) < len(waveData2):
  #   waveData1, waveData2 = waveData2, waveData1
  # # print(np.shape(waveData1))
  # gap = len(waveData1)-len(waveData2)
  # waveData2 = np.concatenate(
  #     (waveData2, np.random.randint(-400, 400, size=(gap,))))
  return waveData1[:LEN_WAWE_PAD_TO], waveData2[:LEN_WAWE_PAD_TO]


def _mix_wav(waveData1, waveData2):
  # 混合语音
  mixedData = (waveData1+waveData2)/2
  mixedData = np.array(mixedData, dtype=np.int16)  # 必须指定是16位，因为写入音频时写入的是二进制数据
  return mixedData


def rmNormalization(tmp):
  tmp = (10**(tmp*(LOG_NORM_MAX-LOG_NORM_MIN)+LOG_NORM_MIN))-0.5
  ans = np.where(tmp > 0, tmp, 0)  # 防止计算误差导致的反归一化结果为负数
  return ans


def _extract_norm_log_mag_spec(data):
  # 归一化的幅度谱对数
  mag_spec = spectrum_tool.magnitude_spectrum_librosa_stft(
      data, NFFT, OVERLAP)
  log_mag_spec = np.log10(mag_spec+0.5)
  # log_power_spectrum_normalization
  log_mag_spec[log_mag_spec > LOG_NORM_MAX] = LOG_NORM_MAX
  log_mag_spec[log_mag_spec < LOG_NORM_MIN] = LOG_NORM_MIN
  log_mag_spec += np.abs(LOG_NORM_MIN)
  log_mag_spec /= (np.abs(LOG_NORM_MIN)+LOG_NORM_MAX)
  # mean=np.mean(log_mag_spec)
  # var=np.var(log_mag_spec)
  # log_mag_spec=(log_mag_spec-mean)/var
  return log_mag_spec


def _extract_phase(data):
  theta = spectrum_tool.phase_spectrum_librosa_stft(data, NFFT, OVERLAP)
  return theta


def _extract_feature_x(data_index_list):
  data = []
  for data_index in data_index_list:
    waveData1, waveData2 = _get_waveData1__waveData2(
        data_index[0], data_index[1])
    mixedData = _mix_wav(waveData1, waveData2)
    data.append(_extract_norm_log_mag_spec(mixedData))
  return data


def _extract_feature_y(data_index_list):
  data = []
  for data_index in data_index_list:
    waveData1, waveData2 = _get_waveData1__waveData2(
        data_index[0], data_index[1])
    clean1_log_mag_spec = _extract_norm_log_mag_spec(waveData1)
    clean2_log_mag_spec = _extract_norm_log_mag_spec(waveData2)
    data.append(np.concatenate(
        [clean1_log_mag_spec, clean2_log_mag_spec], axis=1))
  return data


def _extract_feature_x_y(data_index_list):
  datax = []
  datay = []
  for data_index in data_index_list:
    waveData1, waveData2 = _get_waveData1__waveData2(
        data_index[0], data_index[1])
    mixedData = _mix_wav(waveData1, waveData2)
    clean1_log_mag_spec = _extract_norm_log_mag_spec(waveData1)
    clean2_log_mag_spec = _extract_norm_log_mag_spec(waveData2)
    # print(data_index_list)
    datax.append(_extract_norm_log_mag_spec(mixedData))
    datay.append(np.concatenate([clean1_log_mag_spec,
                                 clean2_log_mag_spec],
                                axis=1))
    # print(np.shape(datax))
    # print(np.shape(datay))
  return [np.array(datax), np.array(datay)]


def _extract_x_theta(data_index_list):
  data = []
  for data_index in data_index_list:
    waveData1, waveData2 = _get_waveData1__waveData2(
        data_index[0], data_index[1])
    mixedData = _mix_wav(waveData1, waveData2)
    data.append(_extract_phase(mixedData))
  return data


def _init_data__(rawdata, data_dict_dir, logger):
  logger.set_file(FILE_NAME+'/init_data.log')
  LOG_DIR = '/'.join(logger.file_dir().split('/')[:-1])
  if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
  os.makedirs(LOG_DIR)
  if os.path.exists(data_dict_dir):
    shutil.rmtree(data_dict_dir)
  os.makedirs(data_dict_dir)
  clean_wav_speaker_set_dir = rawdata
  os.makedirs(data_dict_dir+'/train')
  os.makedirs(data_dict_dir+'/validation')
  os.makedirs(data_dict_dir+'/test_cc')
  cwl_train_file = open(data_dict_dir+'/train/clean_wav_dir.list', 'a+')
  cwl_validation_file = open(
      data_dict_dir+'/validation/clean_wav_dir.list', 'a+')
  cwl_test_cc_file = open(data_dict_dir+'/test_cc/clean_wav_dir.list', 'a+')
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
      for wav in speaker_wav_list[:UTT_SEG_FOR_MIX[0]]:
        if wav[-4:] == ".wav":
          cwl_train_file.write(speaker_dir+'/'+wav+'\n')
          clean_wav_list_train.append(speaker_dir+'/'+wav)
      for wav in speaker_wav_list[UTT_SEG_FOR_MIX[0]:UTT_SEG_FOR_MIX[1]]:
        if wav[-4:] == ".wav":
          cwl_validation_file.write(speaker_dir+'/'+wav+'\n')
          clean_wav_list_validation.append(speaker_dir+'/'+wav)
      for wav in speaker_wav_list[UTT_SEG_FOR_MIX[1]:]:
        if wav[-4:] == ".wav":
          cwl_test_cc_file.write(speaker_dir+'/'+wav+'\n')
          clean_wav_list_test_cc.append(speaker_dir+'/'+wav)

  cwl_train_file.close()
  cwl_validation_file.close()
  cwl_test_cc_file.close()
  logger.print_save('train clean: '+str(len(clean_wav_list_train)))
  logger.print_save('validation clean: '+str(len(clean_wav_list_validation)))
  logger.print_save('test_cc clean: '+str(len(clean_wav_list_test_cc)))

  data_class_dir = DATASET_DIRS
  dataset_mixedutt_num = DATASET_SIZES
  all_mixed = 0
  for (clean_wav_list, j) in zip((clean_wav_list_train, clean_wav_list_validation, clean_wav_list_test_cc), range(3)):
    logger.print_save('\n'+data_class_dir[j]+" data preparing...")
    logger.print_save('Current time: ' +
                      str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    mixed_wav_list_file = open(
        data_dict_dir+'/'+data_class_dir[j]+'/mixed_wav_dir.list', 'a+')
    mixed_wave_list = []
    len_wav_list = len(clean_wav_list)
    generated_num = 0
    while generated_num < dataset_mixedutt_num[j]:
      uttid = np.random.randint(len_wav_list, size=2)
      uttid1 = uttid[0]
      uttid2 = uttid[1]
      utt1_dir = clean_wav_list[uttid1]
      utt2_dir = clean_wav_list[uttid2]
      speaker1 = utt1_dir.split('/')[-2]
      speaker2 = utt2_dir.split('/')[-2]
      if speaker1 == speaker2:
        continue
      generated_num += 1
      mixed_wav_list_file.write(utt1_dir+' '+utt2_dir+'\n')
      mixed_wave_list.append([utt1_dir, utt2_dir])
    # for i_utt in range(len_wav_list): # n^2混合，数据量巨大
    #   for j_utt in range(i_utt,len_wav_list):
    #     utt1_dir=clean_wav_list[i_utt]
    #     utt2_dir=clean_wav_list[j_utt]
    #     speaker1 = utt1_dir.split('/')[-2]
    #     speaker2 = utt2_dir.split('/')[-2]
    #     if speaker1 == speaker2:
    #       continue
    #     mixed_wav_list_file.write(utt1_dir+' '+utt2_dir+'\n')
    #     mixed_wave_list.append([utt1_dir, utt2_dir])
    mixed_wav_list_file.close()
    scipy.io.savemat(
        data_dict_dir+'/'+data_class_dir[j]+'/mixed_wav_dir.mat', {"mixed_wav_dir": mixed_wave_list})
    all_mixed += len(mixed_wave_list)
    logger.print_save(data_class_dir[j]+' mixed: '+str(len(mixed_wave_list)))
  logger.print_save('All : '+str(all_mixed))
  logger.print_save(
      '\nData preparation over time: '+str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


# API
def read_data_sets(rawdata):

  DATA_DICT_DIR = '_data/' + FILE_NAME  # 数据字典的位置
  _init_data__(rawdata, DATA_DICT_DIR, DATABASE.get_logger())

  # 数据集字典 {集合名称:集合索引}
  set_dict = {
      "train": scipy.io.loadmat(DATA_DICT_DIR+"/train/mixed_wav_dir.mat")["mixed_wav_dir"],
      "validation": scipy.io.loadmat(DATA_DICT_DIR+"/validation/mixed_wav_dir.mat")["mixed_wav_dir"],
      "test_cc": scipy.io.loadmat(DATA_DICT_DIR+"/test_cc/mixed_wav_dir.mat")["mixed_wav_dir"],
  }
  # 集合元素获取器[x,y,x_y]
  itemgetor_list = [_extract_feature_x,
                    _extract_feature_y,
                    _extract_feature_x_y,
                    _extract_x_theta, ]
  data = DATABASE(set_dict, itemgetor_list)
  data.train
  return data
