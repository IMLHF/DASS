import numpy as np
import scipy
import scipy.signal
import librosa
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def picture_spec(spec,name):
  for i in range(np.shape(spec)[0]):
    spec_t=spec[i]
    plt.pcolormesh(spec_t,)
    plt.title('STFT Magnitude')
    plt.xlabel('Frequency')
    plt.ylabel('Time')
    plt.savefig(name+("%03d.jpg" % i))
    print("write pic "+name+("%03d" % i))
    # plt.show()
    plt.close()

def picture_wave(wave_t,name,framerate):
  nframes=np.shape(wave_t)[0]
  _time = np.arange(0, nframes)*(1.0 / framerate)
  plt.plot(_time, wave_t)
  plt.xlabel("Time(s)")
  plt.ylabel("Amplitude")
  plt.title("Single channel wavedata")
  plt.grid(True)
  plt.savefig(name+".jpg")
  # plt.show()
  print("write pic "+name)
  plt.close()

def magnitude_spectrum_sci_stft(signal, fs, NFFT=512, overlap=256):
  f, t, mag_frames = np.absolute(scipy.signal.stft(signal,
                                                   fs=fs,  # signal的采样率
                                                   window="hamming",
                                                   nperseg=NFFT,
                                                   noverlap=overlap,
                                                   nfft=NFFT,
                                                   ))
  # pow_frames = (1.0 / NFFT) * ((mag_frames) ** 2)
  return t, f, mag_frames.T


def magnitude_spectrum_np_fft(signal, NFFT=512, overlap=256):
  segsize = NFFT  # 每帧长度
  inc = segsize-overlap
  signal_length = len(signal)
  nframes = 1 + int(np.ceil(float(np.abs(signal_length - segsize)) / inc))
  pad_length = int((nframes-1)*inc+segsize)  # 补0后的长度
  zeros = np.zeros((pad_length-signal_length,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
  pad_signal = np.concatenate((signal, zeros))  # 填补后的信号记为pad_signal
  indices = np.tile(np.arange(0, segsize), (nframes, 1))+np.tile(
      np.arange(0, nframes*inc, inc), (segsize, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
  indices = np.array(indices, dtype=np.int32)  # 展开overlap的帧矩阵
  frames = pad_signal[indices]  # 得到展开后帧信号矩阵
  frames *= np.hamming(segsize)  # 汉明窗
  mag_frames = np.absolute(np.fft.rfft(frames,
                                       NFFT,
                                       axis=1,
                                       ))
  # pow_frames = (1.0 / NFFT) * ((mag_frames) ** 2)
  return mag_frames


def magnitude_spectrum_librosa_stft(signal, NFFT, overlap):
  signal=np.array(signal,dtype=np.float)
  tmp = librosa.core.stft(signal,
                          n_fft=NFFT,
                          hop_length=256,
                          window=scipy.signal.windows.hann)
  tmp = np.absolute(tmp)
  return tmp.T


def phase_spectrum_librosa_stft(signal, NFFT, overlap):
  '''
  return theta
  '''
  signal=np.array(signal,dtype=np.float)
  tmp = librosa.core.stft(signal,
                          n_fft=NFFT,
                          hop_length=overlap,
                          window=scipy.signal.windows.hann)
  tmp = np.angle(tmp)
  return tmp.T


def librosa_istft(signal_complex, NFFT, overlap):
  tmp = librosa.core.istft(signal_complex,
                           win_length=NFFT,
                           hop_length=overlap,
                           window=scipy.signal.windows.hann)
  return tmp
