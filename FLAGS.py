class NNET_PARAM:
  decode = 0 # Flag indicating decoding or training.
  decode_show_spec = 1 # Flag indicating show spectrum or not.
  resume_training = 'false' # Flag indicating whether to resume training from cptk.
  input_size = 257 # The dimension of input.
  output_size = 257  # The dimension of output per speaker.
  rnn_size = 496 # Number of rnn units to use.
  rnn_num_layers = 2 # Number of layer of rnn model.
  batch_size = 128 #
  learning_rate = 0.001 # Initial learning rate.
  min_epochs = 10 # Min number of epochs to run trainer without halving.
  max_epochs = 50 # Max number of epochs to run trainer totally.
  halving_factor = 0.5 # Factor for halving.
  start_halving_impr = 0.003 # Halving when ralative loss is lower than start_halving_impr.
  end_halving_impr = 0.001 # Stop when relative loss is lower than end_halving_impr.
  num_threads_processing_data = 8 # The num of threads to read tfrecords files.
  save_dir = 'exp/lstm_pit' # Directory to put the train result.
  keep_prob = 0.8 # Keep probability for training dropout.
  max_grad_norm = 5.0 # The max gradient normalization.
  model_type = 'LSTM' # BLSTM or LSTM

class MIXED_AISHELL_PARAM:
  LOG_NORM_MAX = 5
  LOG_NORM_MIN = -3
  NFFT = 512
  OVERLAP = 256
  FS = 16000
  LEN_WAWE_PAD_TO = 16000*3 # Mixed wave length (16000*3 is 3 seconds)
  UTT_SEG_FOR_MIX=[260,290] # Separate utt to [0:260],[260,290],[290:end]
  DATASET_DIRS = ['train', 'validation', 'test_cc']
  DATASET_SIZES = [1400000, 18000, 180000]
  # WAVE_NORM=True
  WAVE_NORM = False
