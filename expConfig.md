
<table border="10">
<tr align="left"><th>No.</th><td width="50px">C1</td><td>C2</td><td>C3</td><td>C4</td><td>C5</td><td>C6</td>
</tr>
<tr align="left"><th>wav_mag_normalize</th><td>No</td><td>Yes</td><td></td><td>No</td><td>Yes</td><td>No</td>
</tr>
<tr align="left"><th>input_size</th><td>257</td><td></td><td></td><td></td><td></td><td>257</td>
</tr>
<tr align="left"><th>output_size</th><td>257</td><td></td><td></td><td></td><td></td><td>257</td>
</tr>
<tr align="left"><th>rnn_size</th><td>496</td><td></td><td></td><td></td><td>1024</td><td>496</td>
</tr>
<tr align="left"><th>rnn_layers_num</th><td>2</td><td></td><td>3</td><td>2/3?</td><td></td><td>2</td>
</tr>
<tr align="left"><th>batch_size</th><td>128</td><td></td><td></td><td></td><td>64</td><td>128</td>
</tr>
<tr align="left"><th>learning_rate</th><td>0.001</td><td></td><td></td><td></td><td>0.002</td><td>0.001</td>
</tr>
<tr align="left"><th>min_epoches</th><td>10</td><td></td><td></td><td></td><td></td><td>10</td>
</tr>
<tr align="left"><th>max_epoches</th><td>50</td><td></td><td></td><td></td><td></td><td>20</td>
</tr>
<tr align="left"><th>halving_factor</th><td>0.5</td><td></td><td></td><td></td><td></td><td>0.7</td>
</tr>
<tr align="left"><th>start_halving_impr</th><td>0.003</td><td></td><td></td><td></td><td></td><td>0.003</td>
</tr>
<tr align="left"><th>end_halving_impr</th><td>0.001</td><td></td><td></td><td></td><td></td><td>0.0005</td>
</tr>
<tr align="left"><th>keep_prob</th><td>0.8</td><td></td><td></td><td></td><td></td><td>0.8</td>
</tr>
<tr align="left"><th>max_grad_norm</th><td>5.0</td><td></td><td></td><td></td><td></td><td>5.0</td>
</tr>
<tr align="left"><th>model_type</th><td>LSTM</td><td></td><td></td><td>BLSTM</td><td>LSTM</td><td>BLSTM</td>
</tr>

<tr align="left">
<th>final_train_loss</th><td>3.1817</td>
<td>4.1797</td><td>7.7716</td><td>=</td><td>7.5969</td><td>=</td>
</tr>

<tr align="left">
<th>final_validation_loss</th><td>5.0099</td>
<td>6.2518</td><td>7.9592</td><td>=</td><td>7.7481</td><td>=</td>
</tr>

<tr align="left"><th>epoch_cost_time(Normal)</th>
<td>5.5h</td><td>5.5h</td><td>7.0h</td><td></td><td>9.0h</td><td>---------</td>
</tr>
<tr align="left"><th>epoch_cost_time(TFRecord+tf.data)</th>
<td></td><td>---------</td><td></td><td></td><td></td><td></td>
</tr>
<tr align="left"><th>speaker num</th><td>4</td><td></td><td></td><td></td><td></td><td>90</td>
</tr>
<tr align="left"><th>training speech time</th><td>1126h</td><td></td><td></td><td></td><td></td><td>1166h</td>
</tr>
<tr align="left"><th>--------------------------</th>
<td>---------</td>
<td>---------</td>
<td>---------</td>
<td>---------</td>
<td>---------</td>
<td>---------</td>
</tr>
</table>
