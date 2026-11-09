metric: `val_bpb` (lower is better)

```
              n_seeds  val_bpb_mean  val_bpb_std  params_M  train_min  delta_vs_gpt_textbook  paired_p
name                                                                                                  
gpt_modern          2        2.1088       0.0071    3.1626     0.2292                -0.0716    0.0207
lstm                2        2.1356       0.0017    1.0698     0.2366                -0.0447    0.0209
gru                 2        2.1650       0.0010    0.8067     0.2302                -0.0153    0.0798
lstm_512            2        2.1709       0.0082    4.2368     0.2359                -0.0095    0.4635
gpt_textbook        2        2.1803       0.0038    3.4383     0.1731                 0.0000       NaN
rnn                 2        2.4442       0.0008    0.2803     0.0693                 0.2639    0.0077
```
