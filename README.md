# LSTM_TimeSequencePrediction

A small project implementing LSTM to predict the sine waves in the time sequence. 
This project uses PyTorch. It uses torch.nn library to use the LSTM Cell, we train 10 samples each of length 1000 and width of wave being 20.
We want to predict the sine wave values for the next 1000.

We use LBFGS as an optimizer and use learning rate of 0.8

To learn more about LBFGS Optimizer
https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html
