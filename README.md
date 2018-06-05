# Multi-Order-Decoding
This is an implementation of the paper [Does Higher Order LSTM Have Better Accuracy in Chunking and Named Entity Recognition?] [[pdf]](https://arxiv.org/pdf/1711.08231.pdf).

## Environment and Dependency
- Ubuntu 16.04
- Python 2.7
- Tensorflow 1.0 

<br /> 

## Required Files

#### Feature files
The model uses features extracted from original texts. Ignore the feature input in the model if you don't want to use extracted features.

#### Probability Files
The model uses features extracted from original texts. Ignore the feature input in the model if you don't want to use extracted features.
The multi-order-3 LSTM model uses the probabilities generated by single order-1 model and single order-2 model at testing stage. So the probabilities need to be preserved in files. We provied the pretrained order-1 model and order-2 model which are in the lstm-1order file and the lstm-2order file separately. You can use these two models to generate the probability files. You can also get the files by training your own single order-n models. The single order-n model is exactly bi-directional lstm with order-n tag set.
