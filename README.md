# PyTorch Sentiment Analysis

## Note: This repo only works with torchtext 0.9 or above which requires PyTorch 1.8 or above. If you are using torchtext 0.8 then please use [this](https://github.com/bentrevett/pytorch-sentiment-analysis/tree/torchtext08) branch

This repo contains tutorials covering how to do sentiment analysis using [PyTorch](https://github.com/pytorch/pytorch) 1.8 and [torchtext](https://github.com/pytorch/text) 0.9 using Python 3.7.

The first 2 tutorials will cover getting started with the de facto approach to sentiment analysis: recurrent neural networks (RNNs). The third notebook covers the [FastText](https://arxiv.org/abs/1607.01759) model and the comparision to [Vader sentiment](https://github.com/cjhutto/vaderSentiment)
There are also 2 bonus "appendix" notebooks. The first covers loading your own datasets with torchtext, while the second contains a brief look at the pre-trained word embeddings provided by torchtext.

## Getting Started

To install PyTorch, see installation instructions on the [PyTorch website](https://pytorch.org/get-started/locally).

To install torchtext:

``` bash
pip install torchtext
```

We'll also make use of spaCy to tokenize our data. To install spaCy, follow the instructions [here](https://spacy.io/usage/) making sure to install the English models with:

``` bash
python -m spacy download en_core_web_sm
```

## Tutorials

* 1 - [Simple Sentiment Analysis](https://github.com/pdtrung1994/python-project-FastText/blob/main/1%20-%20Simple%20Sentiment%20Analysis.ipynb) 

    This tutorial covers the workflow of a PyTorch with torchtext project. We'll learn how to: load data, create train/test/validation splits, build a vocabulary, create data iterators, define a model and implement the train/evaluate/test loop. The model will be simple and achieve poor performance, but this will be improved in the subsequent tutorials.

* 2 - [Upgraded Sentiment Analysis](https://github.com/pdtrung1994/python-project-FastText/blob/main/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb)

    Now we have the basic workflow covered, this tutorial will focus on improving our results. We'll cover: using packed padded sequences, loading and using pre-trained word embeddings, different optimizers, different RNN architectures, bi-directional RNNs, multi-layer (aka deep) RNNs and regularization.

* 3 - [Faster Sentiment Analysis](https://github.com/pdtrung1994/python-project-FastText/blob/main/Sentiment_Analysis.ipynb)

    After we've covered all the fancy upgrades to RNNs, we'll look at a different approach that does not use RNNs. More specifically, we'll implement the model from Bag of Tricks for Efficient Text Classification. This simple model achieves comparable performance as the Upgraded Sentiment Analysis, but trains much faster.
