---
layout: post
title: Generating Constitution with recurrent neural networks
tags:
- draft
- RNN
- NLP
---

By [Narek Hovsepyan](https://github.com/Harhro94) and [Hrant Khachatrian]

Few months ago [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/) wrote a [great blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) about recurrent neural networks. He explained how these networks work and implemented a character-level RNN language model which learns to generate Paul Graham essays, [Shakespeare works](http://cs.stanford.edu/people/karpathy/char-rnn/shakespear.txt), [Wikipedia articles](http://cs.stanford.edu/people/karpathy/char-rnn/wiki.txt), [LaTeX articles](http://cs.stanford.edu/people/jcjohns/fake-math/4.pdf) and even C++ code. He also released the code of the network on [Github](https://github.com/karpathy/char-rnn). We decided to test it on some legal texts in Armenian.
  
<!--more-->

## Contents
{:.no_toc}
* TOC
{:toc}

## Character-level RNN language model

Andrej did a great job explaining how the recurrent networks learn and even visualized how they work on text input in [his blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). The program, called `char-rnn`, treats the input as a sequence of characters and has no prior knowledge about them. For example, it doesn't know that the text is in English, that there are words and there are sentences, that the space character has a special meaning and so on. After some training it manages to figure out that some character combinations appear more often than the others, learns to predict English words, uses proper punctuation and even learns that open parentheses must be closed. When trained on Wikipedia articles it can generate text in MediaWiki format without syntax errors, although the text has little or no meaning. 

## Data

We decided to test Karpathy's RNN on Armenian text. Armenian language has a [unique alphabet](https://en.wikipedia.org/wiki/Armenian_alphabet), and the characters are encoded in the Unicode space by the codes [U+0530 - U+058F](http://www.unicode.org/charts/PDF/U0530.pdf). In UTF-8 these symbols use two bytes where the first byte is always `0xD4`, `0xD5` or `0xD6`. So the neural net has two look at almost 2 times larger distances in order to be able to learn the words. Also, the Armenian alphabet contains 39 letters, 50% more than Latin.

Recently the main political topic in Armenia is the Constitutional reform. This helped us to choose the corpus for training. We took all three versions of the Constitution of Armenia (the [first version](http://www.arlis.am/documentview.aspx?docID=1) voted in 1995, the [updated version](http://www.arlis.am/documentview.aspx?docID=75780) of 2005 and the [new proposal](http://moj.am/storage/uploads/Sahmanadrakan_1-15.docx) which will be voted later this year) and concatenated them in a [single text file](https://github.com/YerevaNN/char-rnn-constitution/blob/master/data/input.txt). The size of the corpus is just 440 KB, which is roughly 224 000 Unicode symbols (all non-Armenian symbols, including spaces and numbers use 1 byte). Andrej suggests to use at least 1MB data, so our corpus is very small. On the other hand the text is quite specific, the vocabulary is very small and the structure of the text is fairly simple.

All articles are of the following form:
```
Հոդված 1. Հայաստանի Հանրապետությունը ինքնիշխան, ժողովրդավարական, սոցիալական, իրավական պետություն է:
```
The first word `Հոդված` means "Article". Sentences end with the symbol `:`. 

## Network parameters

`char-rnn` works with basic recurrent neural networks, LSTM networks and GRU-RNNs. In our experiments we only used LSTM network with 2 layers. We trained for 50 epochs with the default learning rate parameters (base rate is `2e-3`, which decays by a factor of `0.97` after each `10` epochs). We wanted to understand how the size of LSTM internal state (`rnn_size`), [dropout](https://www.youtube.com/watch?v=UcKPdAM8cnI) and batch size affect the performance. We used [grid search](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search) over the following values:

* `rnn_size`: 128, 256, 512
* `batch_size`: 25, 50, 100
* `dropout`: 0, 0.2, 0.4 and at the end we added 0.6

After installing Lua, Torch and CUDA (as described on [`char-rnn` page](https://github.com/karpathy/char-rnn#requirements)) we have moved our mini-corpus to `/data/input.txt` and ran the [`run.sh` file](....), which contains commands like this:

```
th train.lua -data_dir data/ -batch_size 50 -dropout 0.4 -rnn_size 512 -gpuid 0 -savefile bs50s512d0.4 | tee log_bs50s512d0.4
```

File names encode the hyperparameters, and the output of `char-rnn` is logged using [`tee` command](https://en.wikipedia.org/wiki/Tee_(command)).

## Analysis

We have adapted [this script](https://github.com/YerevaNN/Caffe-python-tools/blob/master/plot_loss.py) written by Hrayr to plot the behavior of loss functions during the 50 epochs. The script, which runs on `char-rnn` output is available on [Github](https://github.com/YerevaNN/char-rnn-constitution/blob/master/plot_loss.py). These graphs show, for example, that we practically do not gain anything after 25 epochs.

|![Training and validation loss](/public/2015-11-11/plot_bs50s256all.png "Training and validation loss") |
| --- |
| Training (blue to aqua) and validation (red to green) loss over 50 epochs. Used 256 RNN size, 50 batch size and various dropout values. Plotted using [this script](https://github.com/YerevaNN/char-rnn-constitution/blob/master/plot_loss.py). | 

This graph also shows that when no dropout is used, validation loss actually increases after 20 epochs. 


512 - large chunks
128 - new words

numeration is wrong
most articles start with 1


## Results



The preprocessing script is available on [GitHub](https://github.com/YerevaNN/Spoken-language-identification-CNN/blob/master/augment_data.py).

## Network architecture

I took the network architecture designed for the Kaggle's [diabetic retinopathy detection contest]({% post_url 2015-08-17-diabetic-retinopathy-detection-contest-what-we-did-wrong %}). 
It has 6 convolutional layers and 2 fully connected layers with 50% dropout. 
Activation function is always ReLU. Learning rates are set to be higher for
the first convolutional layers and lower for the top convolutional layers. 
The last fully connected layer has 176 neurons and is trained using a softmax loss. 

It is important to note that this network does not take into account the sequential characteristics
of the audio data. Although recurrent networks perform well on speech recognition tasks 
(one notable example is [this paper](http://arxiv.org/abs/1303.5778) 
by A. Graves, A. Mohamed and G. Hinton, cited by 272 papers according to the Google Scholar), 
I didn't have time to learn how they work.

I trained the CNN on [Caffe](http://caffe.berkeleyvision.org) with 32 images in a batch,
its description in Caffe prototxt format is available [here](https://github.com/YerevaNN/Spoken-language-identification-CNN/blob/master/prototxt/main_32r-2-64r-2-64r-2-128r-2-128r-2-256r-2-1024rd0.5-1024rd0.5_DLR.prototxt).

| Nr| Type	| Batches| Channels | Width | Height| Kernel size / stride |
| 0 | Input	| 32	| 1 		| 858	| 256	| 			| 
| 1	| Conv	| 32	| 32		| 852	| 250	| 7x7 / 1	|
| 2	| ReLU	| 32	| 32		| 852	| 250	| 			|
| 3 | MaxPool|32	| 32		| 426	| 125	| 3x3 / 2	|
| 4	| Conv	| 32	| 64		| 422	| 121	| 5x5 / 1	|
| 5	| ReLU	| 32	| 64		| 422	| 121	| 			|
| 6 | MaxPool|32	| 64		| 211	| 60	| 3x3 / 2	|
| 7	| Conv	| 32	| 64		| 209	| 58	| 3x3 / 1	|
| 8	| ReLU	| 32	| 64		| 209	| 58	| 			|
| 9 | MaxPool|32	| 64		| 104	| 29	| 3x3 / 2	|
| 10| Conv	| 32	| 128		| 102	| 27	| 3x3 / 1	|
| 11| ReLU	| 32	| 128		| 102	| 27	| 			|
| 12| MaxPool|32	| 128		| 51	| 13	| 3x3 / 2	|
| 13| Conv	| 32	| 128		| 49	| 11	| 3x3 / 1	|
| 14| ReLU	| 32	| 128		| 49	| 11	| 			|
| 15| MaxPool|32	| 128		| 24	| 5	| 3x3 / 2	|
| 16| Conv	| 32	| 256		| 22	| 3	| 3x3 / 1	|
| 17| ReLU	| 32	| 256		| 22	| 3	| 			|
| 18| MaxPool|32	| 256		| 11	| 1	| 3x3 / 2	|
| 19| Fully connected |20	| 1024		|  |  |  |
| 20| ReLU	|20	| 1024		|  |  |  |
| 21| Dropout|20	| 1024		|  |  |  |
| 22| Fully connected|20	| 1024		|  |  |  |
| 23| ReLU	|20	| 1024		|  |  |  |
| 24| Dropout|20	| 1024		|  |  |  |
| 25| Fully connected|20	| 176		|  |  |  |
| 26| Softmax Loss|1	| 176		|  |  |  |

[Hrant](https://github.com/Hrant-Khachatrian) suggested to try the [`ADADELTA` solver](http://arxiv.org/abs/1212.5701).
It is a method which dynamically calculates learning rate for every network parameter, and the 
training process is said to be independent of the initial choice of learning rate. Recently it
was [implemented in Caffe](https://github.com/BVLC/caffe/pull/2782).

In practice, the base learning rate set in the Caffe solver did matter. At first I tried to use `1.0` 
learning rate, and the network didn't learn at all. Setting the base learning rate to `0.01`
helped a lot and I trained the network for 90 000 iterations (more than 50 epochs). 
Then I switched to `0.001` base learning rate for another 60 000
iterations. The solver is available [here](https://github.com/YerevaNN/Spoken-language-identification-CNN/blob/master/prototxt/solver.main.adadelta.prototxt). 
Not sure why the base learning rate mattered so much at the early stages of the training.
One possible reason could be the large learning rate coefficients on the lower convolutional layers.
Both tricks (dynamically updating the learning rates in `ADADELTA` and large learning rate coefficients)
aim to fight the gradient vanishing problem, and maybe their combination is not a very good idea. 
This should be carefully analysed.

|![Training (blue) and validation (red) loss](/public/2015-10-11/no-augm-loss.jpg "Training (blue) and validation (red) loss") |
| --- |
| Training (blue) and validation (red) loss over the 150 000 iterations on the non-augmented dataset. The sudden drop of training loss corresponds to the point when the base learning rate was changed from `0.01` to `0.001`. Plotted using [this script](https://github.com/YerevaNN/Caffe-python-tools/blob/master/plot_loss.py). | 

The signs of overfitting were getting more and more visible and I stopped at 150 000 iterations. 
The softmax loss got to 0.43 and it corresponded to 3 180 000 score 
(out of 3 520 000 possible). Some ensembling with other models of the same network allowed to
get a bit higher score (3 220 000), but it was obvious that data augmentation is needed to overcome the 
overfitting problem.

## Data augmentation

The most important weakness of our team in the [previous contest]({% post_url 2015-08-17-diabetic-retinopathy-detection-contest-what-we-did-wrong %})
was that we didn't augment the dataset well enough. So I was looking for ways to augment the 
set of spectrograms. One obvious idea was to crop random, say, 9 second intervals of the recordings.
Hrant suggested another idea: to warp the frequency axis of the spectrogram. This process is known as
_vocal tract length perturbation_, and is generally used for speaker normalization at least 
[since 1998](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=650310&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel4%2F89%2F14168%2F00650310).
In 2013 [N. Jaitly and G. Hinton](https://www.cs.toronto.edu/~hinton/absps/perturb.pdf)
used this technique to augment the audio dataset. I [used this formula](https://github.com/YerevaNN/Spoken-language-identification-CNN/blob/master/augment_data.py#L32)
to linearly scale the frequency bins during spectrogram generation:

| ![Frequency warping formula](/public/2015-10-11/frequency-warp-formula.png "Frequency warping formula") |
| --- |
| Frequency warping formula from the [paper by L. Lee and R. Rose](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=650310&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel4%2F89%2F14168%2F00650310). α is the scaling factor. Following Jaitly and Hinton I [chose it uniformly](https://github.com/YerevaNN/Spoken-language-identification-CNN/blob/master/augment_data.py#L92) between 0.9 and 1.1 | 

I also [randomly cropped](https://github.com/YerevaNN/Spoken-language-identification-CNN/blob/master/augment_data.py#L77)
the spectrograms so they had `768x256` size. Here are the results:

|![Spectrogram without modifications](/public/2015-10-11/spectrogram.jpg "Spectrogram without modifications") | 
| Spectrogram of one of the recordings |
|![Cropped spectrogram with warped frequency axis](/public/2015-10-11/spectrogram-warped-cropped.jpg "Cropped spectrogram with warped frequency axis") | 
| Cropped spectrogram of the same recording with warped frequency axis |

For each `mp3` I have created 20 random spectrograms, but trained the network on 10 of them. 
It took more than 2 days to create the augmented dataset and convert it to LevelDB format (the format Caffe suggests). 
But training the network proved to be even harder. For 3 days I couldn't significantly decrease
the train loss. After removing the dropout layers the loss started to decrease but it would take weeks 
to reach reasonable levels. Finally, Hrant suggested to try to reuse the weights of the 
model trained on the non-augmented dataset. The problem was that due to the cropping,
the image sizes in the two datasets were different. But it turned out that convolutional 
and pooling layers in Caffe [work with images of variable sizes](https://github.com/BVLC/caffe/issues/189#issuecomment-36754479), 
only the fully connected layers couldn't reuse the weights from the first model. 
So I just [renamed the FC layers](https://github.com/YerevaNN/Spoken-language-identification-CNN/blob/master/prototxt/augm_32r-2-64r-2-64r-2-128r-2-128r-2-256r-2-1024r-1024r_DLR_nolrcoef.prototxt#L292)
in the prototxt file and [initialized](http://caffe.berkeleyvision.org/tutorial/interfaces.html#command-line)
the network (convolution filters) by the weights of the first model:

{% highlight bash %}
./build/tools/caffe train --solver=solver.prototxt --weights=models/main_32r-2-64r-2-64r-2-128r-2-128r-2-256r-2-1024rd0.5-1024rd0.5_DLR_72K-adadelta0.01_iter_153000.caffemodel

{% endhighlight %}

This helped a lot. I used standard stochastic gradient descent (inverse decay learning rate policy)
with base learning rate `0.001` for 36 000 iterations (less than 2 epochs), then increased 
the base learning rate to `0.01` for another 48 000 iterations (due to the inverse decay policy
the rate decreased seemingly too much). 
These trainings were done without any regularization techniques,
weight decay or dropout layers, and there were clear signs of overfitting. I tried to add 50%
dropout layers on fully connected layers, but the training was extremely slow. To improve the 
speed I used 30% dropout, and trained the network for 120 000 more iterations using [this solver](https://github.com/YerevaNN/Spoken-language-identification-CNN/blob/master/prototxt/solver.augm.nolrcoef.prototxt).
Softmax loss on the validation set reached 0.21 which corresponded to 3 390 000 score. 
The score was calculated by averaging softmax outputs over 20 spectrograms of each recording.
