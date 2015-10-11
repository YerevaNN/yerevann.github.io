---
layout: post
title: Spoken language identification with deep convolutional networks
---

By [Hrayr Harutyunyan](https://github.com/Harhro94)

## Contents
{:.no_toc}
* TOC
{:toc}

## Dataset and scoring

Recently [TopCoder](https://topcoder.com/) announced a [contest](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=16555&compid=49304)
to identify the spoken language in audio recordings. 

The recordings were in one of the 176 languages. Training set consisted of 66176 `mp3` files, 
376 per language, from which I have separated 12320 recordings for validation 
(Python script is [available on GitHub](https://github.com/YerevaNN/Spoken-language-identification-CNN/blob/master/choose_val_set.py)). 
Test set consisted of 12320 `mp3` files. All recordings had the same length (~10 sec) 
and seemed to be noise-free (at least all the samples that I have checked).

<!--more-->

Score was calculated the following way: for every `mp3` top 3 guesses were uploaded in a CSV file. 
1000 points were given if the first guess is correct,
400 points if the second guess is correct and 160 points if the third guess is correct. 
During the contest the score was calculated only on 3520 recordings from the test set. 
After the contest the final score was calculated on the remaining 8800 recordings.

## Preprocessing

I entered the contest just 14 days before the deadline, so didn't have much time to investigate
audio specific techniques. But we had a deep convolutional network developed few months ago,
and it seemed to be a good idea to test a pure CNN on this problem. 
Some Google search revealed that the idea is not new. The earliest attempt I could find was a 
[paper by G. Montavon](http://research.microsoft.com/en-us/um/people/dongyu/nips2009/papers/montavon-paper.pdf)
presented in NIPS 2009 conference. The author used a network with 3 convolutional layers trained on 
[spectrograms](https://en.wikipedia.org/wiki/Spectrogram) of audio recordings, and 
the output of convolutional/subsampling layers was given to a [time-delay neural network](https://en.wikipedia.org/wiki/Time_delay_neural_network). 

I found a [Python script](http://www.frank-zalkow.de/en/code-snippets/create-audio-spectrograms-with-python.html?ckattempt=1) 
which creates a spectrogram of a `wav` file. I used [`mpg123` library](http://www.mpg123.de/index.shtml) 
to convert `mp3` files to `wav` format.

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
| 19| Fully connected |20	| 256		|  |  |  |
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
| Training (blue) and validation (red) loss over the 150 000 iterations on the non-augmented dataset | 

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

|![Frequency warping formula](/public/2015-10-11/frequency-warp-formula.png "Frequency warping formula") |
| --- |
| Frequency warping formula from the [paper by L. Lee and R. Rose](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=650310&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel4%2F89%2F14168%2F00650310). 
? is the scaling factor. Following Jaitly and Hinton I [chose it uniformly](https://github.com/YerevaNN/Spoken-language-identification-CNN/blob/master/augment_data.py#L92)
between 0.9 and 1.1 | 

I also [randomly cropped](https://github.com/YerevaNN/Spoken-language-identification-CNN/blob/master/augment_data.py#L77)
the spectrograms so they had `768x256` size. Here are the results:

|![Spectrogram without modifications](/public/2015-10-11/spectrogram.jpg) "Spectrogram without modifications" | 
| Spectrogram of one of the recordings |
|![Cropped spectrogram with warped frequency axis](/public/2015-10-11/spectrogram-warped-cropped.jpg) "Cropped spectrogram with warped frequency axis" | 
| Cropped spectrogram of the same recording with warped frequency axis |

For each `mp3` I have created 20 random spectrograms, but trained the network on 10 of them. 
It took more than 2 days to create the augmented dataset and convert it to LevelDB format. 
But training the network proved to be even harder. For 3 days I couldn't significantly decrease
the train loss. After removing dropout layers the loss started to decrease but it would take weeks 
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
Softmax loss on validation set reached ................... which corresponded to  ..... score. 
The score was calculated by averaging softmax outputs over 10 spectrograms of each recording.

## Ensembling

30 hours before the deadline I had several models from the same network. And even simple
ensembling (just the sum of softmax activations of different models) performed better than
any individual model. Hrant suggested to use [XGBoost](https://github.com/dmlc/xgboost), 
which is a fast implementation of [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) 
algorithm and is very popular among Kagglers. XGBoost has a good documentation and 
all parameters are [well explained](https://github.com/dmlc/xgboost/blob/master/doc/parameter.md).

To perform the ensembling I was creating a CSV file containing softmax activations 
(or the average of softmax activations among [20](https://github.com/YerevaNN/Spoken-language-identification-CNN/blob/master/ensembling/get_output_layers.py#L40) 
augmented versions of the same recording) using [this script](https://github.com/YerevaNN/Spoken-language-identification-CNN/blob/master/ensembling/get_output_layers.py).
Then I was running XGBoost on these CSV files. The submission file (which was requested by TopCoder)
was generated using [this script](https://github.com/YerevaNN/Spoken-language-identification-CNN/blob/master/make_submission.py).

I also tried to train a [simple neural network](https://github.com/YerevaNN/Spoken-language-identification-CNN/blob/master/ensembling/ensemble.theano.py)
with one hidden layer on the same CSV files. The results were significantly better than
with XGBoost. 

The best result was obtained by ensembling the following models: ..................

Final score was 3 401 840 and it was the [10th result](http://community.topcoder.com/longcontest/stats/?module=ViewOverview&rd=16555)
of the contest.

## What we learned from this contest

This was a quite interesting contest, although too short when compared with Kaggle's contests.

* Plain, [AlexNet](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)-like 
convolutional networks work quite well for fixed length audio recordings
* Vocal tract length perturbation works well as an augmentation technique
* Caffe supports sharing weights between convolutional networks having different input sizes
* Single layer neural network sometimes performs better than XGBoost for ensembling (although 
I had just one day to test the both)


## Unexplored options 

* It is interesting to see if a network with 50% dropout layers will improve the accuracy
* Maybe larger convolutional networks, like _OxfordNet_ will perform better. 
They require much more memory, and it was risky to play with them under a tough deadline
* [Hybrid methods]((http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6288864))
combining CNN and Hidden Markov Models should work better
* We believe it is possible to squeeze more from these models with better ensembling methods
* [Other contestants report](https://apps.topcoder.com/forums/?module=Thread&threadID=866734&start=0&mc=4)
better results based on careful mixing of the results of more traditional techniques, 
including [n-gram](https://en.wikipedia.org/wiki/N-gram)
and [Gaussian Mixture Models](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model).
We believe the combination of these techniques with the deep models will provide very 
good results on this dataset

One important issue is that the organizers of this contest [do not allow](http://apps.topcoder.com/forums//?module=Thread&threadID=866217&start=0&mc=3)
to use the dataset outside the contest. We hope this decision will be changed eventually.

