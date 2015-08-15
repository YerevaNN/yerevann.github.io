---
layout: post
title: Diabetic retinopathy detection contest. What we did wrong
---

After watching the [awesome video course by Hugo Larochelle](https://www.youtube.com/playlist?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH) on neural nets (more on this in the [previous post]({% post_url 2015-07-30-getting-started-with-neural-networks %})) we decided to test our knowledge on some computer vision contest. We looked at [Kaggle](https://www.kaggle.com/competitions) and the only active competition related to computer vision (except for the [digit recognizer contest](https://www.kaggle.com/c/digit-recognizer), for which lots of perfect out-of-the-box solutions exist) was the [Diabetic retinopathy detection contest](https://www.kaggle.com/c/diabetic-retinopathy-detection). This was probably quite hard to become our very first project, but nevertheless we decided to try. The team included [Karen](https://www.linkedin.com/in/mahnerak), [Tigran](https://www.linkedin.com/in/galstyantik), [Hrayr](https://github.com/Harhro94), [Narek](https://www.linkedin.com/pub/narek-hovsepyan/86/b35/380) (1st to 3rd year bachelor students) and [me](https://github.com/Hrant-Khachatrian) (PhD student). Long story short, we finished at the [82nd place](https://www.kaggle.com/c/diabetic-retinopathy-detection/leaderboard), and in this post I will describe in details what we did and what mistakes we made. We hope this will be interesting for those who just start to play with neural networks. Also we hope to get feedback from experts and other participants.

* TOC
{:toc}

## The contest
[Diabetic retinopathy](https://en.wikipedia.org/wiki/Diabetic_retinopathy) is a disease when the retina of the eye is damaged due to diabetes. It is one of the leading causes of blindness in the world. The contest's aim was to see if computer programs can diagnose the disease automatically from the image of the retina. [It seems](https://www.kaggle.com/c/diabetic-retinopathy-detection/forums/t/15605/human-performance-on-the-competition-data-set) the winners slightly surpassed the performance of general ophthalmologists. 

Each eye of the patient can be in one of the 5 levels: from 0 to 4, where 0 corresponds to the healthy state and 4 is the most severe state. Different eyes of the same person can be at different levels (although some contestants managed to leverage the fact that two eyes are not completely independent). Contestants [were given](https://www.kaggle.com/c/diabetic-retinopathy-detection/data) 35126 JPEG images of retinas for training (32.5GB), 53576 images for testing (49.6GB) and a CSV file where level of the disease is written for the train images. The goal was to create another CSV file where disease levels are written for each of the test images. Contestants could submit maximum 5 CSV files per day for evaluation. 

|![Healthy eye: level 0](/public/2015-08-15/eye-0.jpeg "Healthy eye: level 0") | ![Severe state: level 4](/public/2015-08-15/eye-4.jpeg "Severe state: level 4") |
| --- | --- |
| Healthy eye: level 0 | Severe state: level 4 | 

The score was evaluated using a metric called **quadratic weighted kappa**. It is [described](https://www.kaggle.com/c/diabetic-retinopathy-detection/details/evaluation) as being an _agreement_ between two raters, in this case: the agreement between the scores assigned by human rater (which is unknown to contestants) and the predicted scores. If the agreement is random, the score is close 0 (although it can be negative). In case of a perfect agreement the score is 1. It is _quadratic_ in a sense that, for example, if you predict level 4 for a healthy eye, it is 16 times worse than if you predict level 1. Winners achieved a score [more than 0.84](https://www.kaggle.com/c/diabetic-retinopathy-detection/leaderboard). Our best result was around 0.50.

## Software and hardware
It was obvious that we were going to use [convolutional neural networks](https://www.youtube.com/watch?v=rxKrCa4bg1I&index=69&list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH) for predicting. Not only because of its [awesome performance](https://en.wikipedia.org/wiki/Convolutional_neural_network#Applications) on many computer vision problems, including another Kaggle competition on [plankton classification](https://www.kaggle.com/c/datasciencebowl), but also because it was the only technique we knew for image classification. We were aware of several libraries that implement convolutional networks, namely Python-based [Theano](http://deeplearning.net/software/theano/), [Caffe](http://caffe.berkeleyvision.org/) written in C++, [cxxnet](https://github.com/dmlc/cxxnet) (developed by the [2nd  place winners](https://www.kaggle.com/c/datasciencebowl/forums/t/12887/brief-describe-method-and-cxxnet-v2/69545) of the plankton contest) and [Torch](https://github.com/torch/nn/). We chose Caffe because it seemed to be the simplest one for beginners: it allows to define the neural network by a simple text file (like [this](https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt)) and train a network without writing a single line of code.  

We didn't have a computer with CUDA-enabled GPU in the university, but our friends at [Cyclop Studio](http://cyclopstudio.com/) donated us an Intel Core i5 computer with 4GB RAM and [NVidia GeForce GTX 550 TI](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-550ti/specifications) card. 550 TI has a 1GB of memory which forced us to use very small batch sizes for the neural network. Later we switched to [GeForce GTX 980](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-980/specifications) with 4GB memory, which was completely fine for us.

Karen and Tigran managed to [install Caffe on Ubuntu](http://caffe.berkeleyvision.org/install_apt.html) and make it work with CUDA, which was enough to start the training. Later Narek and Hrayr found out how to play with Caffe models [using Python](https://github.com/BVLC/caffe/tree/master/python/caffe), so we can run our models on the test set.

## Image preprocessing
Images from the training and test datasets have very different resolutions, aspect ratios, colors, are cropped in various ways, some are of very low quality, are out of focus etc. Neural networks require a fixed input size, so we had to resize / crop all of them to some fixed dimensions. Karen and Tigran looked at many sample images and decided that optimal resolution which preserves the details required for classification is 512x512. We thought that in 256x256 we might lose the small details that differ healthy eye images from level 1 images. In fact, by the end of the competition we saw that our networks cannot differentiate between level 0 and 1 images even with 512x512, so probably we could safely work on 256x256 from the very beginning (which would be much faster to train). All preprocessing was done using [imagemagick](http://www.imagemagick.org/).

We tried three methods to preprocess the images. First, as suggested by Karen and Tigran, we resized the images and then applied the so called _[charcoal](http://www.imagemagick.org/Usage/transform/#charcoal)_ effect which is basically an edge detector. This highlighted the signs of blood on the retina. One of the challenging problems throughout the contest was to define a naming convention for everything: databases of preprocessed images, convnet descriptions, models, CSV files etc. We used the prefix "_edge_" for anything which was based on the images preprocessed this way. 

|![_edge_ level 0](/public/2015-08-15/eye-edge-0.jpg "_edge_ level 0") | ![_edge_ level 3](/public/2015-08-15/eye-edge-3.jpg "_edge_ level 3") |
| --- | --- |
| Preprocessed image _(edge)_ level 0 | Preprocessed image _(edge)_ level 3 | 

But later we noticed that this method makes the dirt on lens or other optical issues appear similar to a blood, and it really confused our neural networks. The following two images are of healthy eyes (level 0), but both were recognized by almost all our models as level 4.

|![healthy eye](/public/2015-08-15/orig-35297_left-0.jpeg "healthy eye") | ![_edge_, recognized as level 4](/public/2015-08-15/edge-35297_left-0.jpeg "_edge_, recognized as level 4") |
|![healthy eye](/public/2015-08-15/orig-44330_left-0.jpeg "healthy eye") | ![_edge_, recognized as level 4](/public/2015-08-15/edge-44330_left-0.jpeg "_edge_, recognized as level 4") |
| --- | --- |
| Original images of healthy eyes | Preprocessed versions _(edge)_ recognized as level 4 |

So we decided to avoid using filters on the images, to leave all the work to the convolutional network: just resize and convert to one channel image (to save space and memory). We thought that the color information is not very important to detect the disease, although this could be one of our mistakes. Following the discussion at [Kaggle forums](https://www.kaggle.com/c/diabetic-retinopathy-detection/forums/t/13147/rgb-or-grayscale/69138) we decided to use the green channel only. We got our best results (kappa = 0.5) on this dataset. We used prefix _g_ for these images.

Finally we tried to apply the [_equalize_](http://www.imagemagick.org/Usage/color_mods/#equalize) filter on top of the green channel, which makes the histogram of the image uniform. The best kappa score we managed to get on the dataset preprocessed this way was only 0.4. We used prefix _ge_ for these images.
 
|![Just the green channel: _(g)_](/public/2015-08-15/g-99_left-3.jpeg "Just the green channel: _(g)_") | ![Histogram equalization on top of the green channel: _(ge)_](/public/2015-08-15/ge-99_left-3.jpeg "Histogram equalization on top of the green channel: _(ge)_") |
| --- | --- |
| Just the green channel: _(g)_ | Histogram equalization on top of the green channel: _(ge)_ |
 

## Data augmentation

## Training / validation sets separation
 
## Convolution network architecture

## Loss function

## Preparing submissions, attempts to ensemble

## More on this contest

## Acknowledgements

_to be continued_