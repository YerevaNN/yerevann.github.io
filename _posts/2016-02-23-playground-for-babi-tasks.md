---
layout: post
title: Playground for bAbI tasks
tags:
- Recurrent neural networks
- Natural language processing
- Visualization
- draft
---

Recently we have [implemented]({% post_url 2016-02-05-implementing-dynamic-memory-networks %})) Dynamic memory networks in Theano and trained it on Facebook's bAbI tasks which are designed for testing basic reasoning abilities. Our implementation now solves 8 out of 20 bAbI tasks which is still behind state-of-the-art. Today we release a [web application](http://yerevann.com/dmn-ui/) for testing and comparing several network architectures and pretrained models.

<!--more-->

## Contents
{:.no_toc}
* TOC
{:toc}

## Architecture details

One of the key parts in the DMN architecture, as described in the [original paper](http://arxiv.org/abs/1506.07285), is its attention system. DMN obtains internal representations of input sentences and question and passes these to the episodic memory module. Episodic memory passes over all the facts, generates _episodes_, which are finally combined into a _memory_. Each episode is created by looking at all input sentences according to some _attention_. Attention system gives a score for each of the sentences, and if the score is low for some sentence, it will be ignored when constructing the episode. 

Attention system is a simple 2 layer neural network where input is a vector of features which are computed based on input sentence, question and current state of the memory. This vector of features is described in the paper as follows:

![attention module input](/public/2016-02-23/attention-vector.png "attention module input")

where `c` is an input sentence, `q` is the question, `m` is the current state of the memory. We tried to stay as close to the original as possible in our first implementation. But probably we understood these expressions too literally. We implemented `|c-q|` as an [absolute value](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/blob/master/dmn_basic.py#L217) of a difference of two vectors, which caused lots of trouble, as Theano's implementation of (the gradient of) `abs` function gave `NaN`s at random during training. Then, the terms `cWq` and `cWm` actually produce [just two numbers](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/blob/master/dmn_basic.py#L215), and they do not make a real difference in a large vector.
   
Later we implemented another version called [`dmn_smooth`](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/blob/master/dmn_smooth.py#L223) which uses Euclidean distance between two vectors (instead of `abs`). This version is much more stable and gives better results. It is interesting to note that this version trains faster on CPU than on our GPU (GTX 980). It could be because of our not so optimal code or a [known issue] in Theano's `scan` function.

## Architecture extensions
The only significant difference between our implementation and the original DMN, as we understand it, is the fixed number of episodes. In the paper the authors describe a stop condition, so that the network decides if it needs to compute more episodes. Our implementations do not implement this so far.

Our implementations heavily overfit on many tasks. We tried several techniques to fight that, but with little luck. First, we have implemented a version of `dmn_smooth` which supports [mini-batch training](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/blob/master/dmn_batch.py). Then we applied [dropout](https://en.wikipedia.org/wiki/Dropout_(neural_networks)) and [batch normalization](http://arxiv.org/abs/1502.03167) on top of the memory module (before passing to the answer module). All of these tricks help for some tasks for some hyperparameters, but still we could not beat the results obtained using simple `dmn_smooth` trained without mini-batches.

We plan to bring some ideas from the [Neural Reasoner paper](http://arxiv.org/abs/1508.05508), especially the idea of recovering the input sentences using the outputs of the input module.
 

## Results
We train our implementations on bAbI tasks in a weakly supervised setting, as described in our [previous post](http://yerevann.github.io/2016/02/05/implementing-dynamic-memory-networks/#memory-networks). Here we compare our results to [End-to-end memory networks](http://arxiv.org/abs/1410.3916).

So far our best results are obtained by training `dmn_smooth` with 100 neurons for internal representations, 5 memory hops, trained using simple gradient descent for 11 epochs. We train jointly on all 20 bAbI tasks. 

Table

We solve (obtain >95% accuracy) on 8 tasks. Experiments show that our networks do not manage to use several sentences at once (tasks 2, 3 etc.). Task 19 (positional reasoning) remains the most difficult one. It is actually the only task on which none of our implementations overfits. The authors of [Neural Reasoner] claim some success on that task when training on 10 000 examples. We use only 1000 samples per task for all experiments.

## Visualizing Dynamic memory networks

We have created a web application / playground for Dynamic memory networks focused on bAbI tasks. 

Screenshot.



Web app is accessible at [http://yerevann.com/dmn-ui/](http://yerevann.com/dmn-ui/).

## Give feedback and contribute!

Everything described in this post is available on Github. DMN implementations are [here](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano), the Flask-based server of the web app is in the [/server/ folder](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/tree/master/server), UI is [another repository](https://github.com/YerevaNN/dmn-ui). 
