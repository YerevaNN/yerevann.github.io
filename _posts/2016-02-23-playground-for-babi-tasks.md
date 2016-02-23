---
layout: post
title: Playground for bAbI tasks
tags:
- Recurrent neural networks
- Natural language processing
- Visualization
---

Recently we have [implemented]({% post_url 2016-02-05-implementing-dynamic-memory-networks %}) Dynamic memory networks in Theano and trained it on Facebook's bAbI tasks which are designed for testing basic reasoning abilities. Our implementation now solves 8 out of 20 bAbI tasks which is still behind state-of-the-art. Today we release a [web application](http://yerevann.com/dmn-ui/) for testing and comparing several network architectures and pretrained models.

<!--more-->

## Contents
{:.no_toc}
* TOC
{:toc}

## Attention module

One of the key parts in the DMN architecture, as described in the [original paper](http://arxiv.org/abs/1506.07285), is its attention system. DMN obtains internal representations of input sentences and question and passes these to the episodic memory module. Episodic memory passes over all the facts, generates _episodes_, which are finally combined into a _memory_. Each episode is created by looking at all input sentences according to some _attention_. Attention system gives a score for each of the sentences, and if the score is low for some sentence, it will be ignored when constructing the episode. 

Attention system is a simple 2 layer neural network where input is a vector of features computed based on input sentence, question and current state of the memory. This vector of features is described in the paper as follows:

![attention module input](/public/2016-02-23/attention-vector.png "attention module input")

where `c` is an input sentence, `q` is the question, `m` is the current state of the memory. We tried to stay as close to the original as possible in our first implementation, but probably we understood these expressions too literally. We implemented `|c-q|` as an [absolute value](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/blob/master/dmn_basic.py#L217) of a difference of two vectors, which caused lots of trouble, as Theano's implementation of (the gradient of) `abs` function gave `NaN`s at random during training. Then, the terms `cWq` and `cWm` actually produce [just two numbers](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/blob/master/dmn_basic.py#L215), and they do not affect anything in a large vector.
   
Later we implemented another version called [`dmn_smooth`](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/blob/master/dmn_smooth.py#L223) which uses Euclidean distance between two vectors (instead of `abs`). This version is much more stable and gives better results. It is interesting to note that this version trains faster on CPU than on our GPU (GTX 980). It could be because of our not so optimal code or some [issue](https://github.com/Theano/Theano/issues/1168) in Theano's `scan` function.

## Architecture extensions
The only significant difference between our implementation and the original DMN, as we understand it, is the fixed number of episodes. In the paper the authors describe a stop condition, so that the network decides if it needs to compute more episodes. We did not implement it yet.

Our implementations heavily overfit on many tasks. We tried several techniques to fight that, but with little luck. First, we have implemented a version of `dmn_smooth` which supports [mini-batch training](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/blob/master/dmn_batch.py). Then we applied [dropout](https://en.wikipedia.org/wiki/Dropout_(neural_networks)) and [batch normalization](http://arxiv.org/abs/1502.03167) on top of the memory module (before passing to the answer module). All of these tricks help for some tasks for some hyperparameters, but still we could not beat the results obtained using simple `dmn_smooth` trained without mini-batches.

We plan to bring some ideas from the [Neural Reasoner paper](http://arxiv.org/abs/1508.05508), especially the idea of recovering the input sentences based on the outputs of the input module.
 

## Results
We train our implementations on bAbI tasks in a weakly supervised setting, as described in our [previous post](http://yerevann.github.io/2016/02/05/implementing-dynamic-memory-networks/#memory-networks). Here we compare our results to [End-to-end memory networks](http://arxiv.org/abs/1410.3916).

So far our best results are obtained by training `dmn_smooth` with 100 neurons for internal representations, 5 memory hops, using simple gradient descent for 11 epochs. We train jointly on all 20 bAbI tasks. 

| Task | MemN2N best version | Joint100 75.05% |
| --- | ----- | ------ |
| 1. Single supporting fact |	**99.9%**	|	**100%**	|
| 2. Two supporting facts |	81.2%	|	39.7%	|
| 3. Three supporting facts |	68.3%	|	41.5%	|
| 4. Two argument relations |	82.5%	|	75.5%	|
| 5. Three arguments relations |	87.1%	|	50.1%	|
| 6. Yes/no questions |	**98%**	|	**97.7%**	|
| 7. Counting |	89.9%	|	91.4%	|
| 8. Lists/sets |	93.9%	|	**95.2%**	|
| 9. Simple negation |	**98.5%**	|	**99%**	|
| 10. Indefinite knowledge |	**97.4%**	|	87.3%	|
| 11. Basic coreference |	**96.7%**	|	**100%**	|
| 12. Conjuction |	**100%**	|	87%	|
| 13. Compound coreference |	**99.5%**	|	**96.4%**	|
| 14. Time reasoning |	**98%**	|	73.1%	|
| 15. Basic deduction |	**98.2%**	|	53.9%	|
| 16. Basic induction |	49%	|	49.5%	|
| 17. Positional reasoning |	57.4%	|	59.3%	|
| 18. Size reasoning |	90.8%	|	**98.3%**	|
| 19. Path finding |	9.4%	|	9%	|
| 20. Agent's motivations |	**99.8%**	|	**97.1%**	|
| **Average accuracy** |	**84.775%**	|	**75.05%**	|
| **Solved tasks** |	**10**	|	**8**	|

We solve (obtain >95% accuracy) 8 tasks. Our system overperforms MemN2N on some tasks, but on average stays behind by 10 percentage points. Experiments show that our networks do not manage to find connections between several sentences at once (tasks 2, 3 etc.). Task 19 (path finding) remains the most difficult one. It is actually the only task on which none of our implementations overfit. The authors of [Neural Reasoner](http://arxiv.org/abs/1508.05508) claim some success on that task when training on 10 000 examples. We use only 1000 samples per task for all experiments.

## Visualizing Dynamic memory networks

We have created a web application / playground for Dynamic memory networks focused on bAbI tasks. It allows to choose a pretrained model and send custom input sentences and questions. The app shows the predicted answer and visualizes attention scores for each memory step. 

| ![Playground for bAbI tasks](/public/2016-02-23/dmn-ui.png "Playground for bAbI tasks") |
| --- |
| Web-based [playground for bAbI tasks](http://yerevann.com/dmn-ui/) |

These visualizations show that the network does not significantly change its attention for different episodes, so it is very hard to correctly answer the questions from tasks 2 or 3.  

Web app is accessible at **[http://yerevann.com/dmn-ui/](http://yerevann.com/dmn-ui/)**. Note that the vocabulary of bAbI tasks is quite limited, and our implementation of DMN cannot process out-of-vocabulary words. `Sample` button is a good starting point, it gives a random sample from bAbI test set.

## Looking for feedback

Everything described in this post is available on Github. DMN implementations are [here](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano), Flask-based restful server of the web app is in the [/server/ folder](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/tree/master/server), UI is in [another repository](https://github.com/YerevaNN/dmn-ui). Feel free to fork, report issues, and please share your thoughts. 
