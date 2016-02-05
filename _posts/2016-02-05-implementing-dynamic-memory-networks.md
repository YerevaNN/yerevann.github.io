---
layout: post
title: Implementing Dynamic memory networks
tags:
- Recurrent neural networks
- Natural language processing
- draft
---

The Allen Institute for Artificial Intelligence has organized a 4 month [contest](https://www.kaggle.com/c/the-allen-ai-science-challenge) in Kaggle on question answering. The aim is to create a system which can correctly answer the questions from the 8th grade science exams of US schools (biology, chemistry, physics etc.). DeepHack Lab organized a [scientific school + hackathon](http://qa.deephack.me/) devoted to this contest in Moscow. Our team decided to use this opportunity to explore the deep learning techniques on question answering (although they seem to be far behind traditional systems). We tried to implement the Dynamic memory networks described [in a paper by A. Kumar et al.](http://arxiv.org/abs/1506.07285). Here we report some preliminary results. In the next blog post we will describe the techniques we used to get to top 5% in the contest.

<!--more-->

## Contents
{:.no_toc}
* TOC
{:toc}

## bAbI tasks

The questions of this contest are quite hard, they not only require lots of knowledge in natural sciences, but also abilities to make inferences, generalize the concepts, apply the general ideas to the examples and so on. The methods based on deep learning do not seem to be mature enough to handle all of these difficulties. On the other hand these questions have 4 answer candidates. That's why, as was noted by [Dr. Vorontsov](https://www.youtube.com/watch?v=lM2-Mi-2egM), simple search engine indexed on lots of documents will perform better as a question answering system than any "intelligent" system. 

But there is already some work on creating question answering / reasoning systems using neural approaches. As another lecturer of the DeepHack event, [Tomas Mikolov](https://www.youtube.com/watch?v=gi4Zf59_IcU), told us, we should start from easy, even synthetic questions and try to gradually increase the difficulty. This roadmap towards building intelligent question answering systems is described in [a paper](http://arxiv.org/abs/1502.05698) by Facebook researchers Weston, Bordes, Chopra, Rush, Merriënboer and Mikolov, where the authors introduce a benchmark of toy questions called [bAbI tasks](http://fb.ai/babi) which test several basic reasoning capabilities of a QA system. 

Questions in the bAbI dataset are grouped into 20 types, each of them has 1000 samples for training and another 1000 samples for testing. A system is said to have passed a given task, if it correctly answers at least 95% of the questions in the test set. There is also a version with 10K samples, but as Mikolov told during the lecture, deep learning is not necessarily about large datasets, and in this setting it is more interesting to see if the systems can learn answering questions by looking at a few training samples. 

|![some of the bAbI tasks](/public/2016-02-06/babi1.png "some of the bAbI tasks") |
| --- |
|![some of the bAbI tasks](/public/2016-02-06/babi2.png "some of the bAbI tasks") |
| --- |
| Some of the bAbI tasks. More examples can be found in the [paper](http://arxiv.org/pdf/1502.05698v10.pdf). | 


## Memory networks

bAbI tasks were first evaluated on an LSTM-based system, which achieve 50% performance on average and do not pass any task. Then the authors of the paper try [Memory Networks](http://arxiv.org/abs/1410.3916) by Weston et al. It is a recurrent network which has a long-term memory component where it can learn to write some data (the input sentences) and read them later. 

bAbI tasks include not only the answers to the questions but also the numbers of those sentences which help answer the question. This information is taken into account when training MemNN, they not only get the correct answers but also an information about which input sentences affect the answer. Under this so called _strongly supervised_ setting "plain" Memory networks pass 7 of the 20 tasks. Then the authors apply some modifications to them and pass 16 tasks.

|![End-to-end memory networks](/public/2016-02-06/memn2n.png "End-to-end memory networks") |
| --- |
| The structure of MemN2N from the [paper](http://arxiv.org/abs/1410.3916). | 

We are mostly interested in _weakly supervised_ setting, because the additional information on important sentences is not available in many real scenarios. This was investigated in a paper by Sukhbaatar, Szlam, Weston and Fergus (from New York University and Facebook AI Research) where they introduce [End-to-end memory networks](http://arxiv.org/abs/1503.08895) (MemN2N). They investigate many different configurations of these systems and the best version passes 9 tasks out of 20. Facebook's MemN2N repository on GitHub lists [some implementations of MemN2N](https://github.com/facebook/MemNN).

## Dynamic memory networks

Another advancement in the direction of memory networks was made by Kumar, Irsoy, Ondruska, Iyyer, Bradbury, Gulrajani and Socher from Metamind. By the way, Richard Socher is the author of [an excellent course on deep learning and NLP](http://cs224d.stanford.edu/) at Stanford, which helped as a lot to get into the topic. Their [paper](http://arxiv.org/abs/1506.07285) introduces a new system called Dynamic memory networks (DMN) which passes 18 bAbI tasks in the strongly supervised setting. The paper does not talk about weakly supervised setting, so we decided to implement DMN from scratch in [Theano](http://deeplearning.net/software/theano/).

|![High-level structure of DMN](/public/2016-02-06/dmn-high-level.png "High-level structure of DMN") |
| --- |
| High-level structure of DMN from the [paper](http://arxiv.org/abs/1506.07285). | 

The input of the DMN is a sequence of word vectors of input sentences. We followed the paper and used pretrained [GloVe vectors](http://nlp.stanford.edu/projects/glove/) and added the dimensionality of word vectors to the list of hyperparamaters (controlled by the command line argument `--word_vector_size`). DMN architecture treats these vectors as part of a so called `Semantic memory` (in contrast to the `Episodic memory`) which may contain other knowledge as well. Our implementation uses only word vectors and does *not* fine tune them during the training, so we don't consider it as a part of the neural network.

The first module of DMN is an _input module_ that is a [gated recurrent unit](http://arxiv.org/abs/1412.3555) (GRU) running on the sequence of word vectors. GRU is a recurrent unit with 2 more gates that control when its content is updated and when its content is flushed. Its hidden state is meant to represent the input processed so far in a vector. Input module outputs its hidden states either after every word (`--input_mask word`) or after every sentence (`--input_mask sentence`). These outputs are called `facts`.


|![Formal definition of GRU](/public/2016-02-06/gru.png "Formal definition of GRU") |
| --- |
| Formal definition of GRU. `z` is the _update gate_ and `r` is the _reset gate_. More details and images can be found [here](http://deeplearning4j.org/lstm.html). | 

Then there is a _question module_ that processes the question word by word and outputs one vector at the end. This is done by using the same GRU as in the input module using the same weights.

The fact and question vectors extracted from the input enter the _episodic memory_ module. Episodic memory is basically a composition of two nested GRUs. The outer GRU generates the final memory vector working over a sequence of so called _episodes_. This GRU state is initialized by the question vector. The inner GRU generates the episodes.

|![Details of DMN architecture](/public/2016-02-06/dmn-details.png "Details of DMN architecture") |
| --- |
| Details of DMN architecture from the [paper](http://arxiv.org/abs/1506.07285). | 

The inner GRU generates the episodes by passing over the facts from the input module. But when updating its inner state, the GRU takes into account the output of some `attention function` on the current fact. Attention function gives a score (between 0 and 1) to each of the fact, and GRU (softly) ignores the facts having low scores. Attention function depends on the question vector, current fact, and current state of the memory. After each full pass on all facts the inner GRU outputs an episode which is fed into the outer GRU which updates the memory. Then because of the updated memory the attention may give different scores to the facts. The number of step of the outer GRU, that is the number of episodes, can be determined dynamically, but we fix it to simplify the implementation. It is configured by `--memory_hops` setting.

All facts, episodes and memories are in the same `n`-dimensional space, which is controlled by the command line argument `--dim`. Inner and outer GRUs share the weights.

The final state of the memory is being fed into the _answer module_, which produces the answer. We have implemented two kinds of answer modules. First is a simple linear layer on top of the memory vector with softmax activation (`--answer_module feedforward`). This is useful if each answer is just one word (like in the bAbI dataset). The second kind of answer module is another GRU that can produce multiple words (`--answer_module recurrent`). Its implementation is half baked now, as we didn't need it for bAbI.  

The whole system is end-to-end differentiable and is trained using stochastic gradient descent. We use [`adadelta`](http://arxiv.org/abs/1212.5701) by default.

## Initial experiments



## Next steps

Visualizing

Overfitting.

Answer choices.

Evaluate on MCTest.




All related files are in our [Github repository](https://github.com/YerevaNN/char-rnn-constitution).