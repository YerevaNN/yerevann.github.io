---
layout: post
title: Generating Constitution with recurrent neural networks
tags:
- draft
- Recurrent neural networks
- Natural language processing
- Armenian
---

By [Narek Hovsepyan](https://github.com/hnhnarek) and [Hrant Khachatrian](https://github.com/Hrant-Khachatrian)

Few months ago [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/) wrote a [great blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) about recurrent neural networks. He explained how these networks work and implemented a character-level RNN language model which learns to generate Paul Graham essays, [Shakespeare works](http://cs.stanford.edu/people/karpathy/char-rnn/shakespear.txt), [Wikipedia articles](http://cs.stanford.edu/people/karpathy/char-rnn/wiki.txt), [LaTeX articles](http://cs.stanford.edu/people/jcjohns/fake-math/4.pdf) and even C++ code. He also released the code of the network on [Github](https://github.com/karpathy/char-rnn). Lots of people did experiments, like generating [recipes](https://gist.github.com/nylki/1efbaa36635956d35bcc), [Bible](http://cpury.github.io/learning-holiness/) or [Irish folk music](https://soundcloud.com/seaandsailor/sets/char-rnn-composes-irish-folk-music). We decided to test it on some legal texts in Armenian.
  
<!--more-->

## Contents
{:.no_toc}
* TOC
{:toc}

## Character-level RNN language model

Andrej did a great job explaining how the recurrent networks learn and even visualized how they work on text input in [his blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). The program, called `char-rnn`, treats the input as a sequence of characters and has no prior knowledge about them. For example, it doesn't know that the text is in English, that there are words and there are sentences, that the space character has a special meaning and so on. After some training it manages to figure out that some character combinations appear more often than the others, learns to predict English words, uses proper punctuation, and even understands that open parentheses must be closed. When trained on Wikipedia articles it can generate text in MediaWiki format without syntax errors, although the text has little or no meaning. 

## Data

We decided to test Karpathy's RNN on Armenian text. Armenian language has a [unique alphabet](https://en.wikipedia.org/wiki/Armenian_alphabet), and the characters are encoded in the Unicode space by the codes [U+0530 - U+058F](http://www.unicode.org/charts/PDF/U0530.pdf). In UTF-8 these symbols use two bytes where the first byte is always `0xD4`, `0xD5` or `0xD6`. So the neural net has two look at almost 2 times larger distances (when compared to English) in order to be able to learn the words. Also, the Armenian alphabet contains 39 letters, 50% more than Latin.

Recently the main political topic in Armenia is the Constitutional reform. This helped us to choose the corpus for training. We took all three versions of the Constitution of Armenia (the [first version](http://www.arlis.am/documentview.aspx?docID=1) voted in 1995, the [updated version](http://www.arlis.am/documentview.aspx?docID=75780) of 2005, and the [new proposal](http://moj.am/storage/uploads/Sahmanadrakan_1-15.docx) which will be voted later this year) and concatenated them in a [single text file](https://github.com/YerevaNN/char-rnn-constitution/blob/master/data/input.txt). The size of the corpus is just 440 KB, which is roughly 224 000 Unicode symbols (all non-Armenian symbols, including spaces and numbers use 1 byte). Andrej suggests to use at least 1MB data, so our corpus is very small. On the other hand the text is quite specific, the vocabulary is very small and the structure of the text is fairly simple.

All articles are of the following form:

{% highlight text %}
Հոդված 1. Հայաստանի Հանրապետությունը ինքնիշխան, ժողովրդավարական, սոցիալական, իրավական պետություն է:
{% endhighlight %}

The first word, `Հոդված`, means "Article". Sentences end with the symbol `:`. 

## Network parameters

`char-rnn` works with basic recurrent neural networks, LSTM networks and GRU-RNNs. In our experiments we only used LSTM network with 2 layers. Actually we don't really understand how LSTM networks work in details, but we hope to improve our understanding by watching the videos of Richard Socher's excellent [NLP course](http://cs224d.stanford.edu/index.html). 

We trained the network for 50 epochs with the default learning rate parameters (base rate is `2e-3`, which decays by a factor of `0.97` after each `10` epochs). We wanted to understand how the size of LSTM internal state (`rnn_size`), [dropout](https://www.youtube.com/watch?v=UcKPdAM8cnI) and batch size affect the performance. We used [grid search](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search) over the following values:

* `rnn_size`: `128`, `256`, `512`
* `batch_size`: `25`, `50`, `100`
* `dropout`: `0`, `0.2`, `0.4` and at the end we tried `0.6`

After installing Lua, Torch and CUDA (as described on [`char-rnn` page](https://github.com/karpathy/char-rnn#requirements)) we have moved our mini-corpus to `/data/input.txt` and ran the [`run.sh` file](....), which contains commands like this:

{% highlight bash %}
th train.lua -data_dir data/ -batch_size 50 -dropout 0.4 -rnn_size 512 -gpuid 0 -savefile bs50s512d0.4 | tee log_bs50s512d0.4
{% endhighlight %}

File names encode the hyperparameters, and the output of `char-rnn` is logged using [`tee` command](https://en.wikipedia.org/wiki/Tee_(command)).

## Analysis

We have adapted [this script](https://github.com/YerevaNN/Caffe-python-tools/blob/master/plot_loss.py) written by Hrayr to plot the behavior of loss functions during the 50 epochs. The script, which runs on `char-rnn` output is available on [Github](https://github.com/YerevaNN/char-rnn-constitution/blob/master/plot_loss.py). These graphs show, for example, that we practically do not gain anything after 25 epochs.

|![Training and validation loss](/public/2015-11-11/plot_bs50s256all.png "Training and validation loss") |
| --- |
| Training (blue to aqua) and validation (red to green) loss over 50 epochs. RNN size was set to 256 and the batch size was 50. In particular, this graph shows that when no dropout is used, validation loss actually increases after 20 epochs. Plotted using [this script](https://github.com/YerevaNN/char-rnn-constitution/blob/master/plot_loss.py). | 

Experiments showed that, unsuprisingly, training loss is better (after 50 epochs) when RNN size is increased and when dropout ratio is decreased. Under all configurations we got the lowest train losses using batch size `50` (compared to `25` and `100`) and we don't have explanation for this.
   
For validation loss, we have the following tables.

|	 			| **Dropout**	| **0**	 | **0.2**	| **0.4**	| **0.6**	|
| **Batch size**| **RNN Size** 	| 	 	 | 			| 			| 			|
| **25** 		| 128 | 0.5060 | 0.4307 | 0.4813 | 0.5373 |
| 				| `- `256 | `- `0.5322 | `- `0.4185 | `- `0.4021 | `- `0.4261 |
| 				| `- - `512 | `- - `0.5596 | `- - `0.4495 | `- - `0.4380 | `- - `0.4126 |
| **50** 		| 128 | 0.4883 | 0.4452 | 0.4813 | 0.5373 |
| 				| `- `256 | `- `0.5249 | `- `0.3887 | `- `0.3996 | `- `0.4280 |
| 				| `- - `512 | `- - `0.5340 | `- - `0.4420 | `- - `0.3997 | `- - `0.3800 |
| **100**		| 128 | 0.5341 | 0.5144 | 0.5454 | 0.6094 |
| 				| `- `256 | `- `0.5660 | `- `0.4464 | `- `0.4500 | `- `0.4723 |
| 				| `- - `512 | `- - `0.6032 | `- - `0.4804 | `- - `0.4599 | `- - `0.4399 |

When RNN size is only `128`, we notice that the best performance is achieved when dropout is `0.2`. Larger dropout values do not allow the network to learn enough. When RNN size is increased to `256`, the optimal dropout value is somewhere between `0.2` and `0.4`. For RNN size `512`,  the best performance we observed using `60%` dropout. We didn't try to go any further. 

As for batch sizes, we see the best performance on `25` if the RNN size is only `128`. For larger networks, batch size `50` performs better. Overall we obtained the lowest validation score, `0.38`, using `60%` dropout, `50` batch size and `512` RNN size.

## Generated samples

When the trained models are ready, we can generate text samples by using `sample.lua` script included in the repository. It accepts one important parameter called `temperature` which determines how much the network can "fantasize". Higher temperature gives more diversity but at a cost of making more mistakes, as Andrej explains in his blog post. The command looks like this
 
{% highlight bash %}
th sample.lua cv/lm_bs50s128d0_epoch50.00_0.4883.t7 -length 3000 -temperature 0.5 -gpuid 0 -primetext "Հոդված"
{% endhighlight %}

`primetext` parameter allows to give the first characters of the generated sequence. Also it makes the output fully reproducible. Here is a snippet from `bs50s128d0` model, which is available [on Github](....).

{% highlight text %}
Հոդված 111. Սահմանադրական դատարանի կազմավորումը, եթե այլ չեն _հասատատիրի_ _առնչամի_ կարելի սահմանափակվել միայն օրենքով, եթե դա անհրաժեշտ է հանցագործությունների իրավունք:
Յուրաքանչյուր ոք ունի Հայաստանի Հանրապետության քաղաքացիությունը որոշում է կայացնում դատավորին կազմավորման կարգը 
1. Հանրապետության նախագահի կամ նախատեսված դեպքերում նշանակվում է նաև տնտեսական մշակույթի հիմնական ուղղության դրանք _կայտարվակատությունն_ է: Նրանց զինված ուժերի օգտագործման նախարարներից ստացված փոխառությունների կողմից ընդունվում է ընտրված և միջա
զգային _պայմանագվին_ պաշտոնները սահմանվում են օրենքով: 
{% endhighlight %}

There are only 4 non-existent words here (marked by italic), others are completely fine. The sentences have no meaning, some parts are so unnatural that are even difficult to read.

The network easily (even with `128` RNN size) learns to separate the articles by new line and start by the word `Հոդված` followed by some number. But even the best one doesn't manage to use increasing numbers for articles. Actually, very often the article number starts with `1`, because more than one third of the articles in the corpus have numbers starting with `1`.

The simplest version (`128` RNN size, no dropout) sometimes makes "typos" in the text.

When the number of 
512 - large chunks
128 - new words

numeration is wrong
most articles start with 1

## NaNoGenMo

