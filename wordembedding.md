#自然语言处理 - 词向量 
##1、概述

One-hot Representation Distributed representation

![](https://github.com/Kevinwenya/WordEmbedding/blob/master/vector.png)

One-hot Representation 采用稀疏方式存储，会是非常的简洁:也就是给每个词分配一个数字 ID。比如上面举的例子中，老爷爷记为 3，老奶奶记为 4(假设从 0 开始记)。如果要编程实现的话，用 Hash 表给每个词分配一个编 号就可以了。这么简洁的表示方法配合上最大熵、SVM、CRF 等等算法已经很好地完成了 NLP 领域的各种主流任务。当然这种表示方法也存在一个重要的问题就是“词汇鸿沟”现象:任意两个词之间都是孤立的。光从这两个向量中看不出两个词是否有关系，哪怕是老爷爷和老奶奶这样的词也不能幸免于难。此外，这种表示方法还容易发生维数灾难，尤其是在 Deep Learning 相关的一些应用中。
既然上述这种易于理解的 One-hot Representation 词向量表示方式具有这样的重要缺陷，那么就需要一种既能表示词本身又可以考虑语义距离的词向量表示方法，这就是我们上述右边介绍的Distributed representation 词向量表示方法。

######             [0.792, −0.177, −0.107, 0.109, −0.542, ...]。
Distributed representation 最大的贡献就是让相关或者相似的词，在距离上更接近了(看 到这里大家有没有想到普通 hash 以及 simhash 的区别呢?其基于一种分布式假 说，该假说的思想是如果两个词的上下文(context)相同，那么这两个词所表达的语义也是一样的。换言之，两个词的语义是否相同或相似取决于两个词的上下文内容，上下文相同表示两个词是可以等价替换的。以上将 word 映射到一个新的空间中，并以多维的连续实数向量进行表示叫做“Word Representation”或“Word Embedding”。自从 21 世纪以来，人们 逐渐从原始的词向量稀疏表示法过渡到现在的低维空间中的密集表示。用稀疏 表示法在解决实际问题时经常会遇到维数灾难，并且语义信息无法表示，无法揭示 word 之间的潜在联系。而采用低维空间表示法，不但解决了维数灾难问题，并且挖掘了 word 之间的关联属性，从而提高了向量语义上的准确度。而分布式词向量表示，根据使用算法的不同，目前主要有 word2vec 和 glove 两大方法，本文主要通过 word2vec 方法训练中文维基百科语料库的词向量。使得每个词可以用一个密集的向量来表示，如下面这个例子所示:


Wikipedia Extractor 是一个开源的用于抽取维基百科语料库的工具，由 python 写成，且不需要依赖额外的库，抽取文本内容的过程可以参考上面的两 个链接来完成，其使用方法如下:
> 
>

经过上述两行命令处理之后，抽取得到的内容格式为每篇文章被一对<doc> </doc>包起来，而<doc>中的包含了属性有文章的 id、url 和 title 属性，如 <doc id="13" url="https://zh.wikipedia.org/wiki?curid=13" title="数学 ">。然后，将经第一步抽取得到的文件，进行去除标点符号处理。去除文本中的标点符号，可以直接通过正则表达式的方式进行处理，其处 理过程，如下面的 python 脚本所示:


```
from langconv import *

def simple2tradition(line):
    #将简体转换成繁体
    line = Converter('zh-hant').convert(line.decode('utf-8'))
    line = line.encode('utf-8')
    return line

def tradition2simple(line):
    # 将繁体转换成简体
    line = Converter('zh-hans').convert(line.decode('utf-8'))
    line = line.encode('utf-8')
    return line
```


![](https://github.com/Kevinwenya/WordEmbedding/blob/master/cut_words.png)

  经过上述的这些处理过程，我们基本得到了可以用于训练词向量的教卫纯净的语料。其二，使用 gensim 的 wikicorpus 库，可以参考下面的链接: [wikicorpus](http://radimrehurek.com/gensim/corpora/wikicorpus.html)

> 

```
sentences = gensim.models.word2vec.LineSentence(input_file) 
model = gensim.models.Word2Vec(sentences, size=300,

```


1、得到“数学”这个词的 300 维向量表示:

```
```


```
```


```
for word in words:
```


```
概率论 0.721196651459 算术 0.717466473579
```


##附:

```
 ```