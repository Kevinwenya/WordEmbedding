# 自然语言处理 - 词向量 
## 1、概述
若要通过计算机对文本等自然语言进行处理，通常首先需要将这些文本数字 化，也就是用向量来表示文本里的每个词语。
一种最简单的词向量方式是 one-hot representation，就是用一个很长的 向量来表示一个词，向量的长度为词典的大小，向量的分量只有一个 1，其他 全为 0，1 的位置对应该词在词典中的位置。
另一种就是 Distributed Representation，这种表示，最早是 Hinton 于 1986 年提出的，可以克服 one-hot representation 的缺点。其基本想法是: 通过训练将某种语言中的每一个词映射成一个固定长度的短向量(当然这里的 “短”是相对于 one-hot representation 的“长”而言的)，将所有这些向 量放在一起形成一个词向量空间，而每一向量则为该空间中的一个点，在这个 空间上引入“距离”，则可以根据词之间的距离来判断它们之间的(词法、语 义上的)相似性了。
#### 举几个通俗的例子:
1、现代人看到苹果，华为这两个词，第一眼的反应多数都是手机。但是如果 拿给古人看，古人一定想不到手机。为什么呢，因为古人没有相关知识，只 能从字面上去理解这两个词，即<苹,果>，<华,为>。拿给计算机，计算机看 到的也是字面上的意思，这两个字串是八竿子打不着(要是给计算机华为和 华山，它倒是能发现这俩词有点像)。那怎么才能让计算机把这俩词关系起 来呢，这就是统计学习干的事了，因为我们有很多资源可以利用，计算机可 以利用一些算法从这些资源中学习到词之间的关系，就像人类一样，天天听 别人说这手机是苹果，那手机是华为，久了就知道这俩东西都是手机了。但 是苹果在有些语境里也未必是手机。

2、问你这样一个问题:如果你大脑有很多记忆单元，让你记住一款白色奥迪 Q7 运动型轿车，你会用几个记忆单元?你也许会用一个记忆单元，因为这样最节省你的大脑。那么我们再让你记住一款小型灰色雷克萨斯，你会怎么办?显然你会用另外一个记忆单元来记住它。那么如果让你记住所有的车，你要耗费的记忆单元就不再是那么少了，这种表示方法叫做稀疏表达(典型代表就是 one-hot)。这时你可能会换另外一种思路:我们用几个记忆单元来分别识别大小、颜色、品牌等基础信息，这样通过这几个记忆单元的输出，我们就可以表示出所有的车型了。这种表示方法叫做分布式表达，词向量就是一种用 distributed representation 表示的向量

### One-hot Representation             Distributed representation

![](https://github.com/Kevinwenya/WordEmbedding/blob/master/vector.png)

One-hot Representation 采用稀疏方式存储，会是非常的简洁:也就是给每个词分配一个数字 ID。比如上面举的例子中，老爷爷记为 3，老奶奶记为 4(假设从 0 开始记)。如果要编程实现的话，用 Hash 表给每个词分配一个编 号就可以了。这么简洁的表示方法配合上最大熵、SVM、CRF 等等算法已经很好地完成了 NLP 领域的各种主流任务。当然这种表示方法也存在一个重要的问题就是“词汇鸿沟”现象:任意两个词之间都是孤立的。光从这两个向量中看不出两个词是否有关系，哪怕是老爷爷和老奶奶这样的词也不能幸免于难。此外，这种表示方法还容易发生维数灾难，尤其是在 Deep Learning 相关的一些应用中。
既然上述这种易于理解的 One-hot Representation 词向量表示方式具有这样的重要缺陷，那么就需要一种既能表示词本身又可以考虑语义距离的词向量表示方法，这就是我们上述右边介绍的Distributed representation 词向量表示方法。

Distributed representation 是一种低维实数向量，这种向量一般长成这个样子: 
######             [0.792, −0.177, −0.107, 0.109, −0.542, ...]。
Distributed representation 最大的贡献就是让相关或者相似的词，在距离上更接近了(看 到这里大家有没有想到普通 hash 以及 simhash 的区别呢?其基于一种分布式假 说，该假说的思想是如果两个词的上下文(context)相同，那么这两个词所表达的语义也是一样的。换言之，两个词的语义是否相同或相似取决于两个词的上下文内容，上下文相同表示两个词是可以等价替换的。以上将 word 映射到一个新的空间中，并以多维的连续实数向量进行表示叫做“Word Representation”或“Word Embedding”。自从 21 世纪以来，人们 逐渐从原始的词向量稀疏表示法过渡到现在的低维空间中的密集表示。用稀疏 表示法在解决实际问题时经常会遇到维数灾难，并且语义信息无法表示，无法揭示 word 之间的潜在联系。而采用低维空间表示法，不但解决了维数灾难问题，并且挖掘了 word 之间的关联属性，从而提高了向量语义上的准确度。而分布式词向量表示，根据使用算法的不同，目前主要有 word2vec 和 glove 两大方法，本文主要通过 word2vec 方法训练中文维基百科语料库的词向量。使得每个词可以用一个密集的向量来表示，如下面这个例子所示:
###### 男人:[-1.62051833 -1.08284032 -0.95101804......1.70183563 - 0.24874304 -0.6090101], 300 维词向量表示。
*如何得到上述结果的尼?整个过程是怎样的尼?需要掌握哪些知识尼?* 自然语言处理最基本也是最重要的就是语料知识，需要有一个强大的语料库。然后我们对该语料库进行简单的预处理(去除标点符号、繁简转换、去除文章 结构标识等一些基本操作、分词等一些基本操作)。然后再根据需要，进行相关的处理。这里我们是直接基于分词后的语料库，通过 python 的 gensim 库训练了 Word2Vec 模型([gensim_word2vec](http://radimrehurek.com/gensim/models/word2vec))， 当然，也可以基于分词后的语料库，使用 google 的 Word2Vec 工具 ([google_word2vec](https://code.google.com/archive/p/word2vec/))来训练语料。之后，就是根据训练好的 word2vec 模型，使用它来做相关的计算了。

## 2、语料库获取
中文语料库中，质量比较高且比较容易获取的语料资源应该就是维基百科的语料了，而且维基的语料每个月都会重新打包更新一次。官方提供了一个非常好的数据源，维基百科语料库的官方链接为:[wikimedia](https://dumps.wikimedia.org/)，从这里我们可以下载多种语言多种格式的百科数据。比如中文维基百科数据数据源为: [zhwiki](https://dumps.wikimedia.org/zhwiki/)， 这里我们下载了最新的包含标题和正文的版本数据:[zhwiki_latest](https://dumps.wikimedia.org/zhwiki/latest/),zhwiki-latest-pages-artices.xml.bz2 , 中文维基百科的数据并不大，1.4G左右。
[enwiki_latest](https://dumps.wikimedia.org/enwiki/latest/)，enwiki-latest-pages-artices.xml.bz2 ，英文维基百科的数据相对中文的大很多，13.2G左右。由此也可以看出维基百科的缺点，最主要的就是数量较少，相比国内的百度百科、互动百科等，数据量要少一个数量级。当然也可以采用其他的语料资源，比如搜狗的数据资源:[sogou_pingce](http://www.sogou.com/labs/resource/list_pingce.php)，甚至可以使用自己爬取 的语料数据。
## 3、语料预处理
因为从维基百科上下载的语料数据是 xml 压缩文件格式的，所以，需要进 行一些处理，使其成为较为纯净的文本格式数据。通常使用维基百科语料测试 其他任务之前，我们需要如下几步处理:
我们需要将 xml 格式的 wiki 数据转换为 text 格式，也就是对 wiki 的数据 内容进行抽取，这个抽取内容的过程可以通过两种方式来实现。
其一，使用 Wikipedia Extractor ([wikipedia_extractor](http://medialab.di.unipi.it/wiki/Wikipedia_Extractor));  https://github.com/attardi/wikiextractor

Wikipedia Extractor 是一个开源的用于抽取维基百科语料库的工具，由 python 写成，且不需要依赖额外的库，抽取文本内容的过程可以参考上面的两 个链接来完成，其使用方法如下:
> $ git clone https://github.com/attardi/wikiextractor.git wikiextractor
> 
> $ wikiextractor/WikiExtractor.py -b 1000M -o zhwiki_extracted zhwiki-latest-pages-articles.xml.bz2
>
-b 参数指对提取出来的内容进行切片后每个文件的大小，如果要将所有内容 保存在同一个文件，那么就需要把这个参数设得大一下。-o 参数指提取出来的 文件放置的目录，抽取出来的文件的路径为zhwiki_extracted/AA/wiki_00， 更多参数可参考其 github 主页的说明。

经过上述两行命令处理之后，抽取得到的内容格式为每篇文章被一对<doc> </doc>包起来，而<doc>中的包含了属性有文章的 id、url 和 title 属性，如 <doc id="13" url="https://zh.wikipedia.org/wiki?curid=13" title="数学 ">。然后，将经第一步抽取得到的文件，进行去除标点符号处理。去除文本中的标点符号，可以直接通过正则表达式的方式进行处理，其处 理过程，如下面的 python 脚本所示:

![](https://github.com/Kevinwenya/WordEmbedding/blob/master/pre_process.png)

经过上述的第二步处理后，我们观察得到的输出文件，还可以发现，文本 内容中含有繁体字符，为此，为了得到较为纯净的简体语料，我们还需要对 文本进行繁简转换。具体方法，可以通过使用开源的 opencc 来实现。关于 opencc 的使用，可以参考 [opencc](https://github.com/BYVoid/OpenCC), github上给出了较为详细的使用说明和相关介绍，这里不再细述。当然，也 可以采用别的方法:在当前目录下，存放两个 python 脚本 langconv.py、 zh_wiki.py，两个文件可以从如下的路径下载:
  zh_wiki.py:
 [zh_wiki.py](https://github.com/skydark/nstools/blob/master/zhtools/zh_wiki.py)
  langconv.py:
 [langconv.py](https://github.com/skydark/nstools/blob/master/zhtools/langconv.py)
然后如下处理就可以。

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

经过上述的去标点、繁简转换一些处理后，基本上得到了较为纯净的简体中文文本语料，当然还存在一定的噪音，这个可以根据需求在做适当的清理，目前，对于构建词向量的训练语料来说，已达到可以接受的程度。
接下来需要对上述处理的语料进行分词，采用 python 的分词工具 jieba，且 将一篇文章分词后的结果存储在一行，最终得到分词后的较为纯净的维基百科语 料。jieba 在 Linux 或者 mac 下安装都比较方便，pip install jieba 或者 brew install jieba 安装即可。关于 jieba 的详细文档可以参考:  [jieba](https://github.com/fxsjy/jieba), 我们这里暂且只用到分词方法，所以,cut方法即可。那么如何将一篇文章分词后的结果存储在一行尼?由于我们在分 词之前处理繁简转换和去标点的时候，每篇文章是存储在一对<doc></doc>标签中，所以，只要判断当前行处理为do 时，即可认为文章结束，进而开始下一篇。实现的 python 代码如下:

![](https://github.com/Kevinwenya/WordEmbedding/blob/master/cut_words.png)

  经过上述的这些处理过程，我们基本得到了可以用于训练词向量的教卫纯净的语料。其二，使用 gensim 的 wikicorpus 库，可以参考下面的链接: [wikicorpus](http://radimrehurek.com/gensim/corpora/wikicorpus.html)
## 4、训练词向量模型
这里主要使用的是 python 版本的 gensim，通过 gensim 提供的 API 可以相 对比较容易的进行词向量训练。
#####          [gensim_word2vec](http://radimrehurek.com/gensim/models/word2vec.html)
训练时参考 gensim 下 word2vec 的 api 介绍，
>            http://radimrehurek.com/gensim/apiref.html
当然从这里可以看到gensim中不止提供了word2vec模型，还提供了TFIDF 、LSI 、 LDA 等模型。
关于 word2vec 的使用案例，也可参考 Radim 的一些文章:

>          https://rare-technologies.com/word2vec-tutorial/
>            http://radimrehurek.com/category/gensim/
> 
参考 gensim API，结合训练语料数据，主要训练脚本如下:

```
sentences = gensim.models.word2vec.LineSentence(input_file) 
model = gensim.models.Word2Vec(sentences, size=300,
min_count=10, sg=0, workers=multiprocessing.cpu_count()) model.save(output_file) model.save_word2vec_format(output_file + '.vector',
binary=True)

```
notice:首先将输入的文件转为 gensim 内部的 LineSentence 对象，其次, gensim.models.Word2Vec 初始化一个 Word2Vec 模型，size 参数表示训练的向量的数;min_count 表示忽略那些出现次数小于这个数值的词语，认为他们 是没有意义的词语，一般的取值范围为(0，100);sg 表示采用何种算法进行 训练，取 0 时表示采用 CBOW 模型，取 1 表示采用 skip-gram 模型;workers 表示开多少个进程进行训练，采用多进程训练可以加快训练过程，这里开的进程 数与 CPU 的核数相等。最后将训练后的得到的词向量存储在文件中，存储的格 式可以是 gensim 提供的默认格式(save 方法)，也可以与原始 c 版本 word2vec 的 vector 相同的格式(save_word2vec_format 方法)，加载时分别采用 load 方 法和 load_word2vec_format 方法即可，以上这些在 gensim word2vec API 中都 有介绍。

  如果我们已经基于现有的语料库训练了一个词向量模型，但是当这个语料库要扩充的时候，如何训练新增的这些文章，从而更新我们的模型尼?我们可以先加载我们先前已经训练好的词向量模型，然后再添加新的文章进行训练。同样新增的文章的格式也要满足每行一篇文章，每篇文章的词语通过空格 分开的格式。这里需要注意的是加载的模型只能 是通过 model.save()存储的模型，从 model.save_word2vec_format()恢复过来的模型只能用于查询。
## 5、使用词向量模型
通过 gensim 加载训练好的词向量模型,可以根据 gensim 提供的 API,进行其他的相关计算，现举几个例子如下: 

1、得到“数学”这个词的 300 维向量表示:

```
> print model[u'数学']
[ -9.39297318e-01 -1.00252163e+00 3.89001146e-02 1.59526956e+00 ......2.11555183e-01 1.37527204e+00 -6.25237226e- 01 -1.52071941e+00]
```

2、比如计算两个词[“数学和历史”]、[“数学和物理”]的相似度:

```
> print model.similarity(u'数学',u'历史')
> print model.similarity(u'数学',u'物理') [“数学和历史”]:0.379155901645 [“数学和物理”]: 0.502581810193
```

3、比如计算和给定词相似的词有哪些，仍以“数学”为例:

```
words = model.most_similar(u"数学") 
for word in words:
	print word[0], word[1]
```

数学:

```
微积分 0.759175658226 数学分析 0.742166996002 逻辑学 0.723565936089
model = gensim.models.Word2Vec.load(exist_model) model.train(new_sentences)
model = gensim.models.Word2Vec.load_word2vec_format("wiki.text.vector", binary=False)
       
概率论 0.721196651459 算术 0.717466473579
数论 0.715003728867 纯数学 0.710333704948 几何学 0.704210162163 计算机科学 0.701360404491 拓扑学 0.69933784008
```

当然关于词向量，还有很多其他有趣的性质，这个可以通过查看API实现。

## 附:
##### (1)利用 Google 开源的 word2vec 工具
首先我们从网上下载一个源码，因为 google 官方的 svn 库已经不在了，所以只能从 csdn 下了，但是因为还要花积分才能下载，所以我干脆分享到了我的 git 上 (https://github.com/warmheartli/ChatBotCourse/tree/master/word2vec)，大家可以直接下载下来后直接执行 make 编译(如果是 mac 系统要把代码里所有的#include <malloc.h> 替换成#include <sys/malloc.h>)
编译后生成 word2vec、word2phrase、word-analogy、distance、compute-accuracy 几个二进制文件我们先用 word2vec 来训练首先我们要有训练语料，其实就是已经切好词(空格分隔)的文本，比如我们已经有了这个文 本文件叫做 train.txt，内容是"人工 智能 一直 以来 是 人类 的 梦想 造 一台 可以 为 你 做 一切 事情 并且 有 情感 的 机器 人"并且重复100遍会生成一个 vectors.bin 文件，这个就是训练好的词向量的二进制文件，利用这个文件我们可以求近义词了
##### (2)利用tensorflow深度学习框架生成词向量(wordembedding)

```
 https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py
 语料:百度百科 660 万篇文档
 词汇总数:2363511
 向量维度:200 维
 语料文件:baikew2v.bin
 语料大小:1.9G 训练工具:gensim
 主要训练参数:mincount=10,window=5,size=200
 ```
