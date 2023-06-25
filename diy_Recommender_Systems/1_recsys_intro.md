# 1 Overview of Recommender Systems

- 推荐系统对个人用户和行业很重要。**协同过滤 CF 是推荐中的一个关键概念。**
- 反馈有两种类型：隐式反馈（点击率）和显式反馈（评分）。**在过去十年中，已经探索了许多推荐任务。**

In the last decade, the Internet has evolved into a platform for large-scale online services, which profoundly changed the way we communicate, read news, buy products, and watch movies. In the meanwhile, the unprecedented number of items (we use the term *item* to refer to movies, news, books, and products.) offered online requires a system that can $\text{\color{yellow}\colorbox{black}{help us discover items}}$ that we preferred. $\text{\color{red}\colorbox{black}{Recommender systems}}$ are therefore powerful information filtering tools that can facilitate personalized services and provide tailored experience to individual users. $\text{\color{yellow}\colorbox{black}{In short}}$, recommender systems play a pivotal role in utilizing the wealth of data available to make choices manageable. Nowadays, recommender systems are at the core of a number of online services providers such as **Amazon, Netflix, and YouTube**. Recall the example of Deep learning books recommended by Amazon in [Fig. 1.3.3]().

<center>
    <img style="border-radius: 0.1125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 4px 0 rgba(34,36,38,.08);" 
    src="https://d2l.ai/_images/stackedanimals.png" width = "20%"/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 4px;">
      Fig. 1.3.3 A donkey, a dog, a cat, and a rooster.
  	</div>
</center>

The $\text{\color{yellow}\colorbox{black}{benefits}}$ of employing recommender systems are two-folds:

- On the one hand, it can largely $\text{\color{red}{reduce users’ effort}}$ in finding items and alleviate the issue of information overload.
- On the other hand, it can $\text{\color{red}{add business value}}$ to online service providers and is an important source of revenue.

This chapter will introduce the $\text{\color{red}\colorbox{black}{fundamental concepts}}$, $\text{\color{red}\colorbox{black}{classic models}}$ and $\text{\color{red}\colorbox{black}{recent advances}}$ with deep learning in the field of recommender systems, together with implemented examples.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://d2l.ai/_images/rec-intro.svg"/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig. 17.1.1 Illustration of the Recommendation Process¶
  	</div>
</center>

## 1.1 Collaborative Filtering 协同过滤

We start the journey with the $\text{\color{red}\colorbox{white}{important concept}}$ in recommender systems—$\text{\color{red}\colorbox{black}{collaborative filtering (CF)}}$, which was first coined by the Tapestry system [[Goldberg et al., 1992](https://d2l.ai/chapter_references/zreferences.html#id86 "Goldberg, D., Nichols, D., Oki, B. M., & Terry, D. (1992). Using collaborative filtering to weave an information tapestry. Communications of the ACM, 35(12), 61–71.")], $\text{\color{yellow}\colorbox{black}{referring to}}$ “people collaborate to help one another $\textbf{\color{black}\colorbox{white}{perform the filtering process}}$ $\text{\color{yellow}\colorbox{black}{in order to}}$ handle the large amounts of email and messages posted to newsgroups”. This term has been enriched with more senses. $\text{\color{yellow}\colorbox{black}{In a broad sense}}$, it is the process of $\textbf{\color{black}\colorbox{white}{filtering for information or patterns}}$ using techniques involving collaboration among multiple users, agents, and data sources. CF has many forms and numerous CF methods proposed since its advent.

该技术通过分析用户或者事物之间的相似性（“协同”），来预测用户可能感兴趣的内容并将此内容推荐给用户。

Overall, CF techniques can be categorized into:

- memory-based CF,
- model-based CF, and
- their hybrid [[Su &amp; Khoshgoftaar, 2009](https://d2l.ai/chapter_references/zreferences.html#id262 "Su, X., & Khoshgoftaar, T. M. (2009). A survey of collaborative filtering techniques. Advances in artificial intelligence, 2009.")].

$\text{\color{red}\colorbox{black}{Representative memory-based CF}}$ techniques are $\text{\color{red}\colorbox{white}{nearest neighbor-based CF}}$ such as user-based CF and item-based CF [[Sarwar et al., 2001](https://d2l.ai/chapter_references/zreferences.html#id242 "Sarwar, B. M., Karypis, G., Konstan, J. A., Riedl, J., & others. (2001). Item-based collaborative filtering recommendation algorithms. Www, 1, 285–295.")]. Latent factor models such as $\text{\color{red}\colorbox{white}{matrix factorization}}$ are examples of $\text{\color{red}\colorbox{black}{model-based CF}}$.

$\text{\color{red}\colorbox{black}{Memory-based CF}}$ has limitations in dealing with sparse and large-scale data since it computes the similarity values based on common items. $\text{\color{red}\colorbox{black}{Model-based methods}}$ become more popular with its better capability in dealing with sparsity and scalability. Many model-based CF approaches can be extended with neural networks, leading to more flexible and scalable models with the computation acceleration in deep learning [[Zhang et al., 2019](https://d2l.ai/chapter_references/zreferences.html#id327 "Zhang, S., Yao, L., Sun, A., & Tay, Y. (2019). Deep learning based recommender system: a survey and new perspectives. ACM Computing Surveys (CSUR), 52(1), 5.")]. $\text{\color{yellow}\colorbox{black}{In general}}$, CF only uses the user-item interaction data to make predictions and recommendations. Besides CF, $\text{\color{red}\colorbox{black}{content-based}}$ and $\text{\color{red}\colorbox{black}{context-based}}$ **recommender systems** are also useful in incorporating the content descriptions of items/users and contextual signals such as timestamps and locations. Obviously, we may need to adjust the model types/structures when different input data is available.

## 1.2 Explicit Feedback and Implicit Feedback (显示、隐式)

To learn the preference of users, the system shall $\text{\color{red}\colorbox{white}{collect feedback}}$ from them. The feedback can be either explicit or implicit [[Hu et al., 2008](https://d2l.ai/chapter_references/zreferences.html#id118 "Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. 2008 Eighth IEEE International Conference on Data Mining (pp. 263–272).")].

- For example, [IMDb](https://www.imdb.com/) collects $\text{\color{red}\colorbox{white}{star ratings}}$ ranging from one to ten stars for movies. YouTube provides the $\text{\color{red}\colorbox{white}{thumbs-up and thumbs-down buttons}}$ for users to show their preferences. It is apparent that gathering $\text{\color{red}\colorbox{black}{explicit feedback}}$ requires users to indicate their interests proactively. $\text{\color{yellow}\colorbox{black}{Nonetheless}}$, explicit feedback is not always readily available as many users may be reluctant to rate products.
- Relatively speaking, $\text{\color{red}\colorbox{black}{implicit feedback}}$ is often readily available since it is mainly concerned with modeling implicit behavior such as user clicks. $\text{\color{yellow}\colorbox{black}{As such}}$, many recommender systems are centered on implicit feedback which indirectly reflects user’s opinion through $\text{\color{red}\colorbox{white}{observing user behavior}}$. There are diverse forms of implicit feedback including $\text{\color{red}\colorbox{white}{purchase history}}$, $\text{\color{red}\colorbox{white}{browsing history}}$, $\text{\color{red}\colorbox{white}{watches}}$ and even $\text{\color{red}\colorbox{white}{mouse movements}}$, a user that purchased many books by the same author probably likes that author. Note that implicit feedback is inherently noisy. We can only *guess* their preferences and true motives. A user watched a movie does not necessarily indicate a positive view of that movie.

## 1.3 Recommendation Tasks

A number of recommendation tasks have been investigated in the past decades. Based on the $\text{\color{red}\colorbox{white}{domain of applications}}$, there are

- movies recommendation,
- news recommendations,
- point-of-interest recommendation [[Ye et al., 2011](https://d2l.ai/chapter_references/zreferences.html#id318 "Ye, M., Yin, P., Lee, W.-C., & Lee, D.-L. (2011). Exploiting geographical influence for collaborative point-of-interest recommendation. Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval (pp. 325–334).")]
- and so forth.

It is also possible to differentiate the tasks based on the $\text{\color{red}\colorbox{white}{types of feedback and input data}}$, for example, the $\text{\color{red}\colorbox{black}{rating prediction task}}$ aims to predict the explicit ratings. Top-n recommendation (item ranking) ranks all items for each user personally based on the implicit feedback. If time-stamp (时间戳) information is also included, we can build $\text{\color{red}\colorbox{black}{sequence-aware recommendation}}$ [[Quadrana et al., 2018](https://d2l.ai/chapter_references/zreferences.html#id214 "Quadrana, M., Cremonesi, P., & Jannach, D. (2018). Sequence-aware recommender systems. ACM Computing Surveys (CSUR), 51(4), 66.")]. Another popular task is called $\text{\color{red}\colorbox{black}{click-through rate prediction}}$, which is also based on implicit feedback, but various categorical features can be utilized. Recommending for new users and recommending new items to existing users are called $\text{\color{red}\colorbox{black}{cold-start recommendation}}$ [[Schein et al., 2002](https://d2l.ai/chapter_references/zreferences.html#id243 "Schein, A. I., Popescul, A., Ungar, L. H., & Pennock, D. M. (2002). Methods and metrics for cold-start recommendations. Proceedings of the 25th annual international ACM SIGIR conference on Research and development in information retrieval (pp. 253–260).")].

## Summary

* Recommender systems are important for individual users and industries. Collaborative filtering is a key concept in recommendation.
* There are two types of feedbacks: implicit feedback and explicit feedback.  A number of recommendation tasks have been explored during the last decade.

## Exercises

1. Can you explain how recommender systems influence your daily life?
2. What interesting recommendation tasks do you think can be investigated?

[Discussions](https://discuss.d2l.ai/t/398)
