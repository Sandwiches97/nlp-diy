# 8.1. Sequence Models

- 内插法（在现有观测值之间进行估计）和外推法（对超出已知观测范围进行预测）在实践的难度上差别很大。因此，对于你所拥有的序列数据，在训练时始终要尊重其时间顺序，即最好不要基于未来的数据进行训练。
- 序列模型的估计需要专门的统计工具，两种较流行的选择是**自回归模型**和**隐变量自回归模型**。
- 对于时间是向前推进的因果模型，正向估计通常比反向估计更容易。
- 对于 直到时间 $t$ 的观测序列，其在时间步的预测输出是$\{x_{i} | i = t+1,...,t+k\}$，即“$k$ step 预测”。随着我们对预测时间值的增加，会造成误差的快速累积和预测质量的极速下降，k 越大越明显。
  - 注意分清测试、训练集的作用，它们二者是分开的。我们并不是用训练集去预测测试集，而是**训练集的数据**预测**训练集**，用**测试集的数据**去预测**测试集**；

Imagine that you are watching movies on Netflix. As a good Netflix user, you decide to rate each of the movies religiously. After all, a good movie is a good movie, and you want to watch more of them, right? As it turns out, things are not quite so simple. People’s opinions on movies can change quite significantly over time. In fact, psychologists even have names for some of the effects:

* There is  *anchoring* （锚定效应） , based on someone else’s opinion. For instance, after the Oscar awards, ratings for the corresponding movie go up, even though it is still the same movie. This effect persists for a few months until the award is forgotten. It has been shown that the effect lifts rating by over half a point [[Wu et al., 2017]](https://d2l.ai/chapter_references/zreferences.html#wu-ahmed-beutel-ea-2017).
* There is the  *hedonic adaptation*（享乐适应） , where humans quickly adapt to accept an improved or a worsened situation as the new normal. For instance, after watching many good movies, the expectations that the next movie is equally good or better are high. Hence, even an average movie might be considered as bad after many great ones are watched.
* There is  *seasonality* （季节性）. Very few viewers like to watch a Santa Claus movie in August.
* In some cases, movies become unpopular due to the misbehaviors of directors or actors in the production.
* Some movies become cult movies, because they were almost comically bad. *Plan 9 from Outer Space* and *Troll 2* achieved a high degree of notoriety for this reason.

In short, movie ratings are anything but stationary. Thus, using temporal dynamics led to more accurate movie recommendations [[Koren, 2009]](https://d2l.ai/chapter_references/zreferences.html#koren-2009). Of course, sequence data are not just about movie ratings. The following gives more illustrations.

* Many users have highly particular behavior when it comes to the time when they open apps. For instance, social media apps are much more popular after school with students. Stock market trading apps are more commonly used when the markets are open.
* It is much harder to predict tomorrow’s stock prices than to fill in the blanks for a stock price we missed yesterday, even though both are just a matter of estimating one number. After all, foresight is so much harder than hindsight. In statistics, the former (predicting beyond the known observations) is called *extrapolation* whereas the latter (estimating between the existing observations) is called  *interpolation* .
* Music, speech, text, and videos are all sequential in nature. If we were to permute them they would make little sense. The headline *dog bites man* is much less surprising than  *man bites dog* , even though the words are identical.
* Earthquakes are strongly correlated, i.e., after a massive earthquake there are very likely several smaller aftershocks, much more so than without the strong quake. In fact, earthquakes are spatiotemporally correlated, i.e., the aftershocks typically occur within a short time span and in close proximity.
* Humans interact with each other in a sequential nature, as can be seen in Twitter fights, dance patterns, and debates.

## 8.1.1. Statistical Tools

We need statistical tools and new deep neural network architectures to deal with sequence data. To keep things simple, we use the stock price (FTSE 100 index) illustrated in [Fig. 8.1.1](https://d2l.ai/chapter_recurrent-neural-networks/sequence.html#fig-ftse100) as an example.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://d2l.ai/_images/ftse100.png" width = "65%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig. 8.1.1 FTSE 100 index over about 30 years.
  	</div>
</center>

Let us denote the prices by $x_t$, i.e., at **time step** $t\in \mathbb{Z}^+$ we observe price $x_t$. Note that for sequences in this text, $t$ will typically be discrete and vary over integers or its subset. Suppose that a trader who wants to do well in the stock market on day $t$ predicts $x_t$ via

$$
x_t \sim P(x_t | x_{t-1}, ..., x_1). \tag{8.1.1}

$$

### 8.1.1.1. Autoregressive Models 自回归模型

In order to achieve this, our trader could use a **regression model** such as the one that we trained in [Section 3.3](https://d2l.ai/chapter_linear-networks/linear-regression-concise.html#sec-linear-concise). There is just one major problem: the number of inputs, $x_{t−1},…,x_1$ varies, depending on $t$. (与回归模型最主要的不同，就是这里的 $x$ 依赖时间 $t$)  That is to say, the number increases with the amount of data that we encounter, and we will need an approximation to make this computationally tractable. Much of what follows in this chapter will revolve around how to estimate $P(x_t∣x_{t−1},…,x_1)$ efficiently. In a nutshell (简而言之) it boils down to two strategies as follows.

- **First**, assume that the potentially rather long sequence $x_{t−1},…,x_1$ is not really necessary. In this case we might content ourselves with some timespan of length $τ$ and only use $x_{t−1},…,x_{t−τ}$ observations. The immediate benefit is that now **the number of arguments is always the same,**(固定窗口，参数总量不变) at least for $t>τ$. This allows us to train a deep network as indicated above. Such models will be called  **autoregressive models** , as they quite literally perform regression on themselves.
- The **second** strategy, shown in [Fig. 8.1.2](https://d2l.ai/chapter_recurrent-neural-networks/sequence.html#fig-sequence-model), is to keep some summary $h_t$ of the past observations, and at the same time update $h_t$ in addition to the prediction $x^t$. This leads to models that estimate $x_t$ with $x^t=P(x_t∣h_t)$ and moreover updates of the form $h_t=g(h_{t−1},x_{t−1})$. Since $h_t$ is never observed, these models are also called  **latent autoregressive models** .（隐变量自回归模型）

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://d2l.ai/_images/sequence-model.svg" width = "65%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
     Fig. 8.1.2 A latent autoregressive model.
  	</div>
</center>

Both cases raise the obvious question of how to generate training data. One typically uses historical observations (历史观测数据) to predict the next observation given the ones up to right now. Obviously we do not expect time to stand still （我们希望时间能一个一个动起来）. However, a common assumption is that while the specific values of $x_t$ might change, at least the dynamics of the sequence itself will not （动力学特性不变）. This is reasonable, since novel dynamics are just that, novel and thus not predictable using data that we have so far. Statisticians call dynamics that do not change  **stationary** 平稳性. Regardless of what we do, we will thus get an estimate of the entire sequence via

$$
P(x_1,…,x_T)=∏_{t=1}^TP(x_t∣x_{t−1},…,x_1).\tag{8.1.2}

$$

Note that the above considerations still hold if we deal with discrete objects, such as words, rather than continuous numbers. The only difference is that in such a situation we need to use a classifier rather than a regression model to estimate $P(x_t∣x_{t−1},…,x_1)$.

### 8.1.1.2. Markov Models

Recall the approximation that in an **autoregressive model** we use only $x_{t−1},…,x_{t−τ}$ instead of $x_{t−1},…,x_1$ to estimate $x_t$. Whenever this approximation is accurate we say that the sequence satisfies a  **Markov condition** . In particular, if $τ=1$, we have a **first-order Markov model** and $P(x)$ is given by

$$
P(x_1,…,x_T)=∏_{t=1}^TP(x_t∣x_{t−1}),\ \ \text{where}\ P(x_1∣x_0)=P(x_1). \tag{8.1.3}

$$

Such models are particularly nice whenever $x_t$ assumes only a discrete value, since in this case dynamic programming can be used to compute values **along the chain exactly** (沿着Markov chain计算即可). For instance, we can compute $P(x_{t+1}∣x_{t−1})$ （利用全概率公式将 $x_t$ 加进去）efficiently:

$$
\begin{aligned}
P(x_{t+1} \mid x_{t-1})
&= \frac{\sum_{x_t} P(x_{t+1}, x_t, x_{t-1})}{P(x_{t-1})}\\
&= \frac{\sum_{x_t} P(x_{t+1} \mid x_t, x_{t-1}) P(x_t, x_{t-1})}{P(x_{t-1})}\\
&= \sum_{x_t} P(x_{t+1} \mid x_t) P(x_t \mid x_{t-1})
\end{aligned}

$$

by using the fact that we only need to take into account a very short history of past observations: $P(x_{t+1} \mid x_t, x_{t-1}) = P(x_{t+1} \mid x_t)$. （齐次Markov性）
Going into details of dynamic programming is beyond the scope of this section. Control and reinforcement learning algorithms use such tools extensively.

### 8.1.1.3 Causality 因果性

In principle, there is nothing wrong with unfolding $P(x_1, \ldots, x_T)$ in reverse order (倒序展开). After all, by conditioning we can always write it via

$$
P(x_1, \ldots, x_T) = \prod_{t=T}^1 P(x_t \mid x_{t+1}, \ldots, x_T). \tag{8.1.5}

$$

In fact, if we have a Markov model, we can obtain a reverse conditional probability distribution, too. In many cases, **however**, there exists a natural direction for the data, namely **going forward in time**. It is clear that future events cannot influence the past. Hence, if we change $x_t$, we may be able to influence what happens for $x_{t+1}$ going forward **but not the converse**. That is, if we change $x_t$, **the distribution over past events will not change**（过去的分布不会改变）.  Consequently, it ought to be easier to explain $P(x_{t+1} \mid x_t)$ rather than $P(x_t \mid x_{t+1})$. For instance, it has been shown that in some cases we can find $x_{t+1} = f(x_t) + \epsilon$ for some additive noise $\epsilon$, whereas the converse is not true [[Hoyer et al., 2009]](https://d2l.ai/chapter_references/zreferences.html#hoyer-janzing-mooij-ea-2009). This is great news, since it is typically the forward direction that we are interested in estimating. The book by Peters et al. has explained more on this topic [[Peters et al., 2017a]](https://d2l.ai/chapter_references/zreferences.html#peters-janzing-scholkopf-2017). We are barely scratching the surface of it.

## Training

After reviewing so many statistical tools, let us try this out in practice. We begin by generating some data. To keep things simple we (**generate our sequence data by using a sine function with some additive noise for time steps $1, 2, \ldots, 1000$.**)

Next, we need to turn such a sequence into features and labels that our model can train on. Based on the embedding dimension $\tau$ we [**map the data into pairs $y_t = x_t$ and $\mathbf{x}_t = [x_{t-\tau}, \ldots, x_{t-1}]$.**] The astute （精明的） reader might have noticed that this gives us $\tau$ fewer data examples, since we do not have sufficient history for the first $\tau$ of them.

- A simple fix, in particular if the sequence is long, is to discard those few terms.
- Alternatively we could pad the sequence with zeros.

Here we only use the first 600 feature-label pairs for training. (`features` 的每次只取$\tau$ 个窗口)

Here we keep the architecture fairly simple: just an MLP with two fully-connected layers, ReLU activation, and squared loss. (注意这里我们线性层的输入为 $\tau = 4$ ）

Now we are ready to train the model. The code below is essentially identical to the training loop in previous sections, such as [Section 3.3](https://d2l.ai/chapter_linear-networks/linear-regression-concise.html#sec-linear-concise). Thus, we will not delve into much detail.

## 8.1.3. Prediction

Since the training loss is small, we would expect our model to work well. Let us see what this means in practice. The first thing to check is how well the model is able to predict what happens just in the next time step, namely the  **one-step-ahead prediction** .即，我们在预测后400个数据时，输入是

The one-step-ahead predictions look nice, just as we expected. Even beyond 604 (`n_train + tau`) observations the predictions still look trustworthy.

**However**, there is just one little problem to this: **if we observe sequence data only until time step 604** （也就是我们后面的数据，都要使用模型预测出来的结果去预测）, we cannot hope to receive the inputs for all the future one-step-ahead predictions. Instead, we need to work our way forward one step at a time:

$$
\hat{x}_{605} = f(x_{601}, x_{602}, x_{603}, x_{604}), \\
\hat{x}_{606} = f(x_{602}, x_{603}, x_{604}, \hat{x}_{605}), \\
\hat{x}_{607} = f(x_{603}, x_{604}, \hat{x}_{605}, \hat{x}_{606}),\\
\hat{x}_{608} = f(x_{604}, \hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}),\\
\hat{x}_{609} = f(\hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}, \hat{x}_{608}),\\
\ldots

$$

Generally, for an observed sequence up to $x_t$, its predicted output $\hat{x}_{t+k}$ at time step $t+k$ is called the $k$**-step-ahead prediction**. Since we have observed up to $x_{604}$, its $k$-step-ahead prediction is $\hat{x}_{604+k}$. In other words, we will have to [**use our own predictions to make multistep-ahead predictions**].
Let us see how well this goes.

As the above example shows, this is **a spectacular failure.** The predictions decay to a constant pretty quickly after a few prediction steps.
Why did the algorithm work so poorly?
This is ultimately due to the fact that the errors build up. Let us say that after step 1 we have some error $\epsilon_1 = \bar\epsilon$.
Now the *input* for step 2 is perturbed by $\epsilon_1$, hence we suffer some error in the order of $\epsilon_2 = \bar\epsilon + c \epsilon_1$ for some constant $c$, and so on. The error can diverge rather rapidly from the true observations. This is a common phenomenon. For instance, weather forecasts for the next 24 hours tend to be pretty accurate but beyond that the accuracy declines rapidly. We will discuss methods for improving this throughout this chapter and beyond. Let us [**take a closer look at the difficulties in $k$-step-ahead predictions**]
by computing predictions on the entire sequence for $k = 1, 4, 16, 64$.

This clearly illustrates how the quality of the prediction changes as we try to predict further into the future. While the 4-step-ahead predictions still look good, anything beyond that is almost useless.

## Summary

* There is quite a difference in difficulty between interpolation and extrapolation. Consequently, if you have a sequence, always respect the temporal order of the data when training, i.e., never train on future data.
* Sequence models require specialized statistical tools for estimation. Two popular choices are autoregressive models and latent-variable autoregressive models.
* For causal models (e.g., time going forward), estimating the forward direction is typically a lot easier than the reverse direction.
* For an observed sequence up to time step **𝑡**t, its predicted output at time step **𝑡**+**𝑘**t+k is the **𝑘**k *-step-ahead prediction* . As we predict further in time by increasing **𝑘**k, the errors accumulate and the quality of the prediction degrades, often dramatically.
