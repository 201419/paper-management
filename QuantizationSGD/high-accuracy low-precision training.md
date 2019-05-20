## HALP: High-Accuracy Low-Precision Training

- https://dawn.cs.stanford.edu//2018/03/09/low-precision/
- https://arxiv.org/abs/1803.03383

这篇文章描述了可以通过一种称为 bit recentering 的技术使用 low-precision computation 获得 high-accuracy solutions 的案例.

Low-precision computation 受到了很大的关注。一些公司甚至开始 developing new hardware architectures，这些架构本身支持并且能加速低精度操作，包括微软的 [Project Brainwave](https://www.microsoft.com/en-us/research/blog/microsoft-unveils-project-brainwave/) 和 Google 的 [TPU](https://en.wikipedia.org/wiki/Tensor_processing_unit). 使用低精度有很多好处，但 low-precision 方法主要用于 inference - not for training.

在这之前，低精度训练算法在使用时需要权衡它的优缺点：**when calculations use fewer bits, more round-off error is added, which limits training accuracy**.

Is it possible to design algorithms that use low precision without it limiting their accuracy?

Here we'll describe a new variant of stochastic gradient descent (SGD) called **high-accuracy low precision (HALP)** that can do it. 

HALP can do better than previous algorithms because it **reduces the two sources of noise** that limit the accuracy of low-precision SGD: **gradient variance and round-off error**.

- HALP 使用称为 [SVRG](https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction) 的技术来减少来自 gradient variance 的噪声. SVRG 定期使用全梯度来减少SGD中使用的梯度样本的方差.
- 为了减少量化带来的噪声，HALP 使用了一种称之为 bit centering 的新技术.直觉上来理解，当接近最优点时，梯度的量级变小，在某种意义上承载的信息更少，因此应该能够压缩它.通过动态地 re-centering and re-scaling our low-precision numbers，我们可以lower the quantization noise as the algorithm converges.

**HALP is provably able to produce arbitrarily accurate solutions at the same linear convergence rate as full-precision SVRG, while using low-precision iterates with a fixed number of bits.**

### Why was low-precision SGD limited?

we want to solve training problems of the form $$ \operatorname{maximize} f(w)=\frac{1}{N} \sum_{i=1}^{N} f_{i}(w) \text { over } w \in \mathbb{R}^{d} $$ solving this is with stochastic gradient descent, which is an iterative algorithm that approaches the optimum by running $$ w_{t+1}=w_{t}-\alpha \nabla f_{i_{t}}\left(w_{t}\right) $$ where $i_{t}$ is an index randomly chosen from $\{1, \dots, N\}$ at each iteration.

直接用 fixed-point 去操作会有表示上的问题，例如算法的最优解可能不能用 fixed-point 去表示. 举个例子，如果使用 8-bit fixed-point representation，那只能存储从 -128 到 +127 之间的数，如果最优解是 100.5，那么算法所能找到最好的解和真实解之间至少相差 0.5

除此之外，将梯度转换为定点所产生的舍入误差会降低收敛速度。这些效应共同限制了 the accuracy of low-precision SGD.

### Bit Centering

When we are running SGD, in some sense what we are actually doing is averaging (or summing up) a bunch of gradient samples. The key idea behind bit centering is **as the gradients become smaller, we can average them with less error using the same number of bits**.

举个例子，对于 [-100, 100] 和 [-1, 1]，对于同样 bit 的定点数，表达后者的精度会更高。

This insight suggests that we **should dynamically update the low-precision representation**: as the gradients get smaller, we should use fixed-point numbers that have a smaller delta and cover a smaller range.

But how do we know how to update our representation? What range do we need to cover?

如果目标函数是 [$\mu$-strongly convex](https://en.wikipedia.org/wiki/Convex_function#Strongly_convex_functions) 的，then whenever we take a full gradient at some point $w$, we can bound the location of the optimum with $$ \left\|w-w^{*}\right\| \leq \frac{1}{\mu}\|\nabla f(w)\| $$

This inequality gives us a range of values in which the solution can be located, and so **whenever we compute a full gradient, we can re-center and re-scale the low-precision representation** to cover this range.

We call this operation **bit centering**.

> Note that even if our objective is not strongly convex, we can still perform bit-centering: now the parameter $μ$ becomes a hyperparameter of the algorithm.

### HALP

HALP is our algorithm which runs SVRG and uses bit centering with a full gradient at every epoch to update the low-precision representation. The full details and algorithm statement are in the [paper](https://arxiv.org/abs/1803.03383).