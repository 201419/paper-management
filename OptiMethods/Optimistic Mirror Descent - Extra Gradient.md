## 《Optimistic Mirror Descent in Saddle-Point Problems: Going the Extra (Gradient) Mile》

- https://arxiv.org/abs/1807.02629
- https://openreview.net/forum?id=Bkg8jjC9KQ

### 1. Introduction

As an example, the “strict saddle” property was shown to hold in a wide range of salient objective functions ranging from low-rank matrix factorization  and dictionary learning , to principal component analysis, and many other models.

A considerable corpus of work has been devoted to exploring and enhancing the stability of GANs, including techniques as diverse as the use of Wasserstein metrics, critic gradient penalties, feature matching, minibatch discrimination, etc.

" ... This is a critical failure of descent methods, but one which `Daskalakis et al. [2018]` showed can be overcome through “optimism”, interpreted in this context as an **“extra-gradient”** step that pushes the training process further along the incumbent gradient – as a result, **optimistic gradient descent (OGD)** succeeds in cases where vanilla **gradient descent (GD)** fails (specifically, unconstrained bilinear saddle-point problems). "

On the positive side, we show that if a problem is **strictly coherent** (a condition satisfied by all strictly convex-concave problems), **MD converges almost surely**, even in stochastic problems (Theorem 3.1).

考虑增加一个额外的梯度 ( extra-gradient ) 步骤，该步骤向前看并沿 “ 未来 ” 梯度迈出一步。这种技术最初由 Korpelevich [1976] 引入，随后作为 Nemirovski [2004] 的 Mirror-Prox 算法的基础获得了极大的普及。

The **extra-gradient technique** is often referred to as **Optimistic Mirror Descent (OMD)**.

We first show that **the last iterate of OMD** converges in all coherent problems, including null-coherent ones. We validate this prediction for **a wide array of standard GAN models** in Section 5.

### 2. Problem Setup and Preliminaries

#### Saddle-point problems.

Consider a saddle-point problem of the general form $$ \min _{x_{1} \in \mathcal{X}_{1}} \max _{x_{2} \in \mathcal{X}_{2}} f\left(x_{1}, x_{2}\right) \tag{SP} $$ where each feasible region $\mathcal{X}_{i}, i=1,2$ is a compact convex subset of a finite-dimensional normed sapce $\mathcal{V}_{i} \equiv \mathbb{R}^{d_{i}}$ and $f : \mathcal{X} \equiv \mathcal{X}_{1} \times \mathcal{X}_{2} \rightarrow \mathbb{R}$ denotes the problem's value function.

Since the individual gradients of $f$ will play a key role in our analysis, we will encode them in a single vector as $$ g(x)=\left(g_{1}(x), g_{2}(x)\right)=\left(\nabla_{x_{1}} f\left(x_{1}, x_{2}\right),-\nabla_{x_{2}} f\left(x_{1}, x_{2}\right)\right) $$

#### Variational inequalities and coherence.

Most of the literature on saddle-point problems has focused on the monotone case, i.e., when $f$ is convex-concave. In such problems, solutions of (SP) can be characterized equivalently as solutions of the **Stampacchia variational inequality** $$ \left\langle g\left(x^{*}\right), x-x^{*}\right\rangle \geq 0 \quad \text { for all } x \in \mathcal{X} \tag{SVI} $$ or, in **Minty** form $$ \left\langle g(x), x-x^{*}\right\rangle \geq 0 \quad \text { for all } x \in \mathcal{X} \tag{MVI} $$

(SP)、(SVI) 和 (MVI) 的解的等价性可以很好地扩展到 beyond the realm of monotone problems. For a concrete example, consider the problem $$ \min _{x_{1} \in[-1,1]} \max _{x_{2} \in[-1,1]}\left(x_{1}^{4} x_{2}^{2}+x_{1}^{2}+1\right)\left(x_{1}^{2} x_{2}^{4}-x_{2}^{2}+1\right) $$ $f$ 只有一个鞍点：$x^{*}=(0, 0)$，而且这个 $x^{*}$ 也是对应变分不等式 (VI) 的唯一解 （尽管 $f$ 甚至都不是  pseudo-monotone）.

Motivated by all this, we introduce below the following **notion of coherence**:
1. Every solution of (SVI) also solve (SP)
2. There exists a solution $p$ of (SP) that satisfies (MVI)
3. Every solution $x^{*}$ of (SP) satisfies (MVI) locally, i.e. for all $x$ sufficiently close to $x^{*}$

If (MVI) holds as a strict inequality whenever x is not a solution thereof, (SP) will be called **strictly coherent**;
If (MVI) holds as an equality for all x 2 X , we will say that (SP) is **null-coherent**.

Moreover, regarding the distinction between coherence and strict coherence, we show in `Appendix A` that **(SP) is strictly coherent when $f$ is strictly convex-concave**.

### 3. Mirror Descent

#### The methods.

The **basic idea** of mirror descent is to generate a **new state variable** $x^{+}$ from some starting state $x$ by taking a “mirror step” along a gradient-like vector $y$.  To do this, let $h : \mathcal{X} \rightarrow \mathbb{R}$ be a continuous function and $K$-strongly convex *distance-generating function* (DGF) on $\mathcal{X}$, i.e. $$ h\left(t x+(1-t) x^{\prime}\right) \leq t h(x)+(1-t) h\left(x^{\prime}\right)-\frac{1}{2} K t(1-t)\left\|x^{\prime}-x\right\|^{2} $$ for all $x, x^{\prime} \in \mathcal{X}$ and all $t \in[0,1]$.

Then, following `Bregman [1967]`, $h$ generates a pseudo-distance on $\mathcal{X}$ via the relation $$ D(p, x)=h(p)-h(x)-\langle\nabla h(x), p-x\rangle \quad \text { for all } p \in \mathcal{X}, x \in \operatorname{dom} \partial h $$ This pseudo-distance is known as the **Bregman divergence**. As we show in `Appendix B`, we have $D(p, x) \geq \frac{1}{2} K\|x-p\|^{2}$, so the convergence of a sequence $X_{n}$ to some target point $p$ can be verified by showing that $D\left(p, X_{n}\right) \rightarrow 0$. 注意到 Bregman divergence 不满足对称性和三角不等式，所以不是真正意义上的距离函数. Moreover, the convergence of $X_{n}$ to $p$ does not necessarily imply that $D\left(p, X_{n}\right) \rightarrow 0$. **Bregman reciprocity**, it will be convenient to assume that such phenomena do not occur, i.e. that $D\left(p, X_{n}\right) \rightarrow 0$ whenever $X_{n} \rightarrow p$.

As with standard Euclidean distances, the Bregman divergence generates an associated proxmapping defined as $$ P_{x}(y)=\underset{x^{\prime} \in \mathcal{X}}{\arg \min }\left\{\left\langle y, x-x^{\prime}\right\rangle+ D\left(x^{\prime}, x\right)\right\} \quad \text { for all } x \in \operatorname{dom} \partial h, y \in \mathcal{Y} $$