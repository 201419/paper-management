## 《A Universal Algorithm for Variational Inequalities Adaptive to Smoothness and Noise》

https://arxiv.org/abs/1902.01637

### 1. Contributions：

- present a universal algorithm for variational inequalities based on the Mirror-Prox algorithm, for both deterministic anfd stochastic settings
- this is done in the general set-up of arbitrary norms and compatible Bregman divergences
- for convex minimization and convex-concave saddle-point problems, this leads to new adaptive algorithms ( new adaptive method can be seen as extension of AdaGrad )

### 2. Variational Inequalities and Gap Functions

#### Preliminaries.

Let ${\|\cdot\|}$ be a general norm and $\|\cdot\|_{*}$ be its **dual norm**, i.e. $\|z\|_{*}=\sup \left\{z^{T} x | \|x\| \leq 1\right\}$

A function $f : \mathcal{X} \mapsto \mathbb{R}$ is **$\mu$-strongly convex** over a convex set $\mathcal{K}$, if for any $x \in \mathcal{X}$ and any $\nabla f(x)$, a subgradient of $f$ at $x$, 
$$ f(y) \geq f(x)+\nabla f(x) \cdot(y-x)+\frac{\mu}{2}\|x-y\|^{2} ; \quad \forall x, y \in \mathcal{X} $$

A function $f : \mathcal{X} \mapsto \mathbb{R}$ is **$L$-smooth** over $\mathcal{K}$ if, 
$$ \|\nabla f(x)-\nabla f(y)\|_{*} \leq L\|x-y\| ; \quad \forall x, y \in \mathcal{X} $$

**Bregman divergence:**  for a convex differentiable function $f(\cdot)$, we define its Bregman divergence as follows, ( note that $\mathcal{D}_{f}(\cdot, \cdot)$ is always non-negative. )
$$ \mathcal{D}_{f}(x, y)=f(x)-f(y)-\nabla f(y) \cdot(x-y) $$

#### Gap function.

Considering a **monotone operator $F$** form $\mathcal{K}$ to $\mathbb{R}^{d}$, which is single-valued for simplicity.

Formally, a monotone operator satisfies,
$$ (x-y) \cdot(F(x)-F(y)) \geq 0 ; \quad \forall(x, y) \in \mathcal{X} \times \mathcal{X} $$

#### Convex-concave Zero-sum Games

Let $\phi : \mathcal{U} \times \mathcal{V} \mapsto \mathbb{R}$, where $\phi(u, v)$ is convex in $u$ and concave in $v$, and $\mathcal{U} \subseteq \mathbb{R}^{d_{1}}, \mathcal{V} \subseteq \mathbb{R}^{d_{2}}$ are compact convex sets.

The convex-concave zero-sum game induced by $\phi$ is defined as follows,
$$\min _{u \in \mathcal{U}} \max _{v \in \mathcal{V}} \phi(u, v)$$
The performance measure for such games is the duality gap which is defined as,
$$ \operatorname{DualGap}(u, v)=\max _{v \in \mathcal{V}} \phi(u, v)-\min _{u \in \mathcal{U}} \phi(u, v) $$
The duality gap is always non-negative, and we seek an (approximate) equilibrium, i.e., a point $\left(u^{*}, v^{*}\right)$ such that $\operatorname{DualGap}\left(u^{*}, v^{*}\right)=0$.

This setting can be classically described as a variational inequality problem. Let us denote,
$$ x :=(u, v) \in \mathcal{U} \times \mathcal{V}  \quad \text { and } \quad \mathcal{K} :=\mathcal{U} \times \mathcal{V} $$
For any $x=(u, v), x_{0}=\left(u_{0}, v_{0}\right) \in \mathcal{X}$, define a gap function and an operator $F : \mathcal{K} \mapsto \mathbb{R}^{d_{1}+d_{2}}$, as follows,
$$ \Delta\left(x, x_{0}\right) :=\phi\left(u, v_{0}\right)-\phi\left(u_{0}, v\right), \quad \text { and } \quad F(x) :=\left(\nabla_{u} \phi(u, v),-\nabla_{v} \phi(u, v)\right) $$
It is immediate to show that this gap function $\operatorname{Dual} \operatorname{Gap}(x) :=\max _{x_{0} \in \mathcal{X}} \Delta\left(x, x_{0}\right)$.

The next lemma from `Nemirovski (2004)`[^1] shows that $\Delta$ is a gap function compatible with $F$ (for completeness we provide its proof in paper *Appendix*).

**Lemma 2.1.** *The following applies for any* $x :=(u, v), x_{0} :=\left(u_{0}, v_{0}\right) \in \mathcal{U} \times \mathcal{V}$
$$ \Delta\left(x, x_{0}\right) :=\phi\left(u, v_{0}\right)-\phi\left(u_{0}, v\right) \leq F(x) \cdot\left(x-x_{0}\right) $$

### 3. Universal Mirror-Prox

`Rakhlin and Sridharan (2013)`[^2] suggest to apply the following learning rate scheme inside Optimistic OGD 
$$ \eta_{t}=D / \max \left\{\sqrt{\sum_{t=1}^{t-1}\left\|g_{t}-M_{t}\right\|^{2}}+\sqrt{\sum_{t=1}^{t-2}\left\|g_{t}-M_{t}\right\|^{2}}, 1\right\} $$

We suggest to use the following adaptive scheme
$$ \eta_{t}=D / \sqrt{G_{0}^{2}+\sum_{\tau=1}^{t-1} Z_{\tau}^{2}}, \quad \text { where } Z_{\tau}^{2} :=\frac{\left\|x_{\tau}-y_{\tau}\right\|^{2}+\left\|x_{\tau}-y_{\tau-1}\right\|^{2}}{5 \eta_{\pi}^{2}} $$

---

**Algorithm Universal Mirror-Prox**  
**Input:** #Iterations $T$, $y_{0}=\arg \min _{x \in \mathcal{X}} \mathcal{R}(x)$, learning rate $\{\eta_{t}\}_{t}$  
**for** $t=1 \ldots T$ **do**  
&emsp;&emsp; Set $M_{t}=F\left(y_{t-1}\right)$  
&emsp;&emsp; Updated:  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$ x_{t} \leftarrow \underset{x \in \mathcal{K}}{\arg \min } M_{t} \cdot x+\frac{1}{\eta_{t}} \mathcal{D}_{\mathcal{R}}\left(x, y_{t-1}\right), \quad \text { and define } g_{t} :=F\left(x_{t}\right) $  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$y_{t} \leftarrow \underset{x \in \mathcal{K}}{\arg \min } g_{t} \cdot x+\frac{1}{\eta_{t}} \mathcal{D}_{\mathcal{R}}\left(x, y_{t-1}\right) $  
**end for**  
**Output:** $\overline{x}_{T}=\frac{1}{T} \sum_{t=1}^{T} x_{t}$  
  


[^1]: Nemirovski, A. (2004). Prox-method with rate of convergence O(1/t) for variational inequalities with lipschitz continuous monotone operators and smooth convex-concave saddle point problems. SIAM Journal on Optimization, 15(1):229–251.
[^2]: Rakhlin, S. and Sridharan, K. (2013). Optimization, learning, and games with predictable sequences. In Advances in Neural Information Processing Systems, pages 3066–3074.