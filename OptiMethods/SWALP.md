## 《SWALP : Stochastic Weight Averaging in Low-Precision Training》

https://arxiv.org/abs/1904.11943

### 1. Contributions

- propose a principled approach to using **stochastic weight averaging** in **low-precision training** (SWALP) where all numbers including the gradient accumulators are quantized.
- prove that SWALP can reach the optimal solution **for quadratic objectives** with no loss of accuracy from the quantization
- prove that SWALP converges to a noise ball that is asymptotically smaller than that of low-precision SGD **for strongly convex objectives**
- method can significantly **reduce the performance gap between low-precision and fullprecision training** ( *8-bit SWALP can match the full-precision SGD baseline on `CIFAR-10` and `CIFAR-100` with both `VGG-16` and `PreResNet-164`* )
- code at https://github.com/stevenygd/SWALP

### 2. Related Works

Inspired by the geometry of the loss function traversed by SGD with a modified learning rate schedule, `Izmailov et al. (2018a)`[^1] proposed Stochastic Weight Averaging (SWA), which performs an equally weighted average of SGD iterates with cyclical or high constant learning rates. 

Released code. URL https://github.com/timgaripov/swa.

`Izmailov et al. (2018a)` develop SWA for deep learning, showing improved generalization. 

While our work is inspired by `Izmailov et al. (2018a)`, we focus on developing SWA for low-precision training.

### 3. Methods

#### Quantization

In order to use low-precision numbers during training, we define a **quantization function** $Q$, which rounds a real number to be stored in fewer bits. 

In this paper, we use **fixed point quantization** with **stochastic rounding** to demonstrate the algorithm and analyze its convergence properties.

We will use **block floating point** `Song et al., 2017`[^2] in our deep learning experiments.

#### Fixed Point Quantization.

In stochastic rounding, numbers are rounded up or down at random such that $\mathbb{E}[Q(w)]=w$ for all $w$ that will not cause overflow. 

Explicitly, suppose we allocate $W$ bits to represent the quantized number and allocate $F$ of the $W$ bits to represent the fractional part (小数部分) of the number. The quantization gap (相邻两个数之间的差值) $\delta=2^{-F}$ represents the distance between successive representable fixed-point numbers. The **upper limit of the representable numbers** is $u=2^{W-F-1}-2^{-F}$ and the **lower limit** is $l=-2^{W-F-1}$.

We write the quantization function as $Q_{\delta} : \mathbb{R} \rightarrow \mathbb{R}$ such that
$$ Q_{\delta}(w)=\left\{\begin{array}{ll}{\operatorname{clip}\left(\delta\left\lfloor\frac{w}{\delta}\right\rfloor, l, u\right)} & {\text { w.p. (with probability) }\left\lceil\frac{w}{\delta}\right\rceil-\frac{w}{\delta}} \\ {\operatorname{clip}\left(\delta\left\lceil\frac{w}{\delta}\right\rceil, l, u\right)} & {\text { w.p. (with probability) } 1-\left(\left\lceil\frac{w}{\delta}\right\rceil-\frac{w}{\delta}\right)}\end{array}\right. \tag{1} $$
where $\operatorname{clip}(x, l, u)=\max (\min (x, u), l)$.

#### Block Floating Point (BFP) Quantization.

Floating-point numbers have individual exponents, and fixed-point numbers all share the same fixed exponent. For block floatingpoint numbers, all numbers within a block share the same exponent, which is allowed to vary like a floating-point exponent.

Suppose we allocate W bits to represent each number in the block and F bits to represent the shared exponent. The shared exponent $E(\mathbf{w})$ for a block of numbers $\mathbf{w}$ is usually set to be the largest exponent in a block to avoid overflow.

In our experiments, we simulated block floating point numbers by using the following formula to compute the shared exponent:
$$ E(\mathbf{w})=\operatorname{clip}\left(\left\lfloor\log _{2}\left(\max _{i}\left|\mathbf{w}_{i}\right|\right)\right\rfloor,- 2^{F-1}, 2^{F-1}-1\right) $$
We then apply equation (1) with $\delta$ replaced by $2^{-E(\mathbf{w})+W-2}$ to quantize all numbers in $w$.

For deep learning experiments, **BFP is preferred over Fixed-Point** because BFP usually has less quantization error caused by overflow and underflow when quantizing DNN models. We will discuss how to design appropriate blocks in **Experiments**, and show that choosing appropriate block design can result in better performance.

---

**Algorithm 1 —— Stochastic Weight Averaging with Low Precision.**  
**Require:** Initial after-warm-up weight $w_{0}$; learning rate $\alpha$ ;  
&emsp;&emsp;&emsp;&emsp;total number of iterations $T$; cycle length $c$ ;  
&emsp;&emsp;&emsp;&emsp;random gradient samples $\nabla \tilde{f}\left(w_{t}\right)$; quantization function $Q$;  
&emsp;&emsp;$\overline{w}_{0} \leftarrow w_{0}$ { Accumulator for SWA (high precision) }  
&emsp;&emsp;$m \leftarrow 1$ { Number of models that have been averaged }  
&emsp;&emsp;**for** $t=1,2, \ldots, T$ **do**  
&emsp;&emsp;&emsp;&emsp;$w_{t} \leftarrow Q\left(w_{t-1}-\alpha \nabla \tilde{f}_{t}\left(w_{t-1}\right)\right)$ { Training with weight quantization;  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$w_{t}$ is stored in low precision }  
&emsp;&emsp;&emsp;&emsp;**if** $t \equiv 0(\bmod c)$ **then**  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$\overline{w}_{m} \leftarrow\left(\overline{w}_{m-1} \cdot m+w_{t}\right) /(m+1)$ { Update model with weight averaging  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;in high precision }  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$m \leftarrow m+1$ { Increment model count }  
&emsp;&emsp;&emsp;&emsp;**end if**  
&emsp;&emsp;**end for**  
&emsp;&emsp;**return** $\overline{w}$  

---

**Algorithm 2 —— SWALP with all numbers quantized during training.**  
**Require:** $L$ layers DNN $\left\{f_{1}, \dots, f_{L}\right\}$; Scheduled learning rate $\alpha_{t}$ ;  
&emsp;&emsp;&emsp;&emsp;Momentum $\rho$; Initial weights $w_{0}^{(i)}, \forall l \in[1, L]$ ;  
&emsp;&emsp;&emsp;&emsp;Total iterations $T$; Warm-up iterations $S$; Cycle length $c$ ;  
&emsp;&emsp;&emsp;&emsp;Quantization functions $Q_{W}, Q_{A}, Q_{G}, Q_{E},$ and $Q_{M}$ ;  
&emsp;&emsp;&emsp;&emsp;Loss function $\mathcal{L}$; Data batch sequence $\left\{\left(x_{i}, y_{i}\right)\right\}_{i=1}^{T}$ ;  
&emsp;&emsp;$\overline{w}_{0}^{(l)} \leftarrow 0, \forall l \in[1, L]$  
&emsp;&emsp;$m \leftarrow 0$  
&emsp;&emsp;**for** $t=1,2, \ldots, T$ **do**  
&emsp;&emsp;&emsp;&emsp;**1. Forward Propagation:**  
&emsp;&emsp;&emsp;&emsp;$a_{t}^{(0)}=x_{i}$  
&emsp;&emsp;&emsp;&emsp;$a_{t}^{(l)}=Q_{A}\left(f_{l}\left(a_{t}^{(l-1)}, w_{t}^{(l)}\right)\right), \forall l \in[1, L]$  
&emsp;&emsp;&emsp;&emsp;**2. Backward Propagation:**  
&emsp;&emsp;&emsp;&emsp;$e_{t}^{(L)}=\nabla_{a_{t}^{(L)}} \mathcal{L}\left(a_{t}^{(L)}, y_{t}\right)$  
&emsp;&emsp;&emsp;&emsp;$e_{t}^{(l-1)}=Q_{E}\left(\frac{\partial f_{l}\left(a_{t}^{(l)}\right)}{\partial a_{t}^{(l-1)}} e_{t}^{(l)}\right), \forall l \in[1, L]$  
&emsp;&emsp;&emsp;&emsp;$g_{t}^{(l)}=Q_{G}\left(\frac{\partial f_{l}}{\partial w_{t}^{(l)}} e_{t}^{(l)}\right), \forall l \in[1, L]$  
&emsp;&emsp;&emsp;&emsp;**3. Low Precision SGD Update (with momentum):**  
&emsp;&emsp;&emsp;&emsp;$v_{t}^{(l)} \leftarrow \rho Q_{M}\left(v_{t-1}^{(l)}\right)+g_{t}^{(l)}, \forall l \in[1, L]$  
&emsp;&emsp;&emsp;&emsp;$w_{t}^{(l)} \leftarrow Q_{W}\left(w_{t-1}-\alpha_{t} v_{t}^{(l)}\right), \forall l \in[1, L]$  
&emsp;&emsp;&emsp;&emsp;**4. High Precision SWA Update:**  
&emsp;&emsp;&emsp;&emsp;**if** $t>S$ and $(t-S) \equiv 0(\bmod c)$ **then**  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$\overline{w}_{m}^{(l)} \leftarrow\left(\overline{w}_{m-1}^{(l)} \cdot m+w_{t}^{(l)}\right) /(m+1), \forall l \in[1, L]$  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$m \leftarrow m+1$  
&emsp;&emsp;&emsp;&emsp;**end if**  
&emsp;&emsp;**end for**  
&emsp;&emsp;**return** $\overline{w}$  


[^1]: Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., and Wilson, A. G. Averaging weights leads to wider optima and better generalization. 2018a.
[^2]: Song, Z., Liu, Z., Wang, C., and Wang, D. Computation error analysis of block floating point arithmetic oriented convolution neural network accelerator design. arXiv preprint arXiv:1709.07776, 2017.