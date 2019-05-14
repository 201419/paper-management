## 稀疏优化

以下来自胡耀华[1]于2018年发表的文章。

最优化理论和方法是应用数学的一个重要分支，为诸多科学和应用问题提供了统一的建模框架和研究方法，相关研究成果在工农业生产、经济管理、金融投资、交通运输、通信控制、图像处理、生命科学等领域已获得广泛应用。

稀疏优化模型是最优化领域中非常热门的研究课题，它旨在寻找一个欠定线性系统的稀疏解，即只有极少数的分量不为零。

稀疏优化模型最早是由美国科学院院士David Donoho等人[2]于1998年提出来的，它的本质思想是结合解的稀疏性结构来构建数学模型，克服欠定线性反问题的不适定性，进而提升模型的稳定性和准确性。

特别地，在2005年，数学家Emmanuel Candes与陶哲轩[3]给出了稀疏优化模型/压缩感知的数学理论，证明了在已知信号的稀疏性的情况下，稀疏优化模型能够利用极少数的采样数(显著优于奈奎斯特采样定理)来重建原信号。此文奠定了稀疏优化模型的理论根基。

在过去的十年中，稀疏优化模型吸引了学术界与业界的大量关注，并且在很多领域都取得了成功的应用，诸如压缩感知[2, 4]、图像科学[5, 6]、机器学习[7, 8]、统计建模[9, 10]、基因组学数据分析[11, 12]。

### 【参考文献】

[1]. 刘思凡, 王浩, 胡耀华. 稀疏优化模型及其正则化方法. Preprint \
[2]. S. S. Chen, D. L. Donoho, and M. A. Saunders. Atomic decomposition by basis pursuit. SIAM Review, 43:129–159, 2001 \
[3]. E. J. Candes and T. Tao. Decoding by linear programming. IEEE transactions on Information Theory, 51(12):4203–4215, 2005 \
[4]. D. L. Donoho. Compressed sensing. IEEE Transactions on Information Theory, 52(8):1289–1306, 2006 \
[5]. A. Beck and M. Teboulle. A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM Journal on Imaging Sciences, 2(1):183–202, 2009 \
[6]. M. Elad. Sparse and Redundant Representations. Springer, New York, 2010 \
[7]. F. Bach, R. Jenatton, J. Mairal, and G. Obozinski. Structured sparsity through convex optimization. Statistical Science, 27(4):450–468, 2012 \
[8]. Y. Hu, C. Li, K. Meng, J. Qin, and X. Yang. Group sparse optimization via ℓp,q regularization. Journal of Machine Learning Research, 18(30):1–52, 2017 \
[9]. J. Fan and R. Li. Variable selection via nonconcave penalized likelihood and its oracle properties. Journal of the American Statistical Association, 96(456):1348–1360, 2001 \
[10]. R. Tibshirani. Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society, series B, 58(1):267–288, 1996 \
[11]. J. Qin, Y. Hu, F. Xu, H. K. Yalamanchili, and J. Wang. Inferring gene regulatory networks by integrating ChIP-seq/chip and transcriptome data via LASSO-type regularization methods. Methods, 67(3):294–303, 2014 \
[12]. J. Wang, Y. Hu, C. Li, and J.-C. Yao. Linear convergence of CQ algorithms and applications in gene regulatory network inference. Inverse Problems, 33(5):055017, 2017