# Hyper-parameter tuning of neural network for high-dimensional problems in the case of Helmholtz equation

In this work we study the effectiveness of common hyper-parameter optimization (HPO) methods for physics-informed neural network (PINN) with application to multidimensional Helmholtz problem. The network was built upon PyTorch framework without the use of special PINN-oriented libraries. We investigate the effect of hyper-parameters on NN model performance and conduct automatic hyper-parameter optimization using different combinations of search algorithms and trial schedulers.

We chose an open-source HPO framework Ray Tune that provides unified interface for many HPO packages as the HPO tool in our work. We consider two search algorithms: random search and Bayesian method based on tree-structured Parzen estimator (TPE), in implementations hyperopt and hpbandster, and the Asynchronous Successive Halving (ASHA) early-stopping algorithm. For our problem, enabling early-stopping algorithm is shown to achieve faster HPO convergence speed than switching from random search to Bayesian method.

Results of this study were presented at DLCP-2023 (https://theory.sinp.msu.ru/doku.php/dlcp2023/start).
