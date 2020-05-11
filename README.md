# Machine Learning Demos

Notebooks and code snippets demonstrating various machine learning techniques:
1. [Framing multi-output Bayesian optimization with GPyOpt](multi-task_bayesian_optimization_demo.ipynb)
 * fitting individual GPs 
 * fitting multi-task GPs using coregionalization
 * BO optimization of single function
 * BO optimization of 2-task problem
 * An implementation of a custom acquisition function
 * Extensive visualization of the optimization process


2. [Probabilistc Matrix Factorization model with mean-field variational inference](probabilisitc_matrix_factorization_vi.ipynb).
 * Probabilistc Matrix Factorization implementation
 * estimating ELBO using MC
 * training using pyTorch automatic differentiation
 * simple evaluation of RMSE on test subset 


3. [A demonstration of Discrete Flows: Invertible Generative Models of Discrete Data](discrete_flows.ipynb)
 * Arithmetic on one-hot encoded vectors
 * Trainig simple discrete transformation
 * MLE-training of an autoregressive flow with masked autoencoder to match a target distribution.

