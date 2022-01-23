 # Machine Learning Demos

Notebooks and code snippets demonstrating various machine learning techniques:


6. [Comparison of REINFORCE vs Gumbel-Softmax vs MDNF gradients and convergence for a simplified objective](reinforce_vs_gumbelsoftmax_gradients.ipynb)
 * Optimization using REINFORCE vs reparametrization gradients (with GradientTape)
 * Gumbel-Softmax relaxation for discrete variables - an illustration of a bias
 * Mixture of Discrete Normalizing Flows relaxation for discrete variables
 
5. [Illustration of how entropy of the relaxed categorical distribution can be estimated and utilized for VI](entropy_of_relaxed_categorical_distribution.ipynb)
 * Comparison (and discussion of gradients) of three estimates of the entropy/KL-term in ELBO 

4. [Variational Autoencoder using Relaxed Categorical distribution](vae_relaxed_categorical.ipynb)
 * Sampling from Gumbel softmax with and without straight-through
 * Implementation of different approaches to estimation of KL divergence
 * Training with Mnist data
 * Reconstruction of digits and unconditional sampling latent codes

3. [A demonstration of Discrete Flows: Invertible Generative Models of Discrete Data](discrete_flows.ipynb)
 * Arithmetic on one-hot encoded vectors
 * Trainig simple discrete transformation
 * MLE-training of an autoregressive flow with masked autoencoder to match a target distribution.

2. [Probabilistc Matrix Factorization model with mean-field variational inference](probabilisitc_matrix_factorization_vi.ipynb).
 * Probabilistc Matrix Factorization implementation
 * estimating ELBO using MC
 * training using pyTorch automatic differentiation
 * simple evaluation of RMSE on test subset 

1. [Framing multi-output Bayesian optimization with GPyOpt](multi-task_bayesian_optimization_demo.ipynb)
 * fitting individual GPs 
 * fitting multi-task GPs using coregionalization
 * BO optimization of single function
 * BO optimization of 2-task problem
 * An implementation of a custom acquisition function
 * Extensive visualization of the optimization process
