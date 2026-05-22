**Accurate Large-sample Uncertainty Quantification using Stochastic Gradient Markov Chain Monte Carlo**
## Experiments

This repository contains code for reproducing the main experiments in the paper. The experiments are organized as follows.

### 1. Motivating Example

The motivating example (Figure 1) illustrates the limitation of continuous-time/SDE-based approximations for SGD/SGLD with large batch size (or learning rate).

**Code location:**

```text
experiments/motivating_example/
```

### 2. Robust Linear Regression

The robust linear regression experiments evaluate uncertainty quantification under model misspecification, where the target covariance is the sandwich covariance rather than the inverse Hessian/posterior covariance. We compare the stationary covariance predicted by different SGD/SGLD theories with the empirical covariance estimated from SGD iterates.

#### 2(a). Simulation

The simulation experiment uses synthetic data with a controlled data-generating process. This allows us to evaluate covariance approximation accuracy, parameter estimation error, and calibration under model misspecification.

**Code location:**

```text
experiments/robust_linear_regression/simulation/
