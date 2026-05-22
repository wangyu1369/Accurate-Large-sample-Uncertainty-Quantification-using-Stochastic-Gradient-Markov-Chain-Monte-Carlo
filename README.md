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

The robust linear regression experiments evaluate uncertainty quantification under model misspecification. 

#### 2(a). Simulation

The simulation experiment uses synthetic misspecified data with outliers.
**Code location:**

```text
experiments/robust_linear_regression/simulation/
```

#### 2(b). Real-world Dataset

The real-world robust linear regression experiment evaluates uncertainty quantification on real-world Boston Housing datasets.

**Code location:**

```text
experiments/robust_linear_regression/real_data/
```
