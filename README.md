<img src="images/UW_main_compact_white.png" alt="Description" width="600">

# Maximizing Sparsity in NN with MIP

**Authors:** Changwon Lee, Jeff Linderoth  
**Affiliation:** University of Wisconsin–Madison  
**Date:** August 2025\
**Motivated by:** Rob Nowak, Fischetti and Jo  

---

Sparsifying a Neural Network(NN) model has been a big interest in many areas.

Works like "[Global Minimizers of ℓᵖ-Regularized Objectives Yield the Sparsest ReLU Neural Networks](https://arxiv.org/pdf/2505.21791)" have proven some interesting properties about the sparsity of a single-hidden-layer NN.

In this paper, we want to leverage the MIP formulation of a Neural Network model with nonlinear activation function and try to find the true sparsest NN weights.
(We will refer to the MIP formulation of a NN presented by "[Fischetti&Jo](https://link.springer.com/content/pdf/10.1007/s10601-018-9285-6.pdf)")

In our problem, we aim to interpolate our dataset exactly to a single-hidden-layer ReLU NN.

We first start with the vanilla setting of the NN and proceed to generalize the architecture of the NN.

We can generalize this setting in several ways:
```
multiple-hidden-layers, different activation function, multivariate output
```

## Problem Setting

* Given **N datasets**: $(x^1, y^1), (x^2, y^2), \dots, (x^N, y^N) \in R^d \times R$
* Find weight/bias parameters: $
* $x^n \rightarrow W^T x^n + b$

4. Each sample passes through:
   \[
   \mathbf{x}^n \rightarrow \text{Affine Combination } (\mathbf{w}^k \overline{\mathbf{x}^n}) 
   \rightarrow \text{Nonlinear Operator (ReLU)} 
   \rightarrow \text{Neuron Activation } h^k
   \]

---

### Sparsity Measure

How to measure the sparsity of a NN is an important issue.

At this point, we define sparsity as the number of nonzero 'path' of the NN, i.e., and we set this as the objective function of our optimization problem.

### Feasibility Check

There are two ways to determine the width $K$ of the single-hidden-layer.


## 🧮 Neural Network Formulation

![Neural Network Diagram](figs/nn_diagram.png)

Each hidden neuron computes:
\[
h^1_i = \max\{0, \mathbf{w}_i \cdot \mathbf{x}^n + b^1_i\} 
      = \max\{0, \mathbf{w}_i \cdot \overline{\mathbf{x}^n}\}
\]

---

## 🔢 MIP Formulation 1

**Decision Variables**

\[
\begin{aligned}
\mathbf{W} &\in \mathbb{R}^{l_1 \times d}, \quad 
  w_{ij} \in \mathbb{R} \\
\mathbf{b} &\in \mathbb{R}^{l_1}, \quad 
  b_i \in \mathbb{R} \\
p^n_i, q^n_i &\in \mathbb{R} \\
z^n_i &\in \{0,1\}, \quad (n=1,\dots,N;\; i=1,\dots,l_1) \\
\mathbf{v} &\in \mathbb{R}^{l_1}, \quad 
  v_i \in \mathbb{R} \\
s_{ij} &\in \mathbb{R}, \quad 
t_{ij} \in \{0,1\}
\end{aligned}
\]

---

**Objective:**
\[
\min \sum_{i=1}^{l_1}\sum_{j=1}^{d} t_{ij}
\]

**Constraints:**
\[
\begin{aligned}
& \sum_{j=1}^{d} w_{ij} x^n_j + b_i = p^n_i - q^n_i, && \forall n,i \\
& p^n_i, q^n_i \ge 0, && \forall n,i \\
& p^n_i \le M(1 - z^n_i), && \forall n,i \\
& q^n_i \le M z^n_i, && \forall n,i \\
& \sum_{i=1}^{l_1} p^n_i v_i = y^n, && \forall n \\
& s_{ij} = w_{ij} v_i, && \forall i,j \\
& -M t_{ij} \le s_{ij} \le M t_{ij}, && \forall i,j
\end{aligned}
\]

---

This sentence uses `$` delimiters to show math inline: $\sqrt{3x-1}+(1+x)^2$

This sentence uses $\` and \`$ delimiters to show math inline: $`\sqrt{3x-1}+(1+x)^2`$

$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$

### Problem Description

We solve the following LP:

$$
\begin{aligned}
\text{maximize} \quad & 5x + 4y \\
\text{subject to}\quad
& 6x + 4y \le 24 \\
& x + 2y \le 6 \\
& x, y \ge 0
\end{aligned}
$$

where:
- $x$ = units of product A  
- $y$ = units of product B

## ⚙️ General MIP Formulation

Same structure as above, generalized for arbitrary layer width \( l_1 \) and input dimension \( d \).

---

## Current Step we are taking now: Eliminate Symmetry to achieve faster run time


## 📚 References

- Fischetti, M., & Jo, J. (2018). *Deep Neural Networks and Mixed Integer Linear Optimization*.  
- Nowak, R. D. (2020). *Optimization Perspectives on Learning*.  
- UW–Madison Optimization Group Research Notes.

---

> _“Optimization provides structure; learning gives adaptability.”_  
> — Research Archive (UW–Madison)
