# Maximizing Sparsity in NN with MIP

**Authors:** Changwon Lee, Jeff Linderoth  
**Affiliation:** University of Wisconsin–Madison  
**Date:** August 2025\
**Motivated by:** Robert Nowak, Fischetti and Jo  

---

This sentence uses `$` delimiters to show math inline: $\sqrt{3x-1}+(1+x)^2$

This sentence uses $\` and \`$ delimiters to show math inline: $`\sqrt{3x-1}+(1+x)^2`$

$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$

## Problem Setting

1. Given **N datasets**:
   $\mathbf{x}$
   $
   (\mathbf{x}^1, y^1), (\mathbf{x}^2, y^2), \dots, (\mathbf{x}^N, y^N)
   $

3. **Input:**  
   Each input vector is *d*-dimensional  
   \[
   \mathbf{x}^n = (x^n_1, x^n_2, \ldots, x^n_d) \in \mathbb{R}^d, \quad \forall n \in [N]
   \]  
   Augmented input:  
   \[
   \overline{\mathbf{x}^n} = (x^n_1, x^n_2, \ldots, x^n_d, 1) \in \mathbb{R}^{d+1}
   \]

4. Each sample passes through:
   \[
   \mathbf{x}^n \rightarrow \text{Affine Combination } (\mathbf{w}^k \overline{\mathbf{x}^n}) 
   \rightarrow \text{Nonlinear Operator (ReLU)} 
   \rightarrow \text{Neuron Activation } h^k
   \]

---

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

## ⚙️ General MIP Formulation

Same structure as above, generalized for arbitrary layer width \( l_1 \) and input dimension \( d \).

---

## 📚 References

- Fischetti, M., & Jo, J. (2018). *Deep Neural Networks and Mixed Integer Linear Optimization*.  
- Nowak, R. D. (2020). *Optimization Perspectives on Learning*.  
- UW–Madison Optimization Group Research Notes.

---

> _“Optimization provides structure; learning gives adaptability.”_  
> — Research Archive (UW–Madison)
