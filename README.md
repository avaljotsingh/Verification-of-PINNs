# Formal Verification and Certified Training of PINNs

**Formal Verification:** This incorporates finding the worst-case residual error of a PINN across the spatio-temporal domain. This can be done by computing the upper bound of the residual error which can be done by analyzing the weights and computing the bounds of the output and its gradient with respect to the inputs. For this, we will explore the existing Neural Network Verification techniques which include complete methods like MILP/SMT encodings and incomplete but scalable methods like Abstract Interpretation and Linear Bound Propagation methods [1, 2].

**Certified Training:** Some of the above techniques can also be leveraged to design appropriate loss function components that will guide the network to learn parameters that achieves smaller formally proven bounds with the same verification algorithm. [3]

## References

1. [Provably Correct Physics-Informed Neural Networks, Eiras et. al.](https://arxiv.org/pdf/2305.10157.pdf)
2. [Beta-CROWN: Efficient Bound Propagation with
Per-neuron Split Constraints for Neural Network
Robustness Verification](https://proceedings.neurips.cc/paper_files/paper/2021/file/fac7fead96dafceaf80c1daffeae82a4-Paper.pdf)
3. [Differentiable Abstract Interpretation for Provably Robust Neural Networks](https://proceedings.mlr.press/v80/mirman18b/mirman18b.pdf)