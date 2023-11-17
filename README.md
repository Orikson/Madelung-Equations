# Madelung Equations
Computational software for solving the Madelung Equations. A project made for APM 598/MAT 494, Mathematics of Quantum Mechanics, at ASU

Authors: Eron Ristich, and Samarth Dev

## The Madelung Equations
The Madelung Equations are a set of equations proposed by Erwin Madelung in 1926 to describe a more physical interpretation of the Schr&ouml;dinger Equation. Notably, Madelung's analysis resulted in a set of equations that are similar in nature to those of classical hydrodynamics, and are often referred to as the "hydrodynamic" interpretation of the Schr&ouml;dinger Equation. The Madelung Equations are as follows:

$$
\begin{align}
&\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \vec{v}) = 0 \\
&\frac{\partial \vec{v}}{\partial t} + (\vec{v} \cdot \nabla)\vec{v} = -\frac{1}{m}\nabla (Q + V)
\end{align}
$$

where $\rho$ is the mass density, $\vec{v}$ is the velocity field, $Q$ is the Bohm quantum potential, $V$ is the potential energy, and $\psi$ is the wave function.
$$\rho = m|\psi|^2$$
$$Q = -\frac{\hbar^2}{2m}\frac{\nabla^2 \sqrt{\rho}}{\sqrt{\rho}}$$

Notably, these equations take the form of the continuity equation and the Euler equation, respectively. The Madelung Equations are a set of nonlinear partial differential equations, and are difficult to solve analytically. However, they can be solved numerically using a variety of methods.

### Derivation
The Madelung Equations can be derived from the Schr&ouml;dinger Equation by first writing the wave function in polar form, 

$$\psi = \sqrt{\frac{1}{m}\rho(\vec{x},t)}e^{iS/\hbar}$$

where $\rho$ is the mass density and $S$ is the quantum action. Substituting this into the Schr&ouml;dinger Equation 

$$i\hbar\frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m}\nabla^2 \psi + V\psi$$

and separating the real and imaginary parts yields mass density conservation and the quantum Hamilton-Jacobi equation, respectively.

We define velocity as the gradient of the quantum action, $\vec{v} = \nabla S/m$, and mass density as the square of the wave function, $\rho = m|\psi|^2$. Substituting these into the quantum Hamilton-Jacobi equation yields the Madelung Equations.

Additionally, one can also find that probability current density $\vec{j}$ follows the same form as the classical hydrodynamics current density,

$$\vec{j} = \frac{1}{m} \rho \vec{v}$$

### Numerical Discretization
We discretize the Madelung Equations using a technique common to classiscal computational fluid dynamics: smoothed particle hydrodynamics (SPH). One benefit of SPH over mesh-based methods is that it is a Lagrangian method, which is useful for adapting to complex or unbounded domains. 

Our implementation follows that outlined by [Mocz and Succi, 2015](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.91.053304), adapted to two dimensions. 



We choose a Gaussian kernel for the smoothing function as opposed to a cubic-spline. Although the cubic-spline has compact support (i.e. it is zero outside of a certain radius) and thus provides computational benefit over the Gaussian kernel, it is less accurate.



## Installation
This project has been tested using [Python 3.11.3](https://www.python.org/downloads/release/python-3113/) and [Taichi 1.6.0](https://github.com/taichi-dev/taichi/releases), on Windows and Linux.

You can install the required packages using the following command:
```bash
pip install -r requirements.txt
```


