# Physics Computations and Modelling

This repository showcases selected projects from my **MPhys in Theoretical Physics**, demonstrating analytical problem-solving, numerical modelling, and programming applied to physical systems.  
The projects highlight skills in **Python**, **NumPy**, **SciPy**, **Matplotlib**, and **Numba**, with applications spanning **quantum mechanics**, **simulation**, and **machine learning**.

## 🧮 Contents

### 1. [2D Ising Model Ferromagnetic-Paramagnetic Phase Transtion Modelling](./numba_acceleration)
Code and Results --- Ref.["Main_Script_With_Numba.ipynb"]

Investigation of **just-in-time compilation** using Numba to accelerate computational physics code.  
Benchmarks performance improvements over standard Python implementations and analyses scalability for large numerical systems.
Simulated the 2D Ising model using the Metropolis–Hastings Monte Carlo algorithm to study the ferromagnetic–paramagnetic phase transition. The code computed the magnetisation, energy, heat capacity, and susceptibility across temperature ranges for lattice sizes up to 30×30.

The critical temperature was estimated from the peaks of heat capacity and magnetic susceptibility, yielding a closest value of Tc = 2.259 J/kB, corresponding to a 0.4 % error from the theoretical value of 2.269 J/kB.

*Skills:* Python optimisation, JIT compilation, scientific computing, benchmarking.

---

### 2. [Quantum Ising Chain Simulation - Producing the Lieb-Robinson Light Cone ](./quantum_ising_model)
Code --- Ref.["Collated 1D Ising Code.py"]
Results --- Ref.["Lieb-Robinson Light Cone.png"]

Python implementation of the **1D transverse-field Ising model**, including Hamiltonian construction, eigenvalue analysis, and time-evolution simulations.  
Visualises magnetisation dynamics over time and explores the effects of perturbations on the quantum ground state.
Simulated the time evolution of a 1D quantum spin chain to reproduce the Lieb–Robinson light cone, which bounds the propagation speed of quantum correlations.
The Hamiltonian included nearest-neighbour spin coupling and an external field, and the central spin was flipped to create a localised excitation.

Using matrix exponentiation to solve the Schrödinger equation numerically, the simulation tracked site-resolved magnetisation

\langle \sigma_z^j(t) \rangle

over time. The resulting heatmap reproduced the light-cone pattern predicted by Lieb and Robinson, showing that correlations propagate outward at a finite maximum velocity — consistent with experimental results from trapped-ion systems.

*Skills:* Quantum systems, matrix algebra, numerical simulation, scientific plotting.

---

### 3. [Machine Learning in Physics (in progress)](./machine_learning_physics)


---

## ⚙️ Technologies Used
`Python` · `NumPy` · `SciPy` · `Matplotlib` · `Numba` · `Scikit-learn` · `Git` · `LaTeX`

---

## 💡 About
Created by **Liam Davies**, MPhys Theoretical Physics student.  
Interested in **innovation, technology, and intellectual property**, particularly in roles bridging **science and law** such as **patent attorney training**.

- 📍 Location: UK  
- 📧 Contact: liamdavies3427@gmail.com  
- 🌐 LinkedIn: https://www.linkedin.com/in/liam-davies-uol/

