# This is the instruction sheet for running the solution for Worksheet 3 problems.

### The Jupyter notebook `WS3_notebook.ipynb` contains solution for all problems Worksheet 3.

Separate `.py` files are also provided for each problem.

## Problem 1
- Run the file `WS3_1.py` to generate the solution of damped harmonic oscillator. The figures for this problems are:        
  1. `Damped_oscillator_Under-damped.jpg`: This is for the *Under-Damped* Case.
  2. `Damped_oscillator_Over-damped.jpg`: This is for the *Over-Damped* Case.

## Problem 2
- Run the file `WS3_2_0.py` to generate solution for simple harmonic oscillator. The figure for this problem is: `anh_osc_L0.jpg`.
- Run the file `WS3_2_a.py` to generate solution for part **a** of this problem, *"Plot of energy vs time for 3 different step sizes"*. The figure for this problem is: `anh_osc_energy.jpg`.       
  For this part, the Y-axis is set to be `Energy - Initial Energy`. This is done to avoid auto scaling of the Y-axis by Matplotlib.
- Run the file `WS3_2_b.py` to generate solution for part **b** of this problem, *"Phase Space plot for different values of Lambda"*. The figure for this problem is: `anh_osc_phase_space.jpg`.
- **Comment on the phase-space plot:** One can see that for higher values of $\lambda$, the plot tends to more rectangular plot, suggesting that the particle spends longer time at the extreme velocities and then quickly decays to $v=0$, which is expected for $x^3$ force term.