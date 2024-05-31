This project is aimed at implementing the DSA algorithm to approximate solutions to the 1D neutron transport equation. The coding part is done in Python using scipy.integrate and scipy.interpolate routines.

* The DSA results are compared with 2 simplifications of the original 1D transport equation, and for all of the parameters the DSA result lies above the 1D difussion result and below the simplified 1D transport equation result (except for the region close to the boundaries).

* The algorithm demonstrates fast absoulte convergence (the absolute error is the norm of the difference between the iterations): the  tolerance level of $10^{-6}$ is achieved in fewer than 10 iterations.

* The values at the boundaries are not correctly solved for by the algorithm. This inconsistency might be caused by the accumulation of the numerical errors over the iterations. A potential next step would be to find a specific approach for the solutions close to the boundary.
