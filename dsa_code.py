%matplotlib inline
%precision 16
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import solve_bvp as scipy_solve_bvp
from scipy.interpolate import CubicSpline

# ODE solver for step 2 of the DSA algorithm
def solve_ode(mu, sigma_s, sigma_t, Q, f_L, f_R, x_L, x_R, phi_k, x_domain):
    # Create a spline object from phi_k and x_domain
    spline_phi_k = CubicSpline(x_domain, phi_k)
    # Define the ODE to be solved for a given mu
    def ode_func(x, phi_solving):
        # Use the spline object to interpolate phi_k at x
        phi_k_interp = spline_phi_k(x)
        return (-sigma_t * phi_solving + sigma_s * phi_k_interp + Q(x)) / mu

    if mu > 0:
        x_span = (x_L, x_R)  # Integrate from left to right
        phi_0 = [f_L]
    else:
        x_span = (x_R, x_L)  # Integrate from right to left
        phi_0 = [f_R]

    # Solve the ODE
    sol = solve_ivp(ode_func, x_span, phi_0, rtol=1e-12, atol=1e-12)

    # Interpolate the solution back onto the x_domain
    # If mu > 0, interpolate directly; if mu < 0, reverse sol.t and sol.y[0] before interpolating
    if mu > 0:
        t_eval = sol.t
        y_eval = sol.y[0]
    else:
        t_eval = sol.t[::-1]
        y_eval = sol.y[0][::-1]

    # Creating a spline for the solution interpolation
    solution_spline = CubicSpline(t_eval, y_eval)
    phi_solved = solution_spline(x_domain)
    return phi_solved


# BVP solver for step 3 of the DSA algorithm
def custom_solve_bvp(x_domain, sigma_s, sigma_a, sigma_t, phi_half_k, phi_k):
    # Create spline objects for interpolation
    spline_phi_k = CubicSpline(x_domain, phi_k)
    spline_phi_half_k = CubicSpline(x_domain, phi_half_k)
    # Define the BVP system of equations
    def bvp_ode(x, y):
        # y[0] = delta, y[1] = nabla
        phi_k_interp = spline_phi_k(x)
        phi_half_k_interp = spline_phi_half_k(x)
        source_term = 3 * sigma_t * (-sigma_s * (phi_half_k_interp - phi_k_interp) + sigma_a * y[0])
        return np.array([y[1], source_term])  # Corresponding to d delta/dx = nabla, d nabla/dx = source term

    def bvp_bc(ya, yb):
        # Boundary conditions for delta and nabla at the boundaries of the domain
        # ya[0] = delta(x_L), ya[1] = nabla(x_L)
        # yb[0] = delta(x_R), yb[1] = nabla(x_R)
        gamma_L = gamma_R = 0.710446
        return np.array([
            ya[0] - gamma_L / sigma_t * ya[1],  # BC at x_L
            yb[0] + gamma_R / sigma_t * yb[1]  # BC at x_R
        ])

    # Initial guess for y (solution), where y[0] = delta, y[1] = nabla
    y_guess = np.zeros((2, len(x_domain)))

    # Using SciPy's solve_bvp
    sol = scipy_solve_bvp(bvp_ode, bvp_bc, x_domain, y_guess, tol=1e-12)
    return sol.sol(x_domain)[0]  # Returning delta(x) over the x_domain

# Now, we define the iterative process for the DSA algorithm
def DSA_algorithm(phi_initial, iterations=5, tol=1e-9):
    phi_k = phi_initial.copy()
    errors = []
    for k in range(iterations):
        phi_half_k = np.zeros_like(phi_k)
        delta_half_k = np.zeros_like(phi_k)
        # Step 2: Solve the ODE for each mu
        for mu in mu_domain:
            phi_half_k += solve_ode(mu, sigma_s, sigma_t, Q, f_L, f_R, x_L, x_R, phi_k, x_domain)

        # Average over mu
        phi_half_k /= mu_domain.size

        # Step 3: Solve the BVP
        delta_half_k = custom_solve_bvp(x_domain, sigma_s, sigma_a, sigma_t, phi_half_k, phi_k)

        # Step 4: Update the scalar flux approximation
        phi_k_plus_1 = phi_half_k + delta_half_k

        # Add the relative error to the errors array
        norm_diff = np.linalg.norm(phi_k_plus_1 - phi_k)
        errors.append(norm_diff)

        if norm_diff != 0 and norm_diff < tol: # We want real convergence, not just repeating the initial guess
            print(f"The algorithm converged with the desired tolerance: {tol} for sigma_s={sigma_s}")
            break

        # Prepare for the next iteration
        phi_k = phi_k_plus_1

    print(errors)
    return phi_k, errors


# Now that we have defined all the functions, we can define the parameters

# Constants and domains
x_L, x_R = 0, 1  # Domain for x
mu_domain = np.linspace(-1, 1, 100)  # Discretized mu domain
x_domain = np.linspace(x_L, x_R, 100)  # Discretized x domain

# Initial guess for phi
phi_initial = np.zeros(len(x_domain))

# Boundary conditions for step 2 (zero flux at the boundary)
f_L = 0
f_R = 0

# Source term that respects the steady-state condition: there is no accumulation of neutrons
Q = lambda x: np.sin(2 * np.pi * x)

# Parameters: cross-sections
sigma_s_array = [0.95, 0.5, 0.005]  # Scattering cross section
sigma_a_array = [0.05, 0.5, 0.995]  # Absorption cross section
sigma_t = 1  # Total cross section (always 1 in our case as the sum of the scattering and the absorption cross sections)

fig, (ax_flux_0, ax_flux_1, ax_flux_2, ax_error) = plt.subplots(4, 1, figsize=(8, 18))  # Arrangement for plotting flux and errors
axes_flux = [ax_flux_0, ax_flux_1, ax_flux_2]

print('Absolute error values in the DSA algorithm :')
for i in range(len(sigma_s_array)):
    sigma_s = sigma_s_array[i]
    sigma_a = sigma_a_array[i]
    phi_approximation, errors = DSA_algorithm(phi_initial, iterations=10, tol=1e-6)
    axes_flux[i].plot(x_domain, phi_approximation, label='DSA Result'.format(sigma_s))
    axes_flux[i].plot(x_domain, 3*sigma_s*Q(x_domain) / (4*np.pi**2 + 3*sigma_a*sigma_s), label='1D Diffusion Result'.format(sigma_s))
    axes_flux[i].plot(x_domain, Q(x_domain) / sigma_a, label='Simplification of Neutron Transport Equation'.format(sigma_s))
    ax_error.loglog(range(1, len(errors) + 1), errors, label=r'$\sigma_S={}$'.format(sigma_s))

ax_flux_0.set_title(r'Scalar Flux Distribution for $\sigma_S=0.95$')
ax_flux_0.set_xlabel(r'Position $x$')
ax_flux_0.set_ylabel(r'Scalar Flux $\phi(x)$')
ax_flux_0.legend()
ax_flux_0.grid(True)

ax_flux_1.set_title(r'Scalar Flux Distribution for $\sigma_S=0.5$')
ax_flux_1.set_xlabel(r'Position $x$')
ax_flux_1.set_ylabel(r'Scalar Flux $\phi(x)$')
ax_flux_1.legend()
ax_flux_1.grid(True)

ax_flux_2.set_title(r'Scalar Flux Distribution for $\sigma_S=0.005$')
ax_flux_2.set_xlabel(r'Position $x$')
ax_flux_2.set_ylabel(r'Scalar Flux $\phi(x)$')
ax_flux_2.legend()
ax_flux_2.grid(True)

ax_error.set_title('Convergence Errors of the DSA Algorithm')
ax_error.set_xlabel('Iteration')
ax_error.set_ylabel('Absolute Error')
ax_error.legend()
ax_error.grid(True)

fig.tight_layout()

plt.show()