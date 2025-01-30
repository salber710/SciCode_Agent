from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

from scipy import integrate
from scipy import optimize
import numpy as np


def f_Schrod(energy, l, r_grid):
    # Constants
    hbar = 1.0545718e-34  # Reduced Planck's constant in J*s
    m = 9.10938356e-31    # Mass of electron in kg
    e = 1.602176634e-19   # Elementary charge in C
    epsilon_0 = 8.854187817e-12  # Vacuum permittivity in F/m
    Z = 1  # Atomic number for hydrogen

    # Precompute constants
    factor = 2 * m / hbar**2
    V_prefactor = -Z * e**2 / (4 * np.pi * epsilon_0)

    # Calculate f(r) using numpy operations for efficiency
    V_r = np.divide(V_prefactor, r_grid, out=np.full_like(r_grid, -np.inf), where=r_grid!=0)
    centrifugal_term = np.divide(l * (l + 1), r_grid**2, out=np.full_like(r_grid, np.inf), where=r_grid!=0)
    f_r = factor * (V_r - energy) + centrifugal_term

    # Handle special case at r = 0
    if r_grid[0] == 0:
        f_r[0] = factor * (-energy) + l * (l + 1) / r_grid[1]**2  # Use small r value for approximation

    return f_r



def Numerov(f_in, u_at_0, up_at_0, step):
    n = len(f_in)
    u = np.zeros(n)
    u[0] = u_at_0
    if n > 1:
        u[1] = u_at_0 + step * up_at_0 + 0.5 * step**2 * f_in[0] * u_at_0

    for i in range(1, n - 1):
        u[i + 1] = (2 * u[i] - u[i - 1] + (step**2 / 12) * (10 * f_in[i] * u[i] + f_in[i + 1] * u[i] + f_in[i - 1] * u[i - 1])) / (1 - (step**2 / 12) * f_in[i + 1])

    return u



def compute_Schrod(energy, r_grid, l):
    f_r = f_Schrod(energy, l, r_grid)
    step = r_grid[0] - r_grid[1]
    u_r = Numerov(f_r, 0, -1e-7, step)
    
    # Reverse u_r for normalization from the largest radius
    u_r_reversed = u_r[::-1]
    r_grid_reversed = r_grid[::-1]
    
    # Normalize using Simpson's rule
    integral = np.trapz(u_r_reversed**2, r_grid_reversed)
    ur_norm = u_r / np.sqrt(integral)
    
    return ur_norm


def shoot(energy, r_grid, l):


    # Compute the wavefunction at the grid points
    wavefunction = compute_Schrod(energy, r_grid, l)
    
    # Normalize the wavefunction by dividing by r^l, handling r=0 case
    normalized_wavefunction = np.where(r_grid != 0, wavefunction / r_grid**l, 0)
    
    # Use a hyperbolic tangent model to extrapolate to r=0 using the first two points
    # Fit a function of the form a * tanh(b * r) + c
    # Approximate b as the inverse of the average of the first two r values
    b_approx = 1 / np.mean(r_grid[:2])
    a_approx = (normalized_wavefunction[1] - normalized_wavefunction[0]) / (np.tanh(b_approx * r_grid[1]) - np.tanh(b_approx * r_grid[0]))
    c_approx = normalized_wavefunction[0] - a_approx * np.tanh(b_approx * r_grid[0])
    
    # The extrapolated value at r=0 is the constant term c
    f_at_0 = c_approx
    
    return f_at_0



def find_bound_states(r_grid, l, energy_grid):
    bound_states = []
    def shoot_energy(energy):
        return shoot(energy, r_grid, l)
    
    # Use a combination of exhaustive search and root finding with a dynamic step size
    step = 1
    i = 0
    while i < len(energy_grid) - 1 and len(bound_states) < 10:
        E1 = energy_grid[i]
        E2 = energy_grid[min(i + step, len(energy_grid) - 1)]
        if shoot_energy(E1) * shoot_energy(E2) < 0:
            energy = brentq(shoot_energy, E1, E2)
            bound_states.append((l, energy))
            step = 1  # Reset step size after finding a root
        else:
            step *= 2  # Increase step size to skip broader ranges
        i += step
    
    return bound_states


def sort_states(bound_states):
    # Sort by energy with a secondary sort by angular momentum using a logarithmic adjustment
    return sorted(bound_states, key=lambda x: (x[1], -math.log1p(x[0] * 0.0001)))



def calculate_charge_density(bound_states, r_grid, Z):
    sorted_states = sort_states(bound_states)
    charge_density = np.zeros_like(r_grid)
    electrons_remaining = Z

    for l, energy in sorted_states:
        degeneracy = 2 * (2 * l + 1)
        wavefunction = compute_Schrod(energy, r_grid, l)
        density_contribution = wavefunction**2 * degeneracy
        if electrons_remaining <= 0:
            density_contribution *= 0
        elif electrons_remaining < degeneracy:
            density_contribution *= (electrons_remaining / degeneracy)
        charge_density += density_contribution
        electrons_remaining -= degeneracy

    return charge_density



def calculate_HartreeU(charge_density, u_at_0, up_at_0, step, r_grid, Z):
    # Constants
    four_pi = 4 * np.pi

    # Initialize arrays
    U = np.zeros_like(r_grid)
    U[0] = u_at_0
    if len(r_grid) > 1:
        U[1] = U[0] + up_at_0 * step

    # Source term for the Poisson equation
    source = -8 * np.pi * charge_density

    # Numerov's method constants
    h2 = step**2
    h12 = h2 / 12

    # Numerov's method integration
    for i in range(1, len(r_grid) - 1):
        r = r_grid[i]
        r_plus = r_grid[i + 1]
        r_minus = r_grid[i - 1]

        # Calculate the effective potential terms for Numerov's method
        g_minus = h12 * source[i - 1]
        g = h12 * source[i]
        g_plus = h12 * source[i + 1]

        # Numerov formula
        U[i + 1] = (2 * U[i] - U[i - 1] + g * U[i] + g_plus * U[i + 1]) / (1 - 5 * h12 / r**2)

    # Multiply by r to convert potential to U(r) = r * V_H(r)
    HartreeU = U * r_grid

    return HartreeU



def f_Schrod_Hartree(energy, r_grid, l, Z, hartreeU):
    # Constants
    hbar = 1.0545718e-34  # Planck constant over 2 pi, J s
    m_e = 9.10938356e-31  # Electron mass, kg
    e_charge = 1.602176634e-19  # Elementary charge, C
    epsilon_0 = 8.854187817e-12  # Permittivity of free space, F/m

    # Precomputed factors
    prefactor = 2 * m_e / hbar**2
    coulomb_factor = -Z * e_charge**2 / (4 * np.pi * epsilon_0)

    # Compute potentials using numpy operations
    V_coulomb = np.where(r_grid > 0, coulomb_factor / r_grid, -np.inf)
    V_centrifugal = l * (l + 1) * hbar**2 / (2 * m_e * np.maximum(r_grid, 1e-10)**2)
    V_hartree = np.where(r_grid > 0, hartreeU / r_grid, 0)

    # Total potential
    V_total = V_coulomb + V_centrifugal + V_hartree

    # Compute f(r)
    f_r = (V_total - energy) * prefactor

    return f_r




def compute_Schrod_Hartree(energy, r_grid, l, Z, hartreeU):
    def effective_potential(r, l, Z, U):
        return -Z / r + l * (l + 1) / (2 * r**2) + U / r

    # Initialize wavefunction array and boundary conditions
    n = len(r_grid)
    u = np.zeros(n)
    u[1] = 1e-5  # small non-zero start to avoid division by zero at r=0

    # Compute the effective potential at each point
    V = effective_potential(r_grid, l, Z, hartreeU)

    # Numerov's method constants
    h = r_grid[1] - r_grid[0]
    w = h**2 / 12

    # Numerov's method to solve the differential equation
    for i in range(1, n-1):
        f1 = 2 * (V[i-1] - energy)
        f2 = 2 * (V[i] - energy)
        f3 = 2 * (V[i+1] - energy)
        u[i+1] = ((12 - 10 * f2 * w) * u[i] - (1 + w * f1) * u[i-1]) / (1 + w * f3)

    # Normalize the wavefunction using Simpson's rule
    normalization_factor = np.sqrt(simps(u**2 * r_grid**2, r_grid))
    u_normalized = u / normalization_factor

    return u_normalized



def extrapolate_polyfit(energy, r_grid, l, Z, hartreeU):
    wavefunction = compute_Schrod_Hartree(energy, r_grid, l, Z, hartreeU)
    # Normalize wavefunction by r^l with a hyperbolic tangent to smoothly approach zero
    normalized_wavefunction = wavefunction / (np.tanh(r_grid * 1e3) * r_grid**l + 1e-10)

    # Fit a polynomial using a transformation of r to r^3 to emphasize small r behavior
    transformed_r = r_grid[:4]**3
    poly_coeffs = np.polyfit(transformed_r, normalized_wavefunction[:4], 3)

    # Extrapolate the value at r = 0 using the transformed polynomial
    u0 = np.polyval(poly_coeffs, 0)

    return u0



# Background: In quantum mechanics, bound states are solutions to the Schrödinger equation where the particle is confined to a finite region of space, typically due to a potential well. These states have discrete energy levels. To find these bound states, we can use a root-finding method to search for energies where the wavefunction satisfies certain boundary conditions. The extrapolate_polyfit function helps us estimate the wavefunction behavior at r=0, which is crucial for determining if a state is bound. The brentq method from scipy is a reliable root-finding algorithm that can be used to find these energies by identifying where the wavefunction's extrapolated value at r=0 changes sign, indicating a bound state.

def find_bound_states_Hartree(r_grid, l, energy_grid, Z, hartreeU):
    '''Input
    r_grid: a 1D array of float
    l: angular momentum quantum number; int
    energy_grid: energy grid used for search; a 1D array of float
    Z: atomic number; int
    hartreeU: the values of the Hartree term U(r) in the form of U(r)=V_H(r)r, where V_H(r) is the actual Hartree potential term in the Schrodinger equation; a 1d array of float
    Output
    bound_states: a list, each element is a tuple containing the angular momentum quantum number (int) and energy (float) of all bound states found
    '''


    bound_states = []

    def shoot_energy(energy):
        return extrapolate_polyfit(energy, r_grid, l, Z, hartreeU)

    # Use a combination of exhaustive search and root finding with a dynamic step size
    step = 1
    i = 0
    while i < len(energy_grid) - 1 and len(bound_states) < 10:
        E1 = energy_grid[i]
        E2 = energy_grid[min(i + step, len(energy_grid) - 1)]
        if shoot_energy(E1) * shoot_energy(E2) < 0:
            energy = brentq(shoot_energy, E1, E2)
            bound_states.append((l, energy))
            step = 1  # Reset step size after finding a root
        else:
            step *= 2  # Increase step size to skip broader ranges
        i += step

    return bound_states


try:
    targets = process_hdf5_to_tuple('12.12', 3)
    target = targets[0]
    energy_grid = -1.2/np.arange(1,20,0.2)**2
    r_grid = np.linspace(1e-8,100,2000)
    Z=28
    nmax = 5
    bound_states=[]
    for l in range(nmax):
        bound_states += find_bound_states(r_grid, l, energy_grid)
    charge_density = calculate_charge_density(bound_states,r_grid,Z)
    hu = calculate_HartreeU(charge_density, 0.0, 0.5, r_grid[0]-r_grid[1], r_grid, Z)
    assert np.allclose(find_bound_states_Hartree(r_grid, 0, energy_grid, Z, hu), target)

    target = targets[1]
    energy_grid = -1.2/np.arange(1,20,0.2)**2
    r_grid = np.linspace(1e-8,100,2000)
    Z=14
    nmax = 3
    bound_states=[]
    for l in range(nmax):
        bound_states += find_bound_states(r_grid, l, energy_grid)
    charge_density = calculate_charge_density(bound_states,r_grid,Z)
    hu = calculate_HartreeU(charge_density, 0.0, 0.5, r_grid[0]-r_grid[1], r_grid, Z)
    assert np.allclose(find_bound_states_Hartree(r_grid, 0, energy_grid, Z, hu), target)

    target = targets[2]
    energy_grid = -1.2/np.arange(1,20,0.2)**2
    r_grid = np.linspace(1e-8,100,2000)
    Z=28
    nmax = 5
    bound_states=[]
    for l in range(nmax):
        bound_states += find_bound_states(r_grid, l, energy_grid)
    charge_density = calculate_charge_density(bound_states,r_grid,Z)
    hu = calculate_HartreeU(charge_density, 0.0, 0.5, r_grid[0]-r_grid[1], r_grid, Z)
    assert np.allclose(find_bound_states_Hartree(r_grid, 2, energy_grid, Z, hu), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e