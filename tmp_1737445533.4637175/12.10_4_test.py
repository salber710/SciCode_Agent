try:
    from scipy import integrate
    from scipy import optimize
    import numpy as np
    
    # Background: 
    # The Schrödinger equation in quantum mechanics describes how the quantum state of a physical system changes over time. 
    # For a central potential problem, such as the hydrogen atom, the radial part of the wave function satisfies a differential equation.
    # When rewritten, the radial Schrödinger equation can be transformed into a form involving a radial function u(r), where
    # u''(r) = f(r) * u(r). The potential involves a Coulomb potential term along with a kinetic term. In atomic units, these terms 
    # simplify considerably. For hydrogen-like atoms with Z = 1, the potential term is -1/r. The kinetic term involves the second 
    # derivative which is related to the angular momentum quantum number l. The function f(r) encapsulates these terms as:
    # f(r) = 2 * (energy + 1/r) - l*(l+1)/r^2. This function f(r) is crucial for solving the radial Schrödinger equation using
    # numerical methods.
    
    def f_Schrod(energy, l, r_grid):
        '''Input 
        energy: a float
        l: angular momentum quantum number; an int
        r_grid: the radial grid; a 1D array of float
        Output
        f_r: a 1D array of float 
        '''
        # Calculate f(r) using the relation derived from the radial Schrödinger equation
        # f(r) = 2 * (energy + 1/r) - l*(l+1)/r^2
        # We assume Z = 1 for hydrogen-like atoms
    
        # Pre-allocate the output array
        f_r = np.zeros_like(r_grid)
    
        # Calculate f(r) for each value of r in r_grid
        for i, r in enumerate(r_grid):
            if r != 0:
                f_r[i] = 2 * (energy + 1/r) - l * (l + 1) / r**2
            else:
                f_r[i] = float('inf')  # Handle the singularity at r = 0
    
        return f_r
    
    
    # Background: The Numerov method is a numerical technique used to solve second-order linear differential equations of the form
    # u''(r) = f(r)u(r). It is particularly useful for equations where the second derivative of u is expressed as a product of u and 
    # a function f(r), as is the case with the radial part of the Schrödinger equation for hydrogen-like atoms. The method is 
    # efficient and accurate for such problems because it incorporates information from multiple points in the discretized domain, 
    # leading to higher-order accuracy. The method requires initial values of the function and its first derivative and uses a 
    # recurrence relation to compute subsequent values. The recurrence relation for the Numerov method is given by:
    # u_{n+1} = (2u_n(1 - 5h^2f_n/12) - u_{n-1}(1 + h^2f_{n-1}/12) + h^2f_{n+1}u_{n+1}/12) / (1 + h^2f_{n+1}/12),
    # where h is the step size and f_n is the value of the function f(r) at the nth point.
    
    def Numerov(f_in, u_at_0, up_at_0, step):
        '''Given precomputed function f(r), solve the differential equation u''(r) = f(r)*u(r)
        using the Numerov method.
        Inputs:
        - f_in: input function f(r); a 1D array of float representing the function values at discretized points.
        - u_at_0: the value of u at r = 0; a float.
        - up_at_0: the derivative of u at r = 0; a float.
        - step: step size; a float.
        Output:
        - u: the integration results at each point in the radial grid; a 1D array of float.
        '''
    
        # Number of points in the grid
        num_points = len(f_in)
    
        # Pre-allocate the output array for u
        u = np.zeros(num_points)
    
        # Initial conditions
        u[0] = u_at_0
        # Calculate u[1] using the Taylor expansion approximation
        u[1] = u_at_0 + step * up_at_0 + 0.5 * step**2 * f_in[0] * u_at_0
    
        # Apply the Numerov method
        for n in range(1, num_points - 1):
            # Calculate the next value of u using the Numerov recurrence relation
            u[n + 1] = ((2 * u[n] * (1 - (5/12) * step**2 * f_in[n]) -
                         u[n - 1] * (1 + (1/12) * step**2 * f_in[n - 1])) /
                        (1 + (1/12) * step**2 * f_in[n + 1]))
    
        return u
    
    
    # Background: The Schrödinger equation describes quantum mechanical systems. For hydrogen-like atoms, the radial part of 
    # the wave function can be computed by solving a second-order differential equation using f_Schrod to calculate the function 
    # f(r) and the Numerov method to compute the radial wavefunction u(r). After computing u(r), it is essential to normalize the 
    # wave function to ensure that its probability interpretation holds. Normalization ensures the total probability of finding 
    # the electron in space sums to one. Simpson's rule is a numerical integration technique that provides an efficient way to 
    # compute definite integrals, which can be used here to normalize the wave function over the given radial grid.
    
    def compute_Schrod(energy, r_grid, l):
        '''Input 
        energy: a float
        r_grid: the radial grid; a 1D array of float
        l: angular momentum quantum number; an int
        Output
        ur_norm: normalized wavefunction u(x) at x = r
        '''
    
    
        # Calculate the function f(r) using the f_Schrod function
        f_r = f_Schrod(energy, l, r_grid)
    
        # Define initial conditions for Numerov method
        u_at_0 = 0.0
        up_at_0 = -1e-7
    
        # Calculate the step size for the radial grid
        step = r_grid[0] - r_grid[1]
    
        # Solve for u(r) using the Numerov method
        u_r = Numerov(f_r, u_at_0, up_at_0, step)
    
        # Normalize u(r) using Simpson's rule over the radial grid
        # Simpson's rule for integration requires the wavefunction squared
        probability_density = u_r**2
    
        # Compute the integral using Simpson's rule over the radial grid
        integral = integrate.simpson(probability_density, r_grid)
    
        # Normalize the wave function
        ur_norm = u_r / np.sqrt(integral)
    
        return ur_norm
    
    
    # Background: In quantum mechanics, the wavefunction describes the quantum state of a particle. For the radial part of
    # the wavefunction of hydrogen-like atoms, it is necessary to perform certain transformations for numerical stability
    # and accuracy. When dealing with angular momentum (characterized by the quantum number l), the radial wavefunction u(r)
    # is often divided by r^l to simplify the boundary conditions at r = 0. In numerical simulations, especially when using
    # the shooting method to solve boundary value problems, it is common to use linear extrapolation to estimate values at
    # boundaries. Here, the task is to extrapolate the wavefunction at r = 0 using the first two points of the numerically 
    # computed wavefunction u(r) transformed by dividing by r^l. This is part of preparing the wavefunction for further 
    # computational steps, such as matching boundary conditions in the shooting method.
    
    def shoot(energy, r_grid, l):
        '''Input 
        energy: a float
        r_grid: the radial grid; a 1D array of float
        l: angular momentum quantum number; an int
        Output 
        f_at_0: float
        '''
    
        # Compute the normalized wavefunction using the compute_Schrod function
        ur_norm = compute_Schrod(energy, r_grid, l)
    
        # Divide the wavefunction by r^l for each point in the radial grid
        ur_transformed = ur_norm / (r_grid**l)
    
        # Perform linear extrapolation using the first two values of ur_transformed
        # ur_transformed[0] corresponds to u(r) / r^l at the first grid point
        # ur_transformed[1] corresponds to u(r) / r^l at the second grid point
        # Linear extrapolation formula: f_at_0 = ur_transformed[0] + (ur_transformed[1] - ur_transformed[0]) * (0 - r_grid[0]) / (r_grid[1] - r_grid[0])
    
        # Calculate the linear extrapolation at r=0
        f_at_0 = ur_transformed[0] + (ur_transformed[1] - ur_transformed[0]) * (0 - r_grid[0]) / (r_grid[1] - r_grid[0])
    
        return f_at_0
    
    
    # Background: In quantum mechanics, bound states are solutions to the Schrödinger equation where the particle is confined to a particular region in space, typically due to a potential well. These states correspond to discrete energy levels, and finding them involves solving the Schrödinger equation for energies where the wavefunction does not diverge at infinity. The shooting method is a numerical technique used to find such eigenvalues by transforming the boundary value problem into an initial value problem. In this context, the goal is to determine energies that result in a wavefunction satisfying the boundary conditions for bound states. The `shoot` function provides a way to quantify how close a trial energy is to a bound state by evaluating the wavefunction behavior at the origin. The `brentq` root-finding algorithm from SciPy is an efficient method to find roots of a function in a specified interval, which in this case is used to find energies where the wavefunction is properly normalized and finite at the boundaries. The search for bound states is restricted to a predefined energy grid.
    
    def find_bound_states(r_grid, l, energy_grid):
        '''Input
        r_grid: a 1D array of float
        l: angular momentum quantum number; int
        energy_grid: energy grid used for search; a 1D array of float
        Output
        bound_states: a list, each element is a tuple containing the angular momentum quantum number (int) and energy (float) of all bound states found
        '''
    
    
        # Define a wrapper function for the shoot function that only takes energy as an argument
        def shoot_energy(energy):
            return shoot(energy, r_grid, l)
    
        # List to store the bound states found
        bound_states = []
    
        # Maximum number of bound states to search for
        max_bound_states = 10
    
        # Iterate over the energy grid to find the bound states using root-finding
        for i in range(len(energy_grid) - 1):
            # Check if the shoot function changes sign between consecutive energies
            if shoot_energy(energy_grid[i]) * shoot_energy(energy_grid[i + 1]) < 0:
                # Find the energy where the shoot function is zero, indicating a bound state
                try:
                    energy_bound = optimize.brentq(shoot_energy, energy_grid[i], energy_grid[i + 1])
                    bound_states.append((l, energy_bound))
                    # Stop if the maximum number of bound states is reached
                    if len(bound_states) >= max_bound_states:
                        break
                except ValueError:
                    # If brentq fails to find a root, continue to the next interval
                    continue
    
        return bound_states
    
    
    # Background: When analyzing quantum systems, it is crucial to identify and sort bound states based on certain criteria.
    # Bound states correspond to specific quantum states with discrete energy levels. These states are characterized by their
    # energy values and angular momentum quantum numbers. In numerical simulations, such as those using the shooting method,
    # we obtain a list of bound states each defined by an energy and angular momentum. It is often necessary to sort these
    # states to analyze them systematically, especially when comparing states across different systems or potential parameters.
    # The sorting criteria typically prioritize energy, as lower energy states are more stable and significant in quantum systems.
    # When energies are similar or identical, the angular momentum quantum number serves as a secondary sorting criterion. 
    # To achieve this, a small weight is introduced to the angular momentum during sorting to ensure it only affects the order 
    # when the energy values are very close or identical.
    
    def sort_states(bound_states):
        '''Input
        bound_states: a list of bound states found by the find_bound_states function, each element is a tuple containing the angular momentum quantum number (int) and energy (float)
        Output
        sorted_states: a list that contains the sorted bound_states tuples according to the following rules: State with lower energy will be in front. If two states have the same energy, the one with smaller angular momentum quantum number will be in front.
        '''
        # Sort the bound states by energy first and then by angular momentum quantum number with a small weight
        sorted_states = sorted(bound_states, key=lambda state: (state[1], state[0] / 10000.0))
        
        return sorted_states
    
    
    # Background: In quantum mechanics, the charge density of a system of bound states is related to the probability density 
    # of finding the electron in a certain region of space. For hydrogen-like atoms, the radial wavefunctions describe how 
    # the probability density is distributed along the radial coordinate. Given a set of bound states, each characterized by 
    # an energy and angular momentum quantum number, we can compute the charge density by considering the occupation of each 
    # state. The degeneracy of a state depends on its angular momentum quantum number, and the charge density is calculated 
    # as the sum of the squared wavefunctions (probability densities) of the occupied states, properly weighted by their 
    # degeneracy. The charge density provides insight into the spatial distribution of the electron cloud around the nucleus.
    
    def calculate_charge_density(bound_states, r_grid, Z):
        '''Input
        bound_states: bound states found using the find_bound_states function; a list of tuples
        r_grid: the radial grid; a 1D array of float
        Z: atomic number; int
        Output
        charge_density: the calculated charge density coming from the bound states; 1D array of float
        '''
    
        # Sort the bound states using the sort_states function
        sorted_bound_states = sort_states(bound_states)
    
        # Pre-allocate the charge density array
        charge_density = np.zeros_like(r_grid)
    
        # Total number of electrons equals the atomic number Z
        total_electrons = Z
    
        # Fill the states according to their degeneracy and available electrons
        for l, energy in sorted_bound_states:
            # Degeneracy of each state: 2(2l + 1)
            degeneracy = 2 * (2 * l + 1)
    
            # If there are no electrons left to be placed, break
            if total_electrons <= 0:
                break
    
            # Determine the number of electrons that can be placed in this state
            electrons_in_state = min(degeneracy, total_electrons)
    
            # Compute the wavefunction for this state using compute_Schrod
            ur_norm = compute_Schrod(energy, r_grid, l)
    
            # Compute the contribution to the charge density from this state
            # Multiply the probability density by the number of electrons in this state
            charge_density += electrons_in_state * ur_norm**2
    
            # Subtract the occupied electrons from the total count
            total_electrons -= electrons_in_state
    
        return charge_density
    
    
    # Background: 
    # In quantum mechanics and computational physics, the Hartree potential is an essential component when considering electron-electron interactions within a many-electron system. It arises from solving the Poisson equation for the electrostatic potential generated by the charge density of the electrons. The Poisson equation, given by ∇²V_H(r) = -8πρ(r), relates the Laplacian of the Hartree potential V_H(r) to the negative of the charge density ρ(r). To solve this equation numerically, a method similar to the Numerov algorithm can be applied. In spherical coordinates, the radial part of the Laplacian simplifies the equation, and we solve for U(r) = V_H(r) * r, which is the quantity of interest. The Numerov method, which we previously used for solving the Schrödinger equation, can be adapted for this purpose by considering the charge density as the source term.
    
    
    
    def calculate_HartreeU(charge_density, u_at_0, up_at_0, step, r_grid, Z):
        '''Input
        charge_density: the calculated charge density of the bound states; 1D array of float
        u_at_0: the value of u at r = 0; float
        up_at_0: the derivative of u at r = 0; float
        step: step size; float.
        r_grid: the radial grid; a 1D array of float
        Z: atomic number; int
        Output
        x: the HartreeU term with U(r)=V_H(r)r; 1D array of float
        '''
    
        # Number of points in the radial grid
        num_points = len(r_grid)
    
        # Pre-allocate the output array for U(r)
        U_r = np.zeros(num_points)
    
        # Initial conditions for U(r) at r = 0
        U_r[0] = u_at_0
        # Calculate U_r[1] using a Taylor expansion approximation
        U_r[1] = u_at_0 + step * up_at_0 - 0.5 * step**2 * 8 * np.pi * charge_density[0] * u_at_0
    
        # Apply the Numerov method to solve the Poisson equation
        for n in range(1, num_points - 1):
            # Using Numerov-like method adapted for the Poisson equation
            U_r[n + 1] = ((2 * U_r[n] * (1 - (5/12) * step**2 * 8 * np.pi * charge_density[n]) -
                           U_r[n - 1] * (1 + (1/12) * step**2 * 8 * np.pi * charge_density[n - 1])) /
                          (1 + (1/12) * step**2 * 8 * np.pi * charge_density[n + 1]))
    
        return U_r
    
    
    # Background: In quantum mechanics, when solving the Schrödinger equation for multi-electron systems, 
    # it is essential to account for electron-electron interactions. The Hartree potential, derived from 
    # the electron charge density, represents a mean-field approximation of these interactions. When included 
    # in the Schrödinger equation, the potential modifies the original terms, resulting in a new expression 
    # for the function f(r), which is used in the differential equation u''(r) = f(r)u(r). The inclusion of 
    # the Hartree term affects the potential energy experienced by the electrons, thus altering the function 
    # f(r) in the equation. For a hydrogen-like atom with nuclear charge Z and a Hartree potential V_H(r), 
    # the modified function f(r) can be expressed as:
    # f(r) = 2 * (energy + Z/r + V_H(r)) - l*(l+1)/r^2.
    # Here, V_H(r) is given as hartreeU/r, where hartreeU is U(r) = V_H(r) * r, making it straightforward 
    # to include in the equation.
    
    def f_Schrod_Hartree(energy, r_grid, l, Z, hartreeU):
        '''Input 
        energy: float
        r_grid: the radial grid; a 1D array of float
        l: angular momentum quantum number; int
        Z: atomic number; int
        hartreeU: the values of the Hartree term U(r) in the form of U(r)=V_H(r)r, where V_H(r) is the actual Hartree potential term in the Schrodinger equation; a 1d array of float
        Output
        f_r: a 1D array of float 
        '''
    
        # Pre-allocate the output array for f(r)
        f_r = np.zeros_like(r_grid)
    
        # Calculate f(r) for each value of r in r_grid
        for i, r in enumerate(r_grid):
            if r != 0:
                # Calculate V_H(r) from hartreeU
                V_H_r = hartreeU[i] / r
                # Calculate f(r) including the Hartree potential term
                f_r[i] = 2 * (energy + Z/r + V_H_r) - l * (l + 1) / r**2
            else:
                f_r[i] = float('inf')  # Handle the singularity at r = 0
    
        return f_r
    
    
    
    # Background: In quantum mechanics, the Schrödinger equation describes how the quantum state of a physical system changes over time.
    # For systems with electron-electron interactions, the Hartree potential, derived from the electron charge density, is included 
    # to account for these interactions in a mean-field approximation. The equation for the radial wavefunction u(r) is modified to 
    # include this Hartree potential, affecting the expression for f(r) in the differential equation u''(r) = f(r)*u(r). 
    # The Numerov method is a numerical technique that efficiently solves such second-order differential equations by using a 
    # recurrence relation, making it particularly suitable for the Schrödinger equation with a Hartree potential. 
    # After solving, it is essential to normalize the wavefunction using numerical integration methods like Simpson's rule 
    # to ensure the total probability sums to one, which is crucial for its physical interpretation.
    
    
    def compute_Schrod_Hartree(energy, r_grid, l, Z, hartreeU):
        '''Input 
        energy: float
        r_grid: the radial grid; a 1D array of float
        l: angular momentum quantum number; int
        Z: atomic number; int
        hartreeU: the values of the Hartree term U(r) in the form of U(r)=V_H(r)r, where V_H(r) is the actual Hartree potential term in the Schrodinger equation; a 1d array of float
        Output
        ur_norm: normalized wavefunction u(x) at x = r
        '''
        
        # Calculate the function f(r) using the f_Schrod_Hartree function
        f_r = f_Schrod_Hartree(energy, r_grid, l, Z, hartreeU)
    
        # Define initial conditions for Numerov method
        u_at_0 = 0.0
        up_at_0 = -1e-7
    
        # Calculate the step size for the radial grid
        step = r_grid[1] - r_grid[0]
    
        # Solve for u(r) using the Numerov method
        u_r = Numerov(f_r, u_at_0, up_at_0, step)
    
        # Normalize u(r) using Simpson's rule over the radial grid
        # Simpson's rule for integration requires the wavefunction squared
        probability_density = u_r**2
    
        # Compute the integral using Simpson's rule over the radial grid
        integral = integrate.simpson(probability_density, r_grid)
    
        # Normalize the wave function
        ur_norm = u_r / np.sqrt(integral)
    
        return ur_norm
    

    from scicode.parse.parse import process_hdf5_to_tuple
    targets = process_hdf5_to_tuple('12.10', 3)
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
    assert np.allclose(compute_Schrod_Hartree(-0.5, r_grid, 2, Z, hu), target)
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
    assert np.allclose(compute_Schrod_Hartree(-0.4, r_grid, 2, Z, hu), target)
    target = targets[2]

    energy_grid = -0.9/np.arange(1,20,0.2)**2
    r_grid = np.linspace(1e-8,100,2000)
    Z=14
    nmax = 5
    bound_states=[]
    for l in range(nmax):
        bound_states += find_bound_states(r_grid, l, energy_grid)
    charge_density = calculate_charge_density(bound_states,r_grid,Z)
    hu = calculate_HartreeU(charge_density, 0.0, 0.5, r_grid[0]-r_grid[1], r_grid, Z)
    assert np.allclose(compute_Schrod_Hartree(-0.5, r_grid, 3, Z, hu), target)
