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
    # u''(r) = f(r)*u(r). It is especially useful for quantum mechanics problems like solving the radial Schrödinger equation.
    # The method is advantageous due to its high accuracy for step sizes that are relatively large. The Numerov method is an
    # implicit method that uses a combination of values at the current, previous, and next steps to calculate the solution.
    # It assumes that the function f(r) is known at discrete points, and uses initial conditions for u(r) and its derivative
    # to propagate the solution over a range of r values. The basic update formula in the Numerov method is:
    # u(r_next) = (2*(1 - 5*h^2*f(r)/12)*u(r) - (1 + h^2*f(r_prev)/12)*u(r_prev)) / (1 + h^2*f(r_next)/12),
    # where h is the step size. This formula allows us to iteratively solve for u(r) across the grid.
    
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
    
        # Number of points in the radial grid based on the input array f_in
        n_points = len(f_in)
        
        # Pre-allocate the solution array
        u = np.zeros(n_points)
        
        # Set initial conditions
        u[0] = u_at_0
        # Using the derivative, estimate the second point using a Taylor expansion
        u[1] = u_at_0 + step * up_at_0
    
        # Apply the Numerov method to solve for u(r) across the grid
        for i in range(1, n_points - 1):
            h2 = step**2
            # Numerov update formula
            u[i + 1] = (2 * (1 - 5 * h2 * f_in[i] / 12) * u[i]
                        - (1 + h2 * f_in[i - 1] / 12) * u[i - 1]) / (1 + h2 * f_in[i + 1] / 12)
    
        return u
    

    from scicode.parse.parse import process_hdf5_to_tuple
    targets = process_hdf5_to_tuple('12.2', 3)
    target = targets[0]

    assert np.allclose(Numerov(f_Schrod(1,3, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)
    target = targets[1]

    assert np.allclose(Numerov(f_Schrod(1,2, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)
    target = targets[2]

    assert np.allclose(Numerov(f_Schrod(2,3, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)
