try:
    from scipy import integrate
    from scipy import optimize
    import numpy as np
    
    
    
    # Background: 
    # The Schrodinger equation for a hydrogen-like atom involves the kinetic energy term and the potential energy term,
    # where the potential energy is due to the electrostatic interaction between the nucleus and the electron.
    # In spherical coordinates, the radial part of the Schrodinger equation can be separated and rewritten to resemble
    # a one-dimensional differential equation. By substituting the wave function ψ(r) with u(r) = rψ(r),
    # the equation is transformed into a second-order differential equation u''(r) = f(r)u(r).
    # For the hydrogen atom (Z=1), the potential energy term is given by -e^2/(4πε₀r), where ε₀ is the permittivity of
    # free space, and e is the elementary charge. In atomic units, the constants simplify such that the potential energy
    # becomes -1/r. The function f(r) can be expressed in terms of the radial grid r, the energy E, and the angular
    # momentum quantum number l as follows:
    # 
    # f(r) = -2(E + 1/r) + l(l + 1)/r^2
    # 
    # Here, E is the energy of the state, and l is the angular momentum quantum number.
    
    def f_Schrod(energy, l, r_grid):
        '''Input 
        energy: a float
        l: angular momentum quantum number; an int
        r_grid: the radial grid; a 1D array of float
        Output
        f_r: a 1D array of float 
        '''
        # Convert inputs to numpy arrays for vectorized operations
        r = np.array(r_grid)
        
        # Calculate the f(r) function
        f_r = -2 * (energy + 1/r) + l * (l + 1) / r**2
        
        return f_r
    

    from scicode.parse.parse import process_hdf5_to_tuple
    targets = process_hdf5_to_tuple('12.1', 3)
    target = targets[0]

    assert np.allclose(f_Schrod(1,1, np.array([0.1,0.2,0.3])), target)
    target = targets[1]

    assert np.allclose(f_Schrod(2,1, np.linspace(1e-8,100,20)), target)
    target = targets[2]

    assert np.allclose(f_Schrod(2,3, np.linspace(1e-5,10,10)), target)
