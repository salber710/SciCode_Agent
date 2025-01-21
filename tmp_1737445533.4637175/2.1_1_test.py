try:
    import numpy as np
    from scipy.integrate import simps
    
    
    
    # Background: 
    # The simulation of light diffraction through a lens involves understanding how a Gaussian beam propagates through optical elements. 
    # A Gaussian beam is characterized by its beam waist and divergence, and its electric field amplitude decreases exponentially from the center.
    # When a Gaussian beam passes through a lens, the lens focuses the beam, altering its phase and intensity distribution. 
    # The refractive index of the lens material and its geometrical properties (thickness, radius of curvature) are crucial in determining the focal length and the phase shift imparted to the beam.
    # The propagation of the beam can be modeled using the Fresnel diffraction integral, which considers the effect of the lens as a phase transformation.
    # The intensity distribution on a plane after the lens can be computed by evaluating the squared magnitude of the complex electric field.
    
    
    
    def simulate_light_diffraction(n, d, RL, R0, lambda_):
        '''Function to simulate light diffraction through a lens and compute the intensity distribution.
        Inputs:
        n (float): Refractive index of the lens material, e.g., 1.5062 for K9 glass.
        d (float): Center thickness of the lens in millimeters (mm).
        RL (float): Radius of curvature of the lens in millimeters (mm).
        R0 (float): Radius of the incident light beam in millimeters (mm).
        lambda_ (float): Wavelength of the incident light in millimeters (mm), e.g., 1.064e-3 mm for infrared light.
        Outputs:
        Ie (numpy.ndarray): A 2D array of intensity values of the diffraction pattern where Ie[i][j] is the ith x and jth y.
        '''
        
        # Constants for discretization
        mr2 = 51
        ne2 = 61
        mr0 = 81
        
        # Simulation space and parameters
        f = RL / (n - 1)  # Focal length of the lens
        k = 2 * np.pi / lambda_  # Wavenumber
        dx = 2 * R0 / (mr0 - 1)  # Discretization step for initial field radius
        dr = 2 * R0 / (mr2 - 1)  # Discretization step for radial points in the simulation space
        
        # Generating radial and angular points
        r = np.linspace(-R0, R0, mr0)
        theta = np.linspace(0, 2 * np.pi, ne2)
        
        # Initial Gaussian beam profile
        E0 = np.exp(-(r**2) / (R0**2))
        
        # Phase shift introduced by the lens
        phase_shift = np.exp(-1j * k * (r**2) / (2 * f))
        
        # Complex field after the lens
        E_lens = E0 * phase_shift
        
        # Intensity distribution calculation
        Ie = np.zeros((mr2, mr2))
        
        for i in range(mr2):
            for j in range(mr2):
                rho = np.sqrt((r[i]**2) + (r[j]**2))
                if rho <= R0:
                    # Calculate intensity via integration over theta and radial components
                    integrand = lambda t: np.abs(E_lens[i] * np.exp(-1j * k * rho * np.cos(t)))**2
                    Ie[i, j] = simps(integrand(theta), theta)
        
        # Normalize intensity
        Ie /= np.max(Ie)
        
        return Ie
    

    from scicode.parse.parse import process_hdf5_to_tuple
    targets = process_hdf5_to_tuple('2.1', 4)
    target = targets[0]

    n=1.5062
    d=3
    RL=0.025e3
    R0=1
    lambda_=1.064e-3
    assert np.allclose(simulate_light_diffraction(n, d, RL, R0, lambda_), target)
    target = targets[1]

    n=1.5062
    d=4
    RL=0.05e3
    R0=1
    lambda_=1.064e-3
    assert np.allclose(simulate_light_diffraction(n, d, RL, R0, lambda_), target)
    target = targets[2]

    n=1.5062
    d=2
    RL=0.05e3
    R0=1
    lambda_=1.064e-3
    assert np.allclose(simulate_light_diffraction(n, d, RL, R0, lambda_), target)
    target = targets[3]

    n=1.5062
    d=2
    RL=0.05e3
    R0=1
    lambda_=1.064e-3
    intensity_matrix = simulate_light_diffraction(n, d, RL, R0, lambda_)
    max_index_flat = np.argmax(intensity_matrix)
    assert (max_index_flat==0) == target
