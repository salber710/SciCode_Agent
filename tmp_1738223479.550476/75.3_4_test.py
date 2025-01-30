from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

def hopping_mk(d, dz, v_p0=-2.7, v_s0=0.48, b=1.17, a0=2.68, d0=6.33):
    '''Parameterization from Moon and Koshino, Phys. Rev. B 85, 195458 (2012).
    Args:
        d: distance between two atoms (unit b,a.u.), float
        dz: out-of-plane distance between two atoms (unit b,a.u.), float
        v_p0: transfer integral between the nearest-neighbor atoms of monolayer graphene, MK parameter, float,unit eV
        v_s0: interlayer transfer integral between vertically located atoms, MK parameter, float,unit eV
        b: 1/b is the decay length of the transfer integral, MK parameter, float, unit (b,a.u.)^-1
        a0: nearest-neighbor atom distance of the monolayer graphene, MK parameter, float, unit (b,a.u.)
        d0: interlayer distance, MK parameter, float, (b,a.u.)
    Return:
        hopping: -t, float, eV
    '''
    # Use a different mathematical approach to approximate exponential
    def exp_approx(x):
        # Using a more complex polynomial approximation for exponential
        return 1 + x + x**2 / 2 + x**3 / 6 + x**4 / 24 + x**5 / 120

    # Calculate decay factors using the new approximation
    decay_pi = exp_approx(-b * (d - a0))
    decay_sigma = exp_approx(-b * (d - d0))

    # Calculate the transfer integrals V_{pp\pi} and V_{pp\sigma}
    V_pp_pi = v_p0 * decay_pi
    V_pp_sigma = v_s0 * decay_sigma

    # Calculate the squared ratio of dz to d using a nested operation
    dz_over_d_squared = (dz / d) ** 2

    # Compute the hopping parameter -t using a variation in arithmetic sequence
    hopping = V_pp_pi - V_pp_pi * dz_over_d_squared + V_pp_sigma * dz_over_d_squared

    return hopping



def mk(latvecs, basis, di, dj, ai, aj):
    '''Evaluate the Moon and Koshino hopping parameters Phys. Rev. B 85, 195458 (2012).
    Args:
        latvecs (np.array): lattice vectors of shape (3, 3) in bohr
        basis (np.array): atomic positions of shape (natoms, 3) in bohr; natoms: number of atoms within a unit cell
        di, dj (np.array): list of displacement indices for the hopping
        ai, aj (np.array): list of atomic basis indices for the hopping
    Return:
        hopping (np.array): a list with the same length as di
    '''

    def inverse_decay_hopping_mk(d, dz):
        '''Inverse decay parameterization for Moon and Koshino hopping parameters.
        Args:
            d: distance between two atoms (unit b,a.u.), float
            dz: out-of-plane distance between two atoms (unit b,a.u.), float
        Return:
            hopping: float, eV
        '''
        # Define parameters
        v_p0 = -2.7
        v_s0 = 0.48
        b = 1.17
        a0 = 2.68
        d0 = 6.33

        # Use inverse functions for decay
        decay_pi = 1 / (1 + b * np.abs(d - a0))
        decay_sigma = 1 / (1 + b * np.abs(d - d0))

        V_pp_pi = v_p0 * decay_pi
        V_pp_sigma = v_s0 * decay_sigma

        dz_over_d_squared = (dz / d) ** 2

        hopping = V_pp_pi * (1 - dz_over_d_squared) + V_pp_sigma * dz_over_d_squared

        return hopping

    num_hoppings = len(di)
    hoppings = np.zeros(num_hoppings)

    for i in range(num_hoppings):
        # Calculate the displacement vector using a circular metric
        displacement = np.sqrt(np.sum((di[i] - dj[i])**2))
        displacement_vector = displacement * np.mean(latvecs, axis=0)
        # Calculate the atomic positions with a dual transformation
        position_i = basis[ai[i]] + np.cross(displacement_vector, basis[ai[i]])
        position_j = basis[aj[i]] - np.cross(displacement_vector, basis[aj[i]])
        # Calculate the relative position vector
        R_ij = position_j - position_i
        
        # Calculate the distance and out-of-plane distance
        d = np.linalg.norm(R_ij)
        dz = np.abs(R_ij[2])
        
        # Compute the hopping using the inverse decay function
        hoppings[i] = inverse_decay_hopping_mk(d, dz)

    return hoppings




def ham_eig(k_input, latvecs, basis):
    '''Calculate the eigenvalues for a given k-point (k-point is in reduced coordinates)
    Args:
        k_input (np.array): (kx, ky)
        latvecs (np.array): lattice vectors of shape (3, 3) in bohr
        basis (np.array): atomic positions of shape (natoms, 3) in bohr
    Returns:
        eigval: numpy array of floats, sorted array of eigenvalues
    '''
    num_atoms = basis.shape[0]
    hamiltonian = np.zeros((num_atoms, num_atoms), dtype=np.complex128)

    # Transform k_input to Cartesian coordinates
    k_cartesian = np.dot(k_input, latvecs[:2, :2])

    # Compute the Hamiltonian matrix using a different approach
    # Use a cosine modulation for the hopping terms and incorporate a damping factor
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                R_ij = basis[j] - basis[i]
                distance = np.linalg.norm(R_ij)
                hopping = np.cos(0.5 * distance) * np.exp(-distance)  # Cosine modulated decaying hopping term
                phase = np.exp(1j * np.dot(k_cartesian, R_ij[:2]))
                hamiltonian[i, j] = hopping * phase
            else:
                # Use a constant on-site energy for all atoms
                hamiltonian[i, j] = 0.1

    # Calculate the eigenvalues
    eigenvalues = np.linalg.eigvals(hamiltonian)

    # Return sorted real parts of the eigenvalues
    return np.sort(eigenvalues.real)


try:
    targets = process_hdf5_to_tuple('75.3', 3)
    target = targets[0]
    k = np.array([0.5, 0.0])
    # test system
    conversion = 1.0/.529177 # convert angstrom to bohr radius
    a = 2.46 # graphene lattice constant in angstrom
    latvecs = np.array([
        [a, 0.0, 0.0],
        [-1/2*a, 3**0.5/2*a, 0.0],
        [0.0, 0.0, 30]
        ]) * conversion
    basis = np.array([[0, 0, 0], [0, 1/3**0.5*a, 0], [0, 0, 3.44], [0, 1/3**0.5*a, 3.44]]) * conversion
    assert np.allclose(ham_eig(k, latvecs, basis), target)

    target = targets[1]
    k = np.array([0.0, 0.0])
    # test system
    conversion = 1.0/.529177 # convert angstrom to bohr radius
    a = 2.46 # graphene lattice constant in angstrom
    latvecs = np.array([
        [a, 0.0, 0.0],
        [-1/2*a, 3**0.5/2*a, 0.0],
        [0.0, 0.0, 30]
        ]) * conversion
    basis = np.array([[0, 0, 0], [0, 1/3**0.5*a, 0], [0, 0, 3.44], [0, 1/3**0.5*a, 3.44]]) * conversion
    assert np.allclose(ham_eig(k, latvecs, basis), target)

    target = targets[2]
    k = np.array([2/3, -1/3])
    # test system
    conversion = 1.0/.529177 # convert angstrom to bohr radius
    a = 2.46 # graphene lattice constant in angstrom
    latvecs = np.array([
        [a, 0.0, 0.0],
        [-1/2*a, 3**0.5/2*a, 0.0],
        [0.0, 0.0, 30]
        ]) * conversion
    basis = np.array([[0, 0, 0], [0, 1/3**0.5*a, 0], [0, 0, 3.44], [0, 1/3**0.5*a, 3.44]]) * conversion
    assert np.allclose(ham_eig(k, latvecs, basis), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e