import numpy as np
import numpy.linalg as la

# Background: Graphene is a two-dimensional material consisting of carbon atoms arranged in a hexagonal lattice. 
# The lattice can be described using two basis vectors. In the case of graphene, the armchair direction is along 
# the y-axis, and the zigzag direction is along the x-axis. The lattice constant 'a' is the distance between 
# adjacent carbon atoms. The geometry of monolayer graphene can be generated by placing atoms at specific 
# coordinates based on these lattice vectors. The sliding distance 's' in the y-direction allows for the 
# simulation of a shifted lattice, which is useful in studying bilayer graphene and other phenomena. The 
# parameter 'n' determines the size of the supercell, which is the number of lattice sites to generate in 
# both positive and negative directions along the x and y axes.


def generate_monolayer_graphene(s, a, z, n):
    '''Generate the geometry of monolayer graphene.
    Args:
        s (float): Horizontal in-plane sliding distance.
        a (float): Lattice constant.
        z (float): z-coordinate
        n (int): supercell size
    Returns:
        atoms (np.array): Array containing the x, y, and z coordinates of the atoms.
    '''
    # Define the basis vectors for graphene
    a1 = np.array([np.sqrt(3) * a, 0])  # Zigzag direction
    a2 = np.array([np.sqrt(3) * a / 2, 3 * a / 2])  # Armchair direction

    # Initialize a list to store the atom positions
    atoms = []

    # Loop over the range to generate the supercell
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            # Calculate the position of the first atom in the unit cell
            pos1 = i * a1 + j * a2
            # Calculate the position of the second atom in the unit cell
            pos2 = pos1 + np.array([0, a])
            
            # Add the z-coordinate and sliding distance to the y-coordinate
            atoms.append([pos1[0], pos1[1] + s, z])
            atoms.append([pos2[0], pos2[1] + s, z])

    # Convert the list to a numpy array
    atoms = np.array(atoms)

    return atoms


# Background: In computational materials science, the normal vector at an atom in a lattice is crucial for understanding surface properties and interactions. For a given atom, the normal vector can be determined by considering its nearest neighbors. The cross product of vectors from the atom to its neighbors gives a vector perpendicular to the plane formed by these vectors. By averaging the cross products of vectors to the three nearest neighbors and normalizing the result, we obtain a normal vector for the atom. This vector should be adjusted to point in the correct direction based on the atom's z-coordinate: negative z-direction for atoms with z > 0 and positive z-direction for atoms with z < 0.

def assign_normals(xyzs):
    '''Assign normal vectors on the given atoms
    Args:
        xyzs (np.array): Shape (natoms, 3)
    Returns:
        normed_cross_avg (np.array): Shape (natoms, 3)
    '''



    natoms = xyzs.shape[0]
    normed_cross_avg = np.zeros((natoms, 3))

    for i in range(natoms):
        # Find the 3 nearest neighbors
        distances = la.norm(xyzs - xyzs[i], axis=1)
        nearest_indices = np.argsort(distances)[1:4]  # Exclude the atom itself

        # Calculate cross products of vectors to nearest neighbors
        cross_products = []
        for j in range(3):
            for k in range(j + 1, 3):
                vec1 = xyzs[nearest_indices[j]] - xyzs[i]
                vec2 = xyzs[nearest_indices[k]] - xyzs[i]
                cross_product = np.cross(vec1, vec2)
                cross_products.append(cross_product)

        # Average the cross products
        cross_avg = np.mean(cross_products, axis=0)

        # Normalize the average cross product
        normed_cross_avg[i] = cross_avg / la.norm(cross_avg)

        # Ensure the normal vector points in the correct z-direction
        if (xyzs[i, 2] > 0 and normed_cross_avg[i, 2] > 0) or (xyzs[i, 2] < 0 and normed_cross_avg[i, 2] < 0):
            normed_cross_avg[i] = -normed_cross_avg[i]

    return normed_cross_avg


# Background: The KC potential is a model used to describe interactions between layers in materials like graphene.
# It consists of a repulsive part and an attractive part. The repulsive part is defined by an exponential term and
# a correction function f(ρ), which depends on the transverse distance ρ between atoms. The transverse distance
# ρ is calculated using the distance vector between atoms and their normal vectors. The function f(ρ) is a polynomial
# in ρ, modulated by an exponential decay. The parameters z0, C, C0, C2, C4, delta, and lambda are specific to the
# KC potential and determine the strength and range of the interaction.

def potential_repulsive(r_ij, n_i, n_j, z0, C, C0, C2, C4, delta, lamda):
    '''Define repulsive potential.
    Args:
        r_ij: (nmask, 3)
        n_i: (nmask, 3)
        n_j: (nmask, 3)
        z0 (float): KC parameter
        C (float): KC parameter
        C0 (float): KC parameter
        C2 (float): KC parameter
        C4 (float): KC parameter
        delta (float): KC parameter
        lamda (float): KC parameter
    Returns:
        pot (nmask): values of repulsive potential for the given atom pairs.
    '''
    # Calculate the distance between atoms
    r_ij_norm = la.norm(r_ij, axis=1)

    # Calculate the transverse distances rho_ij and rho_ji
    dot_rij_ni = np.einsum('ij,ij->i', r_ij, n_i)
    dot_rij_nj = np.einsum('ij,ij->i', r_ij, n_j)
    
    rho_ij_squared = r_ij_norm**2 - dot_rij_ni**2
    rho_ji_squared = r_ij_norm**2 - dot_rij_nj**2

    # Calculate f(rho) for both rho_ij and rho_ji
    def f_rho(rho_squared):
        rho = np.sqrt(rho_squared)
        rho_delta = rho / delta
        return np.exp(-(rho_delta)**2) * (C0 + C2 * (rho_delta)**2 + C4 * (rho_delta)**4)

    f_rho_ij = f_rho(rho_ij_squared)
    f_rho_ji = f_rho(rho_ji_squared)

    # Calculate the repulsive potential
    exp_term = np.exp(-lamda * (r_ij_norm - z0))
    correction_term = C + f_rho_ij + f_rho_ji
    A = 1  # Assuming A is a constant factor, adjust if needed
    pot = exp_term * correction_term - A * (r_ij_norm / z0)**(-6)

    return pot


# Background: The KC potential model for interactions between layers in materials like graphene includes both
# repulsive and attractive components. The attractive part of the potential is primarily governed by a term
# that scales with the inverse sixth power of the distance between atoms, which is typical for van der Waals
# interactions. This term is modulated by a constant factor A and a reference distance z0. The attractive
# potential decreases with increasing distance, reflecting the long-range nature of van der Waals forces.

def potential_attractive(rnorm, z0, A):
    '''Define attractive potential.
    Args:
        rnorm (float or np.array): distance
        z0 (float): KC parameter
        A (float): KC parameter
    Returns:
        pot (float): calculated potential
    '''
    # Calculate the attractive potential using the inverse sixth power law
    pot = -A * (rnorm / z0) ** (-6)
    
    return pot



# Background: In computational materials science, taper functions are used to smoothly transition between 
# different regions of a potential or interaction model. The taper function defined here is a polynomial 
# that smoothly decreases from 1 to 0 as the distance between atoms increases from 0 to a cutoff distance 
# (R_cut). This ensures that interactions smoothly go to zero at the cutoff distance, avoiding discontinuities 
# in the potential energy surface. The polynomial form of the taper function is specifically designed to 
# have zero derivatives at both ends, ensuring a smooth transition.

def taper(r, rcut):
    '''Define a taper function. This function is 1 at 0 and 0 at rcut.
    Args:
        r (np.array): distance
        rcut (float): always 16 ang    
    Returns:
        result (np.array): taper function values
    '''
    # Calculate x_ij
    x_ij = r / rcut
    
    # Initialize the result array
    result = np.zeros_like(x_ij)
    
    # Apply the taper function where x_ij <= 1
    mask = x_ij <= 1
    result[mask] = (20 * x_ij[mask]**7 
                    - 70 * x_ij[mask]**6 
                    + 84 * x_ij[mask]**5 
                    - 35 * x_ij[mask]**4 
                    + 1)
    
    return result


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('66.5', 3)
target = targets[0]

assert np.allclose(taper(np.array([0.0]), 16), target)
target = targets[1]

assert np.allclose(taper(np.array([8.0]), 16), target)
target = targets[2]

assert np.allclose(taper(np.array([16.0]), 16), target)
