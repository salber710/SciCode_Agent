import numpy as np
import numpy.linalg as la

# Background: Graphene is a two-dimensional material consisting of carbon atoms arranged in a hexagonal lattice. 
# The lattice can be described using two basis vectors. In graphene, the armchair direction is typically along the y-axis, 
# and the zigzag direction is along the x-axis. The lattice constant 'a' is the distance between two adjacent carbon atoms. 
# In a monolayer graphene sheet, each carbon atom is bonded to three other carbon atoms, forming a hexagonal pattern. 
# The task is to generate the coordinates of the carbon atoms in a monolayer graphene sheet, considering a sliding distance 's' 
# in the y-direction, a z-coordinate 'z', and a supercell size 'n' which determines the number of lattice sites in both 
# positive and negative x and y directions.


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
    if not isinstance(n, int):
        raise TypeError("Supercell size 'n' must be an integer.")
    if n < 0:
        raise ValueError("Supercell size 'n' must be non-negative.")
    if not all(isinstance(i, (int, float)) for i in [s, a, z]):
        raise TypeError("Inputs s, a, and z must be numeric.")

    # Define the basis vectors for the graphene lattice
    a1 = np.array([np.sqrt(3) * a, 0])  # Zigzag direction
    a2 = np.array([np.sqrt(3) * a / 2, 3 * a / 2])  # Armchair direction

    # Define the basis atoms in the unit cell
    basis_atoms = [
        np.array([0, 0]),
        np.array([np.sqrt(3) * a / 2, a / 2])
    ]

    # Initialize a list to store the atom positions
    atom_positions = []

    # Loop over the supercell range
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            # Calculate the position of each atom in the supercell
            for basis in basis_atoms:
                position = i * a1 + j * a2 + basis
                # Apply the sliding distance in the y-direction
                position[1] += s
                # Append the position with the z-coordinate
                atom_positions.append([position[0], position[1], z])

    # Convert the list to a numpy array
    atoms = np.array(atom_positions)

    return atoms


# Background: In computational materials science, the normal vector at a point on a surface is a vector that is perpendicular to the surface at that point. For a graphene sheet, each carbon atom is bonded to three nearest neighbors, forming a hexagonal lattice. The normal vector at each atom can be determined by averaging the cross products of vectors to its three nearest neighbors. The cross product of two vectors results in a vector that is perpendicular to the plane containing the two vectors. By normalizing these cross products and averaging them, we obtain a normal vector for each atom. The direction of the normal vector is important; it should point outward from the surface. For atoms with a positive z-coordinate, the normal should point in the negative z-direction, and vice versa. If the calculated normal vector points in the wrong direction, it can be corrected by multiplying it by -1.



def assign_normals(xyzs):
    '''Assign normal vectors on the given atoms
    Args:
        xyzs (np.array): Shape (natoms, 3)
    Returns:
        normed_cross_avg (np.array): Shape (natoms, 3)
    '''
    natoms = xyzs.shape[0]
    normed_cross_avg = np.zeros((natoms, 3))

    if natoms < 2:
        return normed_cross_avg  # Return zero vectors if not enough atoms to compute normals

    # Iterate over each atom to calculate its normal vector
    for i in range(natoms):
        # Find the 3 nearest neighbors
        distances = la.norm(xyzs - xyzs[i], axis=1)
        nearest_indices = np.argsort(distances)[1:4]  # Exclude the atom itself

        if len(nearest_indices) < 3:
            continue  # Skip if there are not enough neighbors to form a plane

        # Calculate the cross products of vectors to the 3 nearest neighbors
        cross_products = []
        for j in range(3):
            v1 = xyzs[nearest_indices[j]] - xyzs[i]
            v2 = xyzs[nearest_indices[(j + 1) % 3]] - xyzs[i]
            cross_product = np.cross(v1, v2)
            norm = la.norm(cross_product)
            if norm == 0:
                continue  # Skip degenerate cross products (collinear vectors)
            cross_products.append(cross_product / norm)

        if not cross_products:
            continue  # Skip if no valid cross products were found

        # Average the normalized cross products
        normal = np.mean(cross_products, axis=0)

        # Normalize the averaged vector
        normal /= la.norm(normal)

        # Ensure the normal vector points in the correct z-direction
        if (xyzs[i, 2] > 0 and normal[2] > 0) or (xyzs[i, 2] < 0 and normal[2] < 0):
            normal = -normal

        normed_cross_avg[i] = normal

    return normed_cross_avg



# Background: The Kolmogorov-Crespi (KC) potential is used to model interactions between layers in layered materials like graphene. 
# The repulsive part of the KC potential is defined by an exponential term and a correction term that depends on the transverse 
# distance between atoms in different layers. The transverse distance, denoted as rho, is calculated by subtracting the squared 
# projection of the distance vector onto the normal vector from the squared distance. The function f(rho) is a polynomial 
# function of rho, modulated by an exponential decay. The repulsive potential is then a combination of these terms, 
# with parameters z0, C, C0, C2, C4, delta, and lambda defining the specific form of the potential.

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
    if delta <= 0:
        raise ValueError("delta must be positive")

    # Calculate the squared distances
    r_ij_squared = np.sum(r_ij**2, axis=1)

    # Calculate the dot products for the transverse distances
    r_ij_dot_n_i = np.sum(r_ij * n_i, axis=1)
    r_ij_dot_n_j = np.sum(r_ij * n_j, axis=1)

    # Calculate the transverse distances squared
    rho_ij_squared = r_ij_squared - r_ij_dot_n_i**2
    rho_ji_squared = r_ij_squared - r_ij_dot_n_j**2

    # Calculate f(rho) for both rho_ij and rho_ji
    f_rho_ij = np.exp(-(rho_ij_squared / delta**2)) * (
        C0 + C2 * (rho_ij_squared / delta**2) + C4 * (rho_ij_squared / delta**4)
    )
    f_rho_ji = np.exp(-(rho_ji_squared / delta**2)) * (
        C0 + C2 * (rho_ji_squared / delta**2) + C4 * (rho_ji_squared / delta**4)
    )

    # Calculate the repulsive potential
    exp_term = np.exp(-lamda * (np.sqrt(r_ij_squared) - z0))
    correction_term = C + f_rho_ij + f_rho_ji
    attractive_term = -1 * (np.sqrt(r_ij_squared) / z0)**-6  # Corrected to include the missing parameter A, assumed to be 1 for simplicity

    # Handle the case when r_ij_squared is zero to avoid division by zero in the attractive term
    if np.any(r_ij_squared == 0):
        attractive_term[r_ij_squared == 0] = 0  # Set the attractive term to zero where distance is zero

    pot = exp_term * correction_term + attractive_term

    return pot


# Background: The Kolmogorov-Crespi (KC) potential is used to model interactions between layers in layered materials like graphene.
# The attractive part of the KC potential is defined by a term that depends on the inverse sixth power of the distance between atoms.
# This term is modulated by a parameter A and the equilibrium distance z0. The attractive potential is significant at larger distances
# and is responsible for the van der Waals interactions between layers. The formula for the attractive potential is given by:
# V_ij = -A * (r_ij / z0)^-6, where r_ij is the distance between atoms i and j, z0 is a characteristic distance, and A is a parameter
# that scales the strength of the interaction.


def potential_attractive(rnorm, z0, A):
    '''Define attractive potential.
    Args:
        rnorm (float or np.array): distance
        z0 (float): KC parameter
        A (float): KC parameter
    Returns:
        pot (float or np.array): calculated potential
    '''
    # Calculate the attractive potential using the inverse sixth power law
    if z0 == 0:
        raise ValueError("z0 cannot be zero as it causes division by zero.")
    if np.any(rnorm == 0):
        # Handle the case where rnorm is zero and it's an array
        return np.where(rnorm == 0, -A, -A * (rnorm / z0)**-6)
    pot = -A * (rnorm / z0)**-6

    return pot



# Background: In computational materials science, taper functions are used to smoothly transition between different regions of a potential or interaction. 
# The taper function defined here is a polynomial that smoothly transitions from 1 to 0 as the distance between atoms increases from 0 to a cutoff distance, R_cut. 
# This ensures that interactions smoothly go to zero at the cutoff distance, avoiding discontinuities in the potential energy surface. 
# The specific polynomial form used here is a seventh-degree polynomial, which provides a smooth transition with continuous derivatives up to the sixth order. 
# The function is defined to be zero for distances greater than the cutoff, ensuring that interactions are completely turned off beyond this point.

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

    # Initialize the result array with zeros
    result = np.zeros_like(x_ij)

    # Apply the taper function only where x_ij <= 1
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
