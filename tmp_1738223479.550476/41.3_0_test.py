from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from math import exp



def Conversion(g, pref, t, dep_order):
    '''This function calculates the biomass conversion matrix M
    Inputs:
    g: growth rates based on resources, 2d numpy array with dimensions [N, R] and float elements
    pref: species' preference order, 2d numpy array with dimensions [N, R] and int elements between 1 and R
    t: temporal niches, 1d numpy array with length R and float elements
    dep_order: resource depletion order, a tuple of length R with int elements between 1 and R
    Outputs:
    M: conversion matrix of biomass from resource to species. 2d float numpy array with dimensions [R, N].
    '''

    N, R = g.shape

    M = np.zeros((R, N))

    # Create a list of indices ordered by dep_order
    depletion_indices = [dep_order.index(r+1) for r in range(R)]

    # Iterate over each resource's position in depletion order
    for dep_index, resource in enumerate(dep_order):
        time_available = t[dep_index]

        for species in range(N):
            # Check if current resource is in species' preference list
            if resource in pref[species]:
                pref_index = np.where(pref[species] == resource)[0][0]

                growth_rate = g[species, resource - 1]

                biomass_conversion = exp(growth_rate * time_available) - 1

                M[resource - 1, species] = biomass_conversion

    return M


def GetResPts(M):
    '''This function finds the endpoints of the feasibility convex hull
    Inputs:
    M: conversion matrix of biomass, 2d float numpy array of size [R, N]
    Outputs:
    res_pts: a set of points in the resource supply space that marks the region of feasibility. 2d float numpy array of size [R, N].
    '''



    R, N = M.shape

    # List to store the extreme points
    extreme_points = []

    # Iterate over each resource
    for i in range(R):
        # Create a constraint to ensure x[i] is maximized, by minimizing its negative
        c = -M[i, :]

        # Linear inequality constraints: Mx >= 0
        A_ub = -M
        b_ub = np.zeros(R)

        # Linear equality constraint: sum(x) = 1
        A_eq = np.ones((1, N))
        b_eq = np.array([1])

        # Non-negativity bounds for x
        bounds = [(0, None) for _ in range(N)]

        # Solve the linear program
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        # If the optimization is successful, store the result as an extreme point
        if result.success:
            extreme_points.append(result.x)

    # Convert the list of extreme points into a numpy array and transpose it
    res_pts = np.array(extreme_points).T

    return res_pts



# Background: In ecological modeling, the structural stability of a community refers to its ability to maintain coexistence under varying environmental conditions. For a community with N species and R resources where N=R, the structural stability can be determined by calculating the volume (or in this case, the area for N=R) of the region in the resource supply simplex that supports stable coexistence. This region is defined by the determinant of the conversion matrix M, which represents the transformation of resource availability into species biomass. The simplex is the geometric representation of all possible resource distributions summing to 1. The determinant of M gives the volume of the parallelepiped formed by the columns of M, which, when divided by the total volume of the simplex, gives the fraction of the total resource space that allows for stable coexistence, known as the structural stability.



def StrucStability(g, pref, t, dep_order):
    '''This function gets the community's structural stability
    Inputs:
    g: growth rates based on resources, 2d numpy array with dimensions [N, R] and float elements
    pref: species' preference order, 2d numpy array with dimensions [N, R] and int elements between 1 and R
    t: temporal niches, 1d numpy array with length R and float elements
    dep_order: resource depletion order, a tuple of length R with int elements between 1 and R
    Outputs:
    S: structural stability of the community, float
    '''

    N, R = g.shape

    # Calculate the conversion matrix M
    M = np.zeros((R, N))
    for dep_index, resource in enumerate(dep_order):
        time_available = t[dep_index]
        for species in range(N):
            if resource in pref[species]:
                pref_index = np.where(pref[species] == resource)[0][0]
                growth_rate = g[species, resource - 1]
                biomass_conversion = exp(growth_rate * time_available) - 1
                M[resource - 1, species] = biomass_conversion

    # Calculate the determinant of M
    det_M = np.linalg.det(M)

    # Calculate the volume of the entire simplex
    # The volume of the simplex is 1/(N-1)! for N=R, but since N=R, it's simply 1/(N-1)!
    # For N=2, it's 1, for N=3, it's 1/2, etc.
    simplex_volume = 1 / np.math.factorial(N - 1)

    # Structural stability S is the fraction of the volume of the region defined by M
    # to the volume of the entire simplex
    S = abs(det_M) / simplex_volume

    return S


try:
    targets = process_hdf5_to_tuple('41.3', 3)
    target = targets[0]
    g = np.array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]])
    pref = np.array([[1, 2, 3],
           [2, 1, 3],
           [3, 1, 2]])
    dep_order = (1, 2, 3)
    t = np.array([1, 0, 0])
    assert np.allclose(StrucStability(g, pref, t, dep_order), target)

    target = targets[1]
    g = np.array([[0.68879706, 0.8834816 , 0.70943619],
           [1.04310011, 0.8411964 , 0.86002165],
           [0.97550015, 0.84997877, 1.04842294]])
    pref = np.array([[2, 3, 1],
           [1, 3, 2],
           [3, 1, 2]])
    dep_order = (3, 1, 2)
    t = np.array([0.51569821, 0.57597405, 4.12085303])
    assert np.allclose(StrucStability(g, pref, t, dep_order), target)

    target = targets[2]
    g = np.array([[0.79099249, 1.00928232, 0.90901695, 1.07388973],
           [0.89646902, 0.79124502, 0.79294553, 1.18807732],
           [0.78464268, 1.04435014, 0.97980406, 1.00469375],
           [0.85474971, 0.9244668 , 1.27430835, 0.47863501]])
    pref = np.array([[4, 2, 3, 1],
           [4, 1, 3, 2],
           [2, 4, 3, 1],
           [3, 2, 1, 4]])
    dep_order = (4, 3, 1, 2)
    t = np.array([1.51107846, 0.88238109, 1.58035451, 0.43578957])
    assert np.allclose(StrucStability(g, pref, t, dep_order), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e