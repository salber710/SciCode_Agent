
# Background: B-spline basis functions are a family of piecewise-defined polynomials that are used in numerical analysis and computer graphics for curve fitting and surface modeling. The B-spline basis functions are defined recursively using the Cox-de Boor recursion formula. The recursion starts with piecewise constant functions (degree 0) and builds up to higher degrees. The recursion formula for B-splines is:
# 
# N_{i,0}(x) = 1 if Xi[i] <= x < Xi[i+1], else 0
# N_{i,p}(x) = ((x - Xi[i]) / (Xi[i+p] - Xi[i])) * N_{i,p-1}(x) + ((Xi[i+p+1] - x) / (Xi[i+p+1] - Xi[i+1])) * N_{i+1,p-1}(x)
# 
# where N_{i,p}(x) is the B-spline basis function of degree p, Xi is the knot vector, and i is the index of the basis function. The function Bspline(xi, i, p, Xi) is designed to evaluate the value of the B-spline basis function at a given knot index xi, for a given polynomial index i, degree p, and knot vector Xi.

import numpy as np

def Bspline(xi, i, p, Xi):
    '''Inputs:
    xi : knot index, float
    i : polynomial index, integer
    p : polynomial degree of basis function, integer
    Xi : knot vector, 1d array of arbitrary size
    Outputs:
    float representing the value of the B-spline basis function at xi
    '''
    if not isinstance(i, int) or not isinstance(p, int):
        raise TypeError("Indices i and degree p must be integers.")
    if p < 0:
        raise ValueError("Degree p must be non-negative.")
    if i < 0 or i >= len(Xi) - 1:
        raise IndexError("Index i is out of the valid range.")
    if len(Xi) < 2:
        raise ValueError("Knot vector Xi must contain at least two elements.")

    if p == 0:
        # Base case: degree 0
        if Xi[i] <= xi < Xi[i+1]:
            return 1.0
        else:
            return 0.0
    else:
        # Recursive case: degree p
        denom1 = Xi[i+p] - Xi[i]
        denom2 = Xi[i+p+1] - Xi[i+1]
        
        # Avoid division by zero
        if denom1 == 0:
            alpha = 0
        else:
            alpha = (xi - Xi[i]) / denom1 if Xi[i] != Xi[i+p] else 0
        
        if denom2 == 0:
            beta = 0
        else:
            beta = (Xi[i+p+1] - xi) / denom2 if Xi[i+1] != Xi[i+p+1] else 0
        
        # Recursive computation
        N_ip = Bspline(xi, i, p-1, Xi) if denom1 != 0 else 0
        N_i1p = Bspline(xi, i+1, p-1, Xi) if denom2 != 0 else 0
        
        return alpha * N_ip + beta * N_i1p
