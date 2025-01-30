import numpy as np



# Background: In the context of finite element analysis, shape functions are used to interpolate the solution
# over an element. For a 1D linear element, the shape functions are linear polynomials that define how the 
# solution varies within the element. The shape functions for a linear element are typically defined over 
# a reference element and then mapped to the actual element. In this problem, we have two types of shape 
# functions, ω^1(x) and ω^2(x), which are defined over specific intervals. ω^1(x) is defined for the interval 
# [x_{i-1}, x_i] and linearly interpolates from 0 to 1, while ω^2(x) is defined for the interval [x_i, x_{i+1}] 
# and linearly interpolates from 1 to 0. The parameter h represents the element size, and etype determines 
# which shape function to use.

def basis(i, p, M, h, etype):
    '''Inputs
    i: int, the index of element
    p: array of arbitrary size 1,2, or 3, the coordinates
    M: int, the total number of the nodal dofs
    h: int, the element size
    etype: int, basis function type; When type equals to 1, 
    it returns ω^1(x), when the type equals to 2, it returns the value of function ω^2(x).
    Outputs
    v: array of size 1,2, or 3, value of basis function
    '''

    
    v = np.zeros_like(p)
    
    if etype == 1:
        # Calculate ω^1(x) = (x - x_{i-1}) / h for x in [x_{i-1}, x_i]
        for j in range(len(p)):
            if i > 0 and p[j] >= (i-1)*h and p[j] <= i*h:
                v[j] = (p[j] - (i-1)*h) / h
            else:
                v[j] = 0.0
    elif etype == 2:
        # Calculate ω^2(x) = (x_{i+1} - x) / h for x in [x_i, x_{i+1}]
        for j in range(len(p)):
            if i < M-1 and p[j] >= i*h and p[j] <= (i+1)*h:
                v[j] = ((i+1)*h - p[j]) / h
            else:
                v[j] = 0.0
    
    return v

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('54.1', 3)
target = targets[0]

i = 1
p = np.array([0.2])
M = 10 
h = 0.1 
etype = 1
assert np.allclose(basis(i, p, M, h, etype), target)
target = targets[1]

i = 2
p = np.array([0.5,0.1])
M = 20 
h = 0.5 
etype = 1
assert np.allclose(basis(i, p, M, h, etype), target)
target = targets[2]

i = 3
p = np.array([5,7,9])
M = 30 
h = 0.01 
etype = 2
assert np.allclose(basis(i, p, M, h, etype), target)
