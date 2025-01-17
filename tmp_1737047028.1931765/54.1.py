import numpy as np



# Background: 
# In the context of finite element analysis, shape functions are used to interpolate the solution over an element. 
# For a 1D linear element, the shape functions are linear polynomials that depend on the position within the element.
# The function ω^1(x) is defined for the interval [x_{i-1}, x_i] and linearly interpolates between these two points.
# Similarly, ω^2(x) is defined for the interval [x_i, x_{i+1}] and interpolates between these points.
# The parameter 'h' represents the element size, which is the distance between consecutive nodes.
# The function 'basis' computes the value of these shape functions based on the type specified by 'etype'.

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
        # Calculate ω^1(x) = (x - x_{i-1}) / h_{i-1} for x_{i-1} <= x <= x_i
        for j in range(len(p)):
            if i > 0 and p[j] >= (i-1)*h and p[j] <= i*h:
                v[j] = (p[j] - (i-1)*h) / h
            else:
                v[j] = 0.0
    elif etype == 2:
        # Calculate ω^2(x) = (x_{i+1} - x) / h_i for x_i <= x <= x_{i+1}
        for j in range(len(p)):
            if i < M-1 and p[j] >= i*h and p[j] <= (i+1)*h:
                v[j] = ((i+1)*h - p[j]) / h
            else:
                v[j] = 0.0
    else:
        raise ValueError("etype must be 1 or 2")
    
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
