# -*- coding: utf-8 -*-

from jax import config
config.update("jax_enable_x64", True)  #double precision


import numpy as np
from jax import jit
import jax

import jax.numpy as jnp
import numpy as np
from skopt.space import Space
from skopt.sampler import Halton
from jax import jacfwd
from jax import grad, jit, vmap, pmap
from jax._src.lax.utils import (
    _argnum_weak_type,
    _input_dtype,
    standard_primitive,
)
from jax._src.lax import lax


def One_Dim_Matrix(A):
    """
    We use this function to convert a 2D array into a 1D array containing only the lower triangular matrix of the 2D array.
    : param A : a 2D array
    : return : a 1D array

    """
    dim_x, dim_y = A.shape
    #print(dim_x, dim_y)
    A = A.reshape(1, (dim_x * dim_y))
    return A


def Add_B_tomatrix_A(A, b):
    """
    Given 2 1D arrays this function appends the second array at the end of first array.
    : param A : 1D array
    : param b : 1D array
    : return : 1D array after appending array b to A

    """
    A = jnp.append(A,b)
    return A


def actual_A_1D(A):
    """
    This function takes in a 1D array and breaks it into 2 arrays.
    : param A : 1D array
    : return A_new : 1D array of length = 10
    : return b1 : 1D array of length = 4

    """

    b1 = A[16:20]
    A_new = A[0:16]
    return A_new, b1


def actual_A1_A2(A): # from the returned gradient array of 20 elements, we find the elements of array A and array B
                    # first 16 elemets belong to lower triangular elements of array A and 4 belongs to B
    A1 = A[0:20]
    A2 = A[20:40]

    return A1, A2


def One_D_to_TwoD(A):
    """
    Using a 1D array, returned by the function @actual_A_1D , making a lower triangular matrix A2D
    : param A : 1D array of length = 10
    : return : 2D array

    """
    A = A.reshape(4, 4)
    return A

"""
################  Hamiltonian Equations 
"""

@jit
def f(q, p, alpha_values):
    return p
@jit
def g(q, p, alpha_values):
    
    alpha_values = alpha_values.transpose()
    return jnp.add(jnp.add((-1 * alpha_values[0]) , (-2 * alpha_values[1] * q) ) , jnp.add((-3 * alpha_values[2]* (q**2)) , (-4 * alpha_values[3] * (q**3) )) )
@jit
def Energy_Function(q, p, alpha_values):
    alpha_values = alpha_values.transpose()
    # print("Inside Energy function",alpha_values.shape)
    p = jnp.array(p , dtype=jnp.float64)  # Ensure zn_list is converted to jnp.array
    q = jnp.array(q , dtype=jnp.float64)  # Ensure yn_list is converted to jnp.array
    alpha_values = jnp.array(alpha_values, dtype=jnp.float64)
    return ((jnp.square(p))/2 + jnp.add(jnp.add(( alpha_values[0]* (q)) , (alpha_values[1] * (q**2)) ) , jnp.add((alpha_values[2]* (q**3)) , (alpha_values[3] * (q**4)))  )) 

"""
"""

# @jit
# def f(y, z, alpha_values):
#     return z
# @jit
# def g(y, z, alpha_values):
#     alpha_values = alpha_values.transpose()
#     print("Inside g function",alpha_values.shape)
#     return jnp.add(jnp.add((-1 * alpha_values[0]) , (-2 * alpha_values[1] * y) ) , jnp.add((-3 * alpha_values[2]* (y**2)) , (-4 * alpha_values[3] * (y**3) )) )
# @jit
# def Energy_Function(y, z, alpha_values):
#     print("Inside Energy function",alpha_values.shape)
#     return ((jnp.square(y))/2 + jnp.add(jnp.add(( alpha_values[0]* (z)) , (alpha_values[1] * (z**2)) ) , jnp.add((alpha_values[2]* (z**3)) , (alpha_values[3] * (z**4) )) )) 

@jit
def PRK_step(y0 , z0, h, A1, A2, B1, B2, alpha_values):
    s = A1.shape[0]
    dim = jnp.size(y0)
    tol = jnp.float64(10**(-10))
    K_old = jnp.zeros((s,dim), dtype=jnp.float64)
    L_old = jnp.zeros((s,dim), dtype=jnp.float64)
    
    K_new = f((y0+ h*A1 @ K_old), (z0+ h*A2 @ L_old), alpha_values)
    L_new = g((y0+ h*A1 @ K_old), (z0+ h*A2 @ L_old), alpha_values)

    init_state = 0, K_new, L_new, K_old, L_old, alpha_values

    def body_while_loop(state):
        _, K_new, L_new, K_old, L_old, alpha_values = state
        K_old = K_new
        L_old = L_new
        K_new = f(y0+ h * A1 @ K_old, z0 + h * A2 @ L_old, alpha_values)
        L_new = g(y0+ h * A1 @ K_old, z0 + h * A2 @ L_old, alpha_values)
        
        return _, K_new, L_new, K_old, L_old, alpha_values

    def condition_while_loop(state):
        tol = jnp.float64(10**(-10))
        _, K_new, L_new, K_old, L_old, alpha_values = state
        norms = jnp.sum(jnp.array([jnp.linalg.norm(K_new - K_old) + jnp.linalg.norm(L_new - L_old)]))
    
        return norms > tol

    _, K_new, L_new, K_old, L_old, alpha_values = jax.lax.while_loop(condition_while_loop, body_while_loop, init_state)
    yn = y0 + h * jnp.sum(jnp.multiply(B1, K_new), dtype=jnp.float64)
    zn = z0 + h * jnp.sum(jnp.multiply(B2, L_new), dtype=jnp.float64)

    return yn, zn

def fori_loop_1(i, state):
    yn_list, zn_list, y, z, A1, A2, B1, B2, alpha_values, h, istep = state
    y, z = PRK_step(y, z, h, A1, A2, B1, B2, alpha_values)
    yn_list = yn_list.at[i].set(y.ravel())
    zn_list = zn_list.at[i].set(z.ravel())
    state = yn_list, zn_list, y, z, A1, A2, B1, B2, alpha_values, h, istep
    return state

def fori_loop_2(j, state):
    iyn_list, izn_list, iy, iz, A1, A2, B1, B2, alpha_values, h, istep = state
    iy, iz = PRK_step(iy, iz, h/istep, A1, A2, B1, B2, alpha_values)
    iyn_list = iyn_list.at[j].set(iy.ravel())
    izn_list = izn_list.at[j].set(iz.ravel())
    state = iyn_list, izn_list, iy, iz, A1, A2, B1, B2, alpha_values, h, istep
    return state

def huber_loss(x, delta=1.0):
    return jnp.where(jnp.abs(x) <= delta, 0.5 * jnp.square(x), delta * (jnp.abs(x) - 0.5 ))


def find_error(A1D, H_sequence):

    a1, a2 = actual_A1_A2(A1D) #, H_sequence
    a1,B1 = actual_A_1D(a1)
    A1 = One_D_to_TwoD(a1)
    a2,B2 = actual_A_1D(a2)
    A2 = One_D_to_TwoD(a2)

    B1 = jnp.reshape(B1, (4, 1))
    B2 = jnp.reshape(B2, (4, 1))

    alpha_values = jnp.reshape(jnp.array(H_sequence[:4], dtype=jnp.float64), (1, 4))
    # alpha_values = jnp.reshape(jnp.array(H_sequence[:4]), (1, 4))
    # print("Inside Find Error function :", alpha_values.shape)
    time_factor = 1 # default

    y0 = jnp.reshape(jnp.array(H_sequence[4], dtype=jnp.float64), (1, 1)) # jnp.zeros((1,1)) # p
    z0 = jnp.reshape(jnp.array(H_sequence[5], dtype=jnp.float64), (1, 1)) # jnp.ones((1,1)) # q
    # y0 = jnp.reshape(jnp.array(H_sequence[4]), (1, 1)) # jnp.zeros((1,1)) # p
    # z0 = jnp.reshape(jnp.array(H_sequence[5]), (1, 1)) # jnp.ones((1,1)) # q

    LA1 = jnp.array([
         [0., 0., 0., 0.],
         [5/24, 1/3, -1/24, 0.],
         [1/6, 2/3, 1/6, 0.],
         [0., 0., 0., 0.]])
    LB1 = jnp.array([1/6, 2/3, 1/6, 0.])
    LB1 = LB1.reshape(4, 1)
    
    LA2 = jnp.array([
         [1/6, -1/6, 0., 0.],
         [1/6, 1/3, 0, 0.],
         [1/6, 5/6, 0, 0.],
         [0., 0., 0., 0.]])
    LB2 = jnp.array([1/6, 2/3, 1/6, 0.])
    LB2 = LB2.reshape(4, 1)

    isteps = 100
    NN = 1000
    ## This is for the step size. h denotes the steps size
    ## istep is the smaller steps. in our case istep = 10.

    i = 0
    total_steps = int(time_factor * NN)
    total_isteps = int(time_factor * isteps * NN)

    yn_list = jnp.zeros((total_steps, 1))
    zn_list = jnp.zeros((total_steps, 1))
    iyn_list = jnp.zeros((total_isteps , 1))
    izn_list = jnp.zeros((total_isteps , 1))

    # yn = zn = iyn = izn = []
    h = 1/NN #step size, should be not dependent on time_steps. as i want longer simulations with same time step.
    y = iy = y0
    z = iz = z0
    
    # remember making fori loop inside this function makes it slow. Dont know why though.
    init_state_yz = yn_list, zn_list, y, z, A1, A2, B1, B2, alpha_values, h, isteps
    yn_list, zn_list, _, _, _, _, _, _, _, _, _ = jax.lax.fori_loop(0, total_steps, fori_loop_1, init_state_yz)
    
    H = Energy_Function(yn_list, zn_list, alpha_values) # H should be of type list
    
    # print("############################# Length of H = ==================" , len(H))
    energy_error = jnp.sum(jnp.square(H - H[0])) / len(H) # Hamiltonian Error

    init_state_iyz = iyn_list, izn_list, iy, iz, LA1, LA2, LB1, LB2, alpha_values, h, isteps ## LA1, LA2, LB1, LB2,
    iyn_list, izn_list, _, _, _, _, _, _, _, _, _ = jax.lax.fori_loop(0, total_isteps, fori_loop_2, init_state_iyz) # time istep

    j1_iyn_list = iyn_list[9:total_isteps:isteps]
    j2_izn_list = izn_list[9:total_isteps:isteps]

    err1 = j1_iyn_list.ravel() - yn_list.ravel()
    err2 = j2_izn_list.ravel() - zn_list.ravel()

    ## Squared Loss
    # Now apply the squared error on the differences
    squared_err1 = jnp.square(err1)
    squared_err2 = jnp.square(err2)

    ### Squared loss and Absolute error 
    # final_error = (jnp.sum(squared_err1) + jnp.sum(squared_err2)) / (2 * NN)
    # final_error = (jnp.sum(jnp.abs(err1)) + jnp.sum(jnp.abs(err2))) / (2*NN)
    
    ## Huber Loss : check mail by christian regarding using this error 
    final_error = (jnp.sum(huber_loss(err1)) + jnp.sum(huber_loss(err2))) / (2*NN)

    
    return jnp.add(jnp.sum(final_error), energy_error), energy_error  #, step_size_list_convergence, o_error_list_convergence
