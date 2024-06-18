from jax.config import config
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

import Transformation_Functions as TFunctions
import Functions_toSolve

@jit
def PRK_step(y0 , z0, h, A1, A2, B1, B2, alpha_values):
    s = A1.shape[0]
    dim = jnp.size(y0)
    tol = 10**(-10)
    K_old = jnp.zeros((s,dim))
    L_old = jnp.zeros((s,dim))
    # print("+++++++++++++++++++++ Shape of K_old = ", K_old.shape)
    K_new = Functions_toSolve.f((y0+ h*A1 @ K_old), (z0+ h*A2 @ L_old), alpha_values)
    L_new = Functions_toSolve.g((y0+ h*A1 @ K_old), (z0+ h*A2 @ L_old), alpha_values)

    # print("shape of yn and zn :",y0.shape, y0.shape)
    init_state = 0, K_new, L_new, K_old, L_old, alpha_values

    def body_while_loop(state):
        _, K_new, L_new, K_old, L_old, alpha_values = state
        K_old = K_new
        L_old = L_new
        K_new = Functions_toSolve.f(y0+ h * A1 @ K_old, z0 + h * A2 @ L_old, alpha_values)
        L_new = Functions_toSolve.g(y0+ h * A1 @ K_old, z0 + h * A2 @ L_old, alpha_values)
        return _, K_new, L_new, K_old, L_old, alpha_values

    def condition_while_loop(state):
        _, K_new, L_new, K_old, L_old, alpha_values = state
        norms = jnp.sum(jnp.array([jnp.linalg.norm(K_new - K_old) + jnp.linalg.norm(L_new - L_old)]))
        return norms > tol

    _, K_new, L_new, K_old, L_old, alpha_values = jax.lax.while_loop(condition_while_loop, body_while_loop, init_state)
    yn = y0 + h * jnp.sum(jnp.multiply(B1, K_new))
    zn = z0 + h * jnp.sum(jnp.multiply(B2, L_new))

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

def find_error(A1D, H_sequence, step = 10):
    #converting A1D back to the original matrix form

    time_factor = 1
    a1, a2 = TFunctions.PRKactual_A1_A2(A1D) #, H_sequence
    a1,B1 = TFunctions.actual_A_1D(a1)
    A1 = TFunctions.One_D_to_TwoD(a1)
    a2,B2 = TFunctions.actual_A_1D(a2)
    A2 = TFunctions.One_D_to_TwoD(a2)

    B1 = jnp.reshape(B1, (4, 1))
    B2 = jnp.reshape(B2, (4, 1))

    alpha_values = jnp.reshape(jnp.array(H_sequence[:4]), (1, 4))

    time_factor = 1 # default

    y0 = jnp.reshape(jnp.array(H_sequence[4]), (1, 1)) # jnp.zeros((1,1)) #
    z0 = jnp.reshape(jnp.array(H_sequence[5]), (1, 1)) # jnp.ones((1,1)) #

    istep = 10
    
    yn_list = jnp.zeros((time_factor * step, 1))
    zn_list = jnp.zeros((time_factor * step, 1))
    iyn_list = jnp.zeros((time_factor * istep *step , 1))
    izn_list = jnp.zeros((time_factor * istep * step , 1))

    yn = zn = iyn = izn = []
    h = time_factor/step #step size
    y = iy = y0
    z = iz = z0
    
    # remember making fori loop inside this function makes it slow. Dont know why though.
    init_state_yz = yn_list, zn_list, y, z, A1, A2, B1, B2, alpha_values, h, istep
    yn_list, zn_list, _, _, _, _, _, _, _, _, _ = jax.lax.fori_loop(0, time_factor * step, fori_loop_1, init_state_yz)
    
    H = Functions_toSolve.Energy_Function(yn_list, zn_list, alpha_values) # H should be of type list
    energy_error = jnp.sum(jnp.square(H - H[0])) / len(H) # Hamiltonian Error

    init_state_iyz = iyn_list, izn_list, iy, iz, A1, A2, B1, B2, alpha_values, h, istep
    iyn_list, izn_list, _, _, _, _, _, _, _, _, _ = jax.lax.fori_loop(0, time_factor * istep * step, fori_loop_2, init_state_iyz) # time istep
    j1_iyn_list = iyn_list[9:istep*step:10]
    j2_izn_list = izn_list[9:istep*step:10]

    err1 = j1_iyn_list.ravel() - yn_list.ravel()
    err2 = j2_izn_list.ravel() - zn_list.ravel()

    final_error = (jnp.sum(jnp.abs(err1)) + jnp.sum(jnp.abs(err2))) / (2*step)

    return jnp.sum(final_error) + energy_error #, step_size_list_convergence, o_error_list_convergence
