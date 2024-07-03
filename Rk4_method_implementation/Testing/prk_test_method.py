import jax
import jax.numpy as jnp
from jax import jit, lax
import numpy as np

@jit
def One_Dim_Matrix(A):
    dim_x, dim_y = A.shape
    A = A.reshape(1, (dim_x * dim_y))
    return A

@jit
def Add_B_tomatrix_A(A, b):
    A = jnp.append(A, b)
    return A

@jit
def actual_A_1D(A):
    b1 = A[16:20]
    A_new = A[0:16]
    return A_new, b1

@jit
def actual_A1_A2(A):
    A1 = A[0:20]
    A2 = A[20:40]
    return A1, A2

@jit
def One_D_to_TwoD(A):
    A = A.reshape(4, 4)
    return A

@jit
def f(y, z, alpha_values):
    return z

@jit
def g(y, z, alpha_values):
    return -y

@jit
def Energy_Function(y, z, alpha_values):
    return (jnp.square(y) + jnp.square(z)) / 2

@jit
def PRK_step(y0, z0, h, A1, A2, B1, B2, alpha_values):
    s = A1.shape[0]
    dim = jnp.size(y0)
    tol = 1e-10
    K_old = jnp.zeros((s, dim))
    L_old = jnp.zeros((s, dim))
    K_new = f(y0 + h * A1 @ K_old, z0 + h * A2 @ L_old, alpha_values)
    L_new = g(y0 + h * A1 @ K_old, z0 + h * A2 @ L_old, alpha_values)
    init_state = 0, K_new, L_new, K_old, L_old, alpha_values

    def body_while_loop(state):
        _, K_new, L_new, K_old, L_old, alpha_values = state
        K_old = K_new
        L_old = L_new
        K_new = f(y0 + h * A1 @ K_old, z0 + h * A2 @ L_old, alpha_values)
        L_new = g(y0 + h * A1 @ K_old, z0 + h * A2 @ L_old, alpha_values)
        return _, K_new, L_new, K_old, L_old, alpha_values

    def condition_while_loop(state):
        _, K_new, L_new, K_old, L_old, alpha_values = state
        norms = jnp.sum(jnp.array([jnp.linalg.norm(K_new - K_old) + jnp.linalg.norm(L_new - L_old)]))
        return norms > tol

    _, K_new, L_new, K_old, L_old, alpha_values = lax.while_loop(condition_while_loop, body_while_loop, init_state)
    yn = y0 + h * jnp.sum(jnp.multiply(B1, K_new))
    zn = z0 + h * jnp.sum(jnp.multiply(B2, L_new))
    return yn, zn

@jit
def fori_loop_1(i, state):
    yn_list, zn_list, y, z, A1, A2, B1, B2, alpha_values, h, istep = state
    y, z = PRK_step(y, z, h, A1, A2, B1, B2, alpha_values)
    yn_list = yn_list.at[i].set(y.ravel())
    zn_list = zn_list.at[i].set(z.ravel())
    return yn_list, zn_list, y, z, A1, A2, B1, B2, alpha_values, h, istep

@jit
def fori_loop_2(j, state):
    iyn_list, izn_list, iy, iz, A1, A2, B1, B2, alpha_values, h, istep = state
    iy, iz = PRK_step(iy, iz, h/istep, A1, A2, B1, B2, alpha_values)
    iyn_list = iyn_list.at[j].set(iy.ravel())
    izn_list = izn_list.at[j].set(iz.ravel())
    return iyn_list, izn_list, iy, iz, A1, A2, B1, B2, alpha_values, h, istep

def huber_loss(x, delta=1.0):
    return jnp.where(jnp.abs(x) <= delta, 0.5 * jnp.square(x), delta * (jnp.abs(x) - 0.5 * delta))


def find_error_test(A1D):
    """remove all these lines

     ##  definig to check the grandient
    
    """
    # print("The name is Pandey, Ravi Pandey")
    H_sequence = [-1, -1, -1, -1, -1, -1]

    time_factor = 1
    a1, a2 = actual_A1_A2(A1D)
    a1, B1 = actual_A_1D(a1)
    A1 = One_D_to_TwoD(a1)
    a2, B2 = actual_A_1D(a2)
    A2 = One_D_to_TwoD(a2)
    
    final_error = 0 ##  definig to check the grandien

    B1 = jnp.reshape(B1, (4, 1))
    B2 = jnp.reshape(B2, (4, 1))

    alpha_values = jnp.reshape(jnp.array(H_sequence[:4]), (1, 4))

    y0 = jnp.reshape(jnp.array(H_sequence[4]), (1, 1))
    z0 = jnp.reshape(jnp.array(H_sequence[5]), (1, 1))

    istep = 100
    NN = jnp.array([100])

    yn_list = jnp.zeros((time_factor * NN[0], 1))
    zn_list = jnp.zeros((time_factor * NN[0], 1))
    iyn_list = jnp.zeros((time_factor * istep * NN[0], 1))
    izn_list = jnp.zeros((time_factor * istep * NN[0], 1))

    h = time_factor / NN[0]
    y = y0
    z = z0
    
    init_state_yz = yn_list, zn_list, y, z, A1, A2, B1, B2, alpha_values, h, istep
    yn_list, zn_list, _, _, _, _, _, _, _, _, _ = lax.fori_loop(0, time_factor * NN[0], fori_loop_1, init_state_yz)

    
    H = Energy_Function(yn_list, zn_list, alpha_values)
    energy_error = jnp.sum(jnp.square(H - H[0])) / len(H)

    init_state_iyz = iyn_list, izn_list, y, z, A1, A2, B1, B2, alpha_values, h, istep
    iyn_list, izn_list, _, _, _, _, _, _, _, _, _ = lax.fori_loop(0, time_factor * istep * NN[0], fori_loop_2, init_state_iyz)
    
    ## Gradient using jacfwd works till here correctly 

    j1_iyn_list = iyn_list[9:istep * NN[0]:istep] ## earlier : instead of istep, i had written 10
    j2_izn_list = izn_list[9:istep * NN[0]:istep] ## "" same as above
    

    err1 = j1_iyn_list.ravel() - yn_list.ravel()
    err2 = j2_izn_list.ravel() - zn_list.ravel()
    
    
    final_error = (jnp.sum(jnp.abs(err1)) + jnp.sum(jnp.abs(err2))) / (2 * NN[0])
    
    # final_error = (jnp.sum(huber_loss(err1)) + jnp.sum(huber_loss(err2))) / (2 * NN[0])

    return jnp.sum(final_error)
    

    return jnp.sum(final_error) # + energy_error
