import jax
from jax import config
config.update("jax_enable_x64", True)  #double precision
import jax.numpy as jnp
import numpy as np
import os
import sys
from skopt.space import Space
from skopt.sampler import Halton
from jax import jacfwd
from jax import grad, jit, vmap, pmap
from jax import jit
from jax._src.lax.utils import (
    _argnum_weak_type,
    _input_dtype,
    standard_primitive,)
from jax._src.lax import lax

from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import optax

import prk_method.Test_prk_for_optimization as IRK4
import Important_functions.Convert_1D2D as convert

# Initial weights 
## Lobatto 3A and B fourth order
A1 = jnp.array([
     [0.5, 0., 0., 0.],
     [5/24, 2/3, -3/24, 0.],
     [1/6, 2/3, 1/6, 0.5],
     [0., 0., 0.5, 0.]])
B1 = jnp.array([2/6, 1/3, 1/6, 0.])

A2 = jnp.array([
     [1/6, -1/6, 0., 0.],
     [1/6, 1/3, 0, 0.],
     [1/6, 5/6, 0, 0.],
     [0., 0., 0., 0.]])
B2 = jnp.array([1/6, 2/3, 1/6, 0.])

## Making the Halton code
spacedim = [(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5) ]
space = Space(spacedim)
halton = Halton()
n = 150 # total halton elements

halton_sequence = halton.generate(space, n)
halton_sequence = jnp.array(halton_sequence)

## Dividing in training and validation set. 100 for the training set and 50 for the validation set. 
validation_halton = halton_sequence[100:150]
halton_sequence = halton_sequence[:100]

# Initial A1D, i.e. initial weights in 1D
A1D = convert.Convert_toOneD(A1, A2, B1, B2)

learning_rate = 0.0001
list_optimizers = [optax.adam(learning_rate)]
 
opt_sgd = list_optimizers[0]
opt_state = opt_sgd.init(A1D)
params = A1D

count = 0
data_epoc = 10
data_epoc_list = []
repetetion = 10

flat_halton_sequence = jnp.array(halton_sequence).reshape(-1, 6)

# Compute gradient using jacfwd
def compute_grads_single(A1D, h_element):
    grad_fn = jax.jacfwd(IRK4.find_error)
    return grad_fn(A1D, h_element)

def compute_energy_error(A1D, h_element):
    # print(type(IRK4.find_error(A1D, h_element)[0]))
    return IRK4.find_error(A1D, h_element)[1]

def compute_error_single(A1D, h_element):
    # print(type(IRK4.find_error(A1D, h_element)[0]))
    return IRK4.find_error(A1D, h_element)[0]

## Finding the numeric gradient. Finite difference approximation 
def numerical_gradient(A1D, halton_element, epsilon=1e-5):
    numerical_gradients = jnp.zeros_like(A1D)
    for i in range(len(A1D)):
        A1D_plus = A1D.at[i].set(A1D[i] + epsilon)
        A1D_minus = A1D.at[i].set(A1D[i] - epsilon)
        numerical_gradients = numerical_gradients.at[i].set(
            (compute_error_single(A1D_plus, halton_element) - compute_error_single(A1D_minus, halton_element)) / (2 * epsilon)
        )
    return numerical_gradients

# Use jax.vmap to vectorize the function over the batch
compute_grads_batched = jax.vmap(numerical_gradient, in_axes=(None, 0)) # using the numerical gradient instead of jacfwd()
compute_error_batched = jax.vmap(compute_error_single, in_axes=(None, 0))

Total_Error_List = [] # Analytical Error :
Energy_Error_List = [] # Energy Error :
Validation_Error_List = [] # Validation Set Error :

with open('Without_Energy/S1_Outputs/Perturbed_Error.txt', 'w') as S1_output, open('Without_Energy/S1_Outputs/Perturbed_Energy.txt', 'w') as S1_energy, open('Without_Energy/S1_Outputs/Perturbed_Validation.txt', 'w') as S1_validation, open('Without_Energy/S1_Outputs/Perturbed_Final_weights.txt', 'w') as S1_weights:
    ## Batch Size
    batch_size = 100 ## just to remind you right now total halton sequence is also 100, so we are taking the whole set as the batch.
    validation_batch_size = 50
    for k in trange(300000):
        tot_error = 0
        tot_error_energy = 0
        total_error_e = 0
        validation_tot_error = 0
        validation_avg_error = 0
        tot_error = 0
        validation_tot_error = 0
        for batch_idx in range(0, len(flat_halton_sequence), batch_size):
            batch_halton = flat_halton_sequence[batch_idx:batch_idx + batch_size]
            # Compute the gradients for the batch using jax.vmap
            gradF = compute_grads_batched(A1D, batch_halton)
            # Compute the average gradient for the batch
            avg_gradF = jnp.mean(gradF, axis=0)
            # Optimization step, of batch
            updates, opt_state = opt_sgd.update(avg_gradF, opt_state)
            # Apply the updates to A1D
            A1D = optax.apply_updates(A1D, updates)
            # Calculate the total error for the batch and accumulate it
            batch_error = jnp.mean(compute_error_batched(A1D, batch_halton))
            tot_error += batch_error
        
        # Writing A1D to the required file
        np_array_A1D = np.array(A1D)
        np_array_A1D_string = ' '.join(map(str, np_array_A1D))
        S1_weights.seek(0)  # Move to the beginning of the file
        S1_weights.write(np_array_A1D_string) #+ '\n')

        avg_error = tot_error / (len(flat_halton_sequence) // batch_size) # ? 
        # Write the errors to the respective files
        S1_output.write(f"{avg_error}\n")
        S1_output.flush()
        
        ## Calculating the energy Error :
        for e in range(len(halton_sequence)):
            energy_e = IRK4.find_error(A1D, halton_sequence[e])
            tot_error_energy += energy_e[1]
        avg_error_energy = tot_error_energy / len(halton_sequence)
        S1_energy.write(f"{avg_error_energy}\n")
        S1_energy.flush()  
        
        ## Calculating the validation set error :   
        for v in range(0,len(validation_halton)):
            validation_tot_error += IRK4.find_error(A1D, validation_halton[v])[0]
        
        validation_avg_error = validation_tot_error / len(validation_halton)
        S1_validation.write(f"{validation_avg_error}\n")
        S1_validation.flush()

