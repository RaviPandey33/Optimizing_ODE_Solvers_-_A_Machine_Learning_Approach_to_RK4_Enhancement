import jax
from jax import config
config.update("jax_enable_x64", True)  #double precision

import jax.numpy as jnp
import json
import os
import sys
from skopt.space import Space
from skopt.sampler import Halton
from jax import jacfwd

# Special Transform Functions
from jax import grad, jit, vmap, pmap
import jax
from jax import jit
import numpy as np

from jax._src.lax.utils import (
    _argnum_weak_type,
    _input_dtype,
    standard_primitive,)
from jax._src.lax import lax

from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import Test_prk_for_optimization as IRK4
import Important_functions.Transformation_Functions as TFunctions
import Important_functions.Convert_1D2D as convert
import Important_functions.Energy_Error as EE


"""
 : using the initial matrix A and B from the source given below :
 : wiki link : https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#:~:text=is%5B13%5D-,0,1/6,-A%20slight%20variation
"""

"""
############################### Original ##########################################
"""
## Lobatto 3A and B fourth order

A1 = A2 = jnp.array([
     [0.1, 0., 0., 0.],
     [5/24, 2/3, -1/24, 1.],
     [2/6, 2/3, 1/6, 0.],
     [0., 1., 0., 0.]])
B1 = B2 = jnp.array([1/6, 2/3, 1/6, 0.])



## Making the Halton code

spacedim = [(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5) ]

space = Space(spacedim)

halton = Halton()
n = 150

halton_sequence = halton.generate(space, n)
halton_sequence = jnp.array(halton_sequence)

## Dividing in training and validation set. 100 for the training set and 50 for the validation set. 
validation_halton = halton_sequence[100:150]
halton_sequence = halton_sequence[:100]
# print(halton_sequence)

# print(len(halton_sequence))
# print(len(validaton_halton))


# Initial A1D
import optax

A1D = convert.Convert_toOneD(A1, A2, B1, B2)
print(A1D.shape)
learning_rate = 0.0001
list_optimizers = [optax.adam(learning_rate)]
# chosing Stochastic Gradient Descent Algorithm.
# # We have created a list here keeping in mind that we may apply all the optimizers in optax by storing their objects in the list
 
opt_sgd = list_optimizers[0]
opt_state = opt_sgd.init(A1D)

params = A1D

count = 0
data_epoc = 10
data_epoc_list = []
repetetion = 10
# length of halton sequence = 10 

tot_eror = 0
error_list_1 = [] 
error_list_2 = []
error_list_3 = [] # For Energy Error 
error_list_4 = [] # For Error 

flat_halton_sequence = jnp.array(halton_sequence).reshape(-1, 6)

# Compute gradient using jacfwd
def compute_grads_single(A1D, h_element):
    grad_fn = jax.jacfwd(IRK4.find_error)
    return grad_fn(A1D, h_element)

def compute_error_single(A1D, h_element):
    return IRK4.find_error(A1D, h_element)

        
## Finding the numeric gradient. Jacfwd is not showing good results. 
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

error_list_1 = [] 
error_list_2 = []
error_list_3 = [] # Energy Error :
error_list_4 = [] # Error :
validation_error_list = []
tot_error = 0
tot_error_energy = 0
total_error_e = 0
validation_tot_error = 0
validation_avg_error = 0

with open('S2_Error(Perturbed_low).txt', 'w') as S1_output, open('S2_Final_weights(Perturbed_low).txt', 'w') as S1_weights:
    ## Batch Size
    batch_size = 100 ## just to remind you right now total halton sequence is also 100, so we are taking the whole set as the batch.
    validation_batch_size = 50
    
    for k in trange(1000000):
        
        tot_error = 0
        validation_tot_error = 0
        for batch_idx in range(0, len(flat_halton_sequence), batch_size):

            # Collect a batch of elements from the flattened Halton sequence
            batch_halton = flat_halton_sequence[batch_idx:batch_idx + batch_size]

            # Compute the gradients for the batch using jax.vmap
            gradF = compute_grads_batched(A1D, batch_halton)

            # Compute the average gradient for the batch
            avg_gradF = jnp.mean(gradF, axis=0)

            # Perform one step of optimization using the averaged gradient
            updates, opt_state = opt_sgd.update(avg_gradF, opt_state)

            # Apply the updates to the weights A1D for the entire batch
            A1D = optax.apply_updates(A1D, updates)

            # Calculate the total error for the batch and accumulate it
            batch_error = jnp.mean(compute_error_batched(A1D, batch_halton))
            tot_error += batch_error

        np_array_A1D = np.array(A1D)
        np_array_A1D_string = ' '.join(map(str, np_array_A1D))
        S1_weights.seek(0)  # Move to the beginning of the file
        S1_weights.write(np_array_A1D_string) #+ '\n')
        # S1_weights.flush()  # Ensure it writes to disk
        # S1_weights.close()
        
        """
        # what am i tying to do here ?
        """
        avg_error = tot_error / (len(flat_halton_sequence) // batch_size) # ?
        error_list_1.append(avg_error) # ?
        
        # Write the errors to the respective files
        S1_output.write(f"{avg_error}\n")
        S1_output.flush()
    