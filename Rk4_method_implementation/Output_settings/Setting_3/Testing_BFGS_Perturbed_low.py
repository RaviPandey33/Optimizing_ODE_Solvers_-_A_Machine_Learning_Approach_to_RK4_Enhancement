import jax
from jax import config
config.update("jax_enable_x64", True)  # Enable double precision

import numpy as np
import jax.numpy as jnp
import json
import os
import sys
from skopt.space import Space
from skopt.sampler import Halton
from jax import jacfwd
from scipy.optimize import minimize
from tqdm import trange
import matplotlib.pyplot as plt

import Test_prk_for_optimization as IRK4
import Important_functions.Transformation_Functions as TFunctions
import Important_functions.Convert_1D2D as convert
import Important_functions.Energy_Error as EE

# Define the A and B matrices
A1 = jnp.array([
     [0.5, 1.0, 1/5, 0.],
     [5/24, 1/3, -1/24, 0.],
     [5/6, 1/3, 1/6, 0.],
     [0., 0., 1/3, 0.5]])
B1 = jnp.array([1/6, 2/3, 1/6, 0.])
## Lobatto 3B
A2 = jnp.array([
     [1/6, -1/6, 0., 0.],
     [1/6, 1/3, 0., 0.],
     [1/6, 5/6, 0., 0.],
     [0., 0., 0., 0.]])
B2 = jnp.array([1/6, 2/3, 1/6, 0.])

# Making the Halton code
spacedim = [(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5)]
space = Space(spacedim)
halton = Halton()
n = 150
halton_sequence = halton.generate(space, n)
halton_sequence = jnp.array(halton_sequence)

# Dividing in training and validation set. 100 for the training set and 50 for the validation set. 
validation_halton = halton_sequence[100:150]
halton_sequence = halton_sequence[:100]

# Initial A1D
A1D = convert.Convert_toOneD(A1, A2, B1, B2)

# Define the objective function for optimization
def objective_function(A1D, *args):
    halton_sequence = args[0]
    total_error = 0
    for h_element in halton_sequence:
        total_error += IRK4.find_error(A1D, h_element)[0]
    return total_error / len(halton_sequence)

# Store errors
error_list_1 = []  # Total error calculation via batch
error_list_2 = []  # Total error calculation when calculating energy error
error_list_3 = []  # Energy Error :
error_list_4 = []  # Validation Error :
validation_error_list = []

with open('Outputs/S3_Error(Perturbed_low).txt', 'w') as S3_output, open('Outputs/S3_Energy_Error(Perturbed_low).txt', 'w') as S3_energy_error, open('Outputs/S3_Validation(Perturbed_low).txt', 'w') as S3_validation, open('Outputs/S3_Final_weights(Perturbed_low).txt', 'w') as S3_weights :
    for epoch in trange(1):
        # Perform one iteration of BFGS
        result = minimize(objective_function, A1D, args=(halton_sequence,), method='L-BFGS-B', options={'maxiter': 1})
        
        # Updating A1D with the result
        A1D = result.x
        
        # Calculating and storing errors
        tot_error = 0
        tot_error_energy = 0
        total_error_e = 0
        validation_tot_error = 0
        
        for h_element in halton_sequence:
            error, energy_error = IRK4.find_error(A1D, h_element)
            tot_error += error
            tot_error_energy += energy_error
        
        avg_error = tot_error / len(halton_sequence)
        avg_error_energy = tot_error_energy / len(halton_sequence)
        
        # error_list_1.append(avg_error)
        # error_list_3.append(avg_error_energy)
        
        # Validation error
        for v_element in validation_halton:
            validation_tot_error += IRK4.find_error(A1D, v_element)[0]
        
        validation_avg_error = validation_tot_error / len(validation_halton)
        # validation_error_list.append(validation_avg_error)

        np_array_A1D = np.array(A1D)
        np_array_A1D_string = ' '.join(map(str, np_array_A1D))
        S3_weights.seek(0)  # Move to the beginning of the file
        S3_weights.write(np_array_A1D_string) #+ '\n')
        
        S3_output.write(f"{avg_error}\n")
        S3_output.flush()
        S3_energy_error.write(f"{avg_error_energy}\n")
        S3_energy_error.flush()
        S3_validation.write(f"{validation_avg_error}\n")
        S3_validation.flush()
        
        """ 
        S3_output : is for the analitical error 
        S3_energy_error : for the energy error
        S3_validation : the validation set error
        S3_weights : the updated weights of A1D
        """
