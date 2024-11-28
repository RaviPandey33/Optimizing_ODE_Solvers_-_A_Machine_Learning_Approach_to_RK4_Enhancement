import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Ensure JAX uses CPU backend
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=100"

import jax
num_devices = jax.local_device_count()
print(jax.devices(), "num devices = ", num_devices)

from jax import config
config.update("jax_enable_x64", True)  #double precision

import multiprocessing
import time  # For timing the execution

import cProfile ## Checking for parallel computing
import pstats ## For analysing the program run


import jax.numpy as jnp
import numpy as np
import os
import sys
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

import withoutE_prk_method.One_Batch_prk_for_optimization as IRK4
import withoutE_prk_method.Valid_Halton as Find_valid_halton
import Important_functions.Convert_1D2D as convert

    
def main():
    output_path = 'Without_Energy/One-batch-PerturbedLobatto-Outputs/'
    # Different Files output path
    Error_path = f'{output_path}Error.txt'
    Energy_path = f'{output_path}Energy.txt'
    Validation_path = f'{output_path}Validation.txt'
    FinalWeight_path = f'{output_path}Final_weights.txt'
    
    # Initial weights 
    ## Perturbed Lobatto 3A and B fourth order
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
    
    """
    ######################################
    """
    ## Making the Halton code and diving it into validation and training set. 
    
    spacedim = [(-0.5, 0.5)]*6
        
    space = Space(spacedim)
    halton = Halton()
    n = 1000  # total halton elements
    
    halton_sequence = halton.generate(space, n)
    halton_sequence = jnp.array(halton_sequence)
    
    """
    ######################################
    """

    """
    ######## Finding correct Halton Elements that dont return infinite solutions ########
    """

    def is_valid_halton_element(halton_element, threshold=1):
        """
        Check if the Halton element is valid based on certain criteria.
        Args:
            halton_element: The current Halton element being tested.
            threshold: Maximum allowed magnitude for any element in the generated output.
        Returns:
            Boolean indicating if the Halton element is valid.
        """
        # Simulate the output using the Halton element
        yn_list, zn_list, _ = Find_valid_halton.find_error(A1D_original_BeforeOptimization, halton_element)
    
        # Check if any values in yn_list or zn_list exceed the threshold (indicating divergence or large values)
        if jnp.any(jnp.abs(yn_list) > threshold) or jnp.any(jnp.abs(zn_list) > threshold):
            return False
        return True
    
    # Initialize an empty list to store valid Halton elements and their indices
    valid_halton_sequence = []
    valid_indices = []
    
    print(f"Total Halton elements to process: {len(halton_sequence)}")
    # halton_sequence = halton_sequence[11:15]
    
    
    for i, halton_element in enumerate(halton_sequence):
        # Print progress every 10 elements
        # if i % 10 == 0:
        #     print(f"Processing element {i + 1}/{len(halton_sequence)}")
        
        if is_valid_halton_element(halton_element, threshold=10):  # Set an appropriate threshold
            valid_halton_sequence.append(halton_element)
            valid_indices.append(i)
    
    # Convert the filtered list to a JAX array
    valid_halton_sequence = jnp.array(valid_halton_sequence)
    
    # Print or save the valid indices for reference
    print("Valid Halton Element Indices:", valid_indices)
    print("Number of Valid Halton Elements:", len(valid_indices))

        
    # Split into training and validation sets
    validation_halton = valid_halton_sequence[100:150] ## Validation Data
    halton_sequence = valid_halton_sequence[:100] ## Training Data
    
    """
    ######################################
    """


    """
    ######################################
    """
    
    # Initial A1D, i.e., initial weights in 1D
    A1D = convert.Convert_toOneD(A1, A2, B1, B2)
    
    learning_rate = 0.001
    adam_optimizer = optax.adam(learning_rate)
    sgd_optimizer = optax.sgd(learning_rate)  # Adding SGD optimizer
    
    # Start with Adam optimizer
    opt_state = adam_optimizer.init(A1D)
    opt_switch_threshold = 1e-6  # Threshold to switch from Adam to SGD

    ######
    
    count = 0
    data_epoc = 10
    data_epoc_list = []
    repetetion = 10
    
    flat_halton_sequence = jnp.array(halton_sequence).reshape(-1, 6)
    
    # Compute gradient using jacfwd
    def jacfwd_gradient(A1D, h_element):
        def loss_fn(A1D):
            error, _ = IRK4.find_error(A1D, h_element)
            return error
        grad_fn = jax.jacfwd(loss_fn)
        return grad_fn(A1D)
    
    def compute_energy_error(A1D, h_element):
        return IRK4.find_error(A1D, h_element)[1]
    
    def compute_error_single(A1D, h_element):
        return IRK4.find_error(A1D, h_element)[0]
    
    ## Finding the numeric gradient. Finite difference approximation 
    def numerical_gradient(A1D, halton_element, epsilon=1e-5):
        numerical_gradients = jnp.zeros_like(A1D, dtype=jnp.float64)
        for i in range(len(A1D)):
            A1D_plus = A1D.at[i].set(A1D[i] + epsilon)
            A1D_minus = A1D.at[i].set(A1D[i] - epsilon)
            numerical_gradients = numerical_gradients.at[i].set(
                (compute_error_single(A1D_plus, halton_element) - compute_error_single(A1D_minus, halton_element)) / (2 * epsilon)
            )
        return numerical_gradients

    """
    ########################################
    Functions for paralalizing the errors to be saved. If this dosent work delete it. 
    """
    # Define the function for energy error and validation error
    def energy_error_fn(A1D, halton_element):
        return IRK4.find_error(A1D, halton_element)[1]  # Return only energy error
    
    def validation_error_fn(A1D, halton_element):
        return IRK4.find_error(A1D, halton_element)[0]  # Return only validation error

    def analytical_error_fn(A1D, halton_element):
        return IRK4.find_error(A1D, halton_element)[0]  # Return only Analytical error

    """
    ########################################
    """
    
    # Use jax.pmap to parallelize the function over multiple devices (e.g., CPUs)
    compute_grads_batched = jax.pmap(jacfwd_gradient, in_axes=(None, 0), axis_name='batch')  # pmap instead of vmap
    compute_error_batched = jax.pmap(compute_error_single, in_axes=(None, 0), axis_name='batch')  # pmap instead of vmap
    
    Total_Error_List = []  # Analytical Error :
    Energy_Error_List = []  # Energy Error :
    Validation_Error_List = []  # Validation Set Error :
    
    batch_size = 10  # Set the batch size to 10
    num_batches = len(flat_halton_sequence) // batch_size

    print(" Number of Batches : ",num_batches)


    #### Open the files in write mode ('w') at the beginning to clear old data
    with open(Error_path, 'w') as f1:
        f1.write("")
    with open(Energy_path, 'w') as f2:
        f2.write("")
    with open(Validation_path, 'w') as f3:
        f3.write("")
    with open(FinalWeight_path, 'w') as f4:
        f4.write("")
        
    # Lists to accumulate data for every 100 steps

    analytical_errors_list = []
    energy_errors_list = []
    validation_errors_list = []
    switch = False # Swtiching from ADAM to SGD

    
    min_error = jnp.float64(float('inf')) 
    min_error_A1D = None
    adam_max_iters = 120000
    
    for k in trange(1000000):
        tot_error = 0
        tot_error_energy = 0
        validation_tot_error = 0
        
        if k > 1:
            for batch_idx in range(num_batches):
                batch_halton = flat_halton_sequence[batch_idx * batch_size: (batch_idx + 1) * batch_size]            
                # Compute the gradients for the batch using jax.pmap
                gradF = compute_grads_batched(A1D, batch_halton)
                # Compute the average gradient for the batch
                avg_gradF = jnp.mean(gradF, axis=0)
                # Optimization step, start with Adam
                if switch == True:
                    updates, opt_state = sgd_optimizer.update(avg_gradF, opt_state)
                else:
                    updates, opt_state = adam_optimizer.update(avg_gradF, opt_state)
                # Apply the updates to A1D
                A1D = optax.apply_updates(A1D, updates)
                # Calculate the total error for the batch and accumulate it
                batch_error = jnp.mean(compute_error_batched(A1D, batch_halton))
                tot_error += batch_error

    
        # Calculate average error and append to the list
        avg_error = tot_error / jnp.float64(num_batches)
        if avg_error < min_error and k>10:
            min_error = avg_error
            min_A1D = A1D
            
        # Every 200 steps, write data to files and flush
        if k % 100 == 0:
            # Write the weights file after every 200 steps
        
            analytical_errors = jax.pmap(analytical_error_fn, in_axes=(None, 0))(A1D, halton_sequence)
            analytical_tot_error = jnp.sum(analytical_errors)
            analytical_avg_error = analytical_tot_error / len(halton_sequence)
            analytical_errors_list.append(f"{analytical_avg_error:.18e}\n")
        
            if k >= adam_max_iters and switch != True:
                print(" Switching the optimizer to SGD :")
                learning_rate = 0.00001
                print("avg_error : ", avg_error)
                switch = True
    
        
            # Parallelize energy error calculation
            energy_errors = jax.pmap(energy_error_fn, in_axes=(None, 0))(A1D, halton_sequence)
            tot_error_energy = jnp.sum(energy_errors)
            avg_error_energy = tot_error_energy / len(halton_sequence)
            energy_errors_list.append(f"{avg_error_energy:.18e}\n")
        
            # Parallelize validation error calculation
            validation_errors = jax.pmap(validation_error_fn, in_axes=(None, 0))(A1D, validation_halton)
            validation_tot_error = jnp.sum(validation_errors)
            validation_avg_error = validation_tot_error / len(validation_halton)
            validation_errors_list.append(f"{validation_avg_error:.18e}\n")
            
            
            with open(Error_path, 'a') as S1_analytical_error:
                S1_analytical_error.writelines(analytical_errors_list)
                S1_analytical_error.flush()
    
            with open(Energy_path, 'a') as S1_energy:
                S1_energy.writelines(energy_errors_list)
                S1_energy.flush()
    
            with open(Validation_path, 'a') as S1_validation:
                S1_validation.writelines(validation_errors_list)
                S1_validation.flush()
    
            # Clear lists after writing to files

            analytical_errors_list.clear()
            energy_errors_list.clear()
            validation_errors_list.clear()
    
            
            jnp_array_A1D = jnp.array(A1D, dtype=jnp.float64)
            np_array_A1D = np.array(jnp_array_A1D, dtype=np.float64)
            
            with open(FinalWeight_path, 'w') as S1_weights:
                np.savetxt(S1_weights, np_array_A1D, delimiter=',', fmt='%.18e')
                S1_weights.flush()

            

if __name__ == "__main__":
    
    # print("Starting profiling... : With Energy : Original Lobatto 3A 3B : Batched : B- ")

    # # Start timing here
    # start_time = time.time()

    # profiler = cProfile.Profile()
    # profiler.run('main()')

    # # Stop timing and calculate the duration
    # end_time = time.time()
    # duration = end_time - start_time

    # print(f"Total time taken: {duration:.2f} seconds")
    # # Get process ID
    # # pid = os.getpid()
    
    # # Save the profiler output with the process ID in the filename
    # # profile_filename = f'/pc2/users/r/rpandey/RK4/Partitioned_RK4/Implicit_RK4/Lobatto_weights/ADAM/Hubber_Loss_Function/For_Scratch/Halton_100/With_Energy/Outputs/Prof_Outputs/profile_output_{pid}.prof'
    # profile_filename = f'/pc2/users/r/rpandey/RK4/Partitioned_RK4/Implicit_RK4/Lobatto_weights/ADAM/Hubber_Loss_Function/For_Scratch/Halton_100/With_Energy/Outputs/Original_Prof_Outputs/B-profile_output.prof'
    # print("Profiling complete, saving data... : ")
    # profiler.dump_stats(profile_filename)
    # print(f"Profile data saved as {profile_filename}.")

    main()
    sys.exit(0)