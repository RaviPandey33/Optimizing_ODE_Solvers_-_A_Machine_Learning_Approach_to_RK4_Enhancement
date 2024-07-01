import Transformation_Functions as TFunctions
import jax.numpy as jnp

def Convert_toOneD(A1, A2, B1, B2):
    A1D = TFunctions.One_Dim_Matrix(A1)
    A1D = TFunctions.Add_B_tomatrix_A(A1D, B1)
    A2D = TFunctions.One_Dim_Matrix(A2)
    A2D = TFunctions.Add_B_tomatrix_A(A2D, B2)
    A1D = TFunctions.Add_B_tomatrix_A(A1D, A2D)
    
    return A1D

def Convert_toTwoD(A1D):
    new_A1, new_B1 = TFunctions.actual_A_1D(A1D[0:20])
    new_A2, new_B2 = TFunctions.actual_A_1D(A1D[20:40])

    #converting A to a 2D Array
    new_A1 = TFunctions.One_D_to_TwoD(new_A1)
    new_A2 = TFunctions.One_D_to_TwoD(new_A2)
    
    return new_A1, new_A2, new_B1, new_B2