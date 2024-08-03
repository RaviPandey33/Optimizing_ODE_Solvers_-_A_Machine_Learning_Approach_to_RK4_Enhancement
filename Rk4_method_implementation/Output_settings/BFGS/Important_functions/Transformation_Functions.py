import numpy
import jax.numpy as jnp

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


def Aminus(M, dr):
    """
    THis function subtracts dr from M and then return the updated value of M
    : param M : 1D array
    : param dr : 1D array
    : return : 1D array after subtracting dr from M

    """

    M = M, dr
    return M


def Aplus(M, dr):
    """
    THis function add dr to M and then return the updated value of M
    : param M : 1D array
    : param dr : 1D array
    : return : 1D array after adding dr to M

    """

    M = jnp.add(M, dr)
    return M


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
    #H_sequence = A[40:]
    return A1, A2 #, H_sequence

def PRKactual_A1_A2(A): # from the returned gradient array of 20 elements, we find the elements of array A and array B
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