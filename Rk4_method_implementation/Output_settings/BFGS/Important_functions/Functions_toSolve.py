import jax.numpy as jnp


def f(y, z, alpha_values):
    return z

def g(y, z, alpha_values):
    alpha_values = alpha_values.transpose()
    return jnp.add(jnp.add((-1 * alpha_values[0]) , (-2 * alpha_values[1] * y) ) , jnp.add((-3 * alpha_values[2]* (y**2)) , (-4 * alpha_values[3] * (y**3) )) )

def Energy_Function(y, z, alpha_values):
    return ((jnp.square(y))/2 + jnp.add(jnp.add(( alpha_values[0]* (z)) , (alpha_values[1] * (z**2)) ) , jnp.add((alpha_values[2]* (z**3)) , (alpha_values[3] * (z**4) )) )) 


# def f(y, z, alpha_values):
#     return z

# def g(y, z, alpha_values):
#     return -y

# def Energy_Function(y, z, alpha_values):
#     return (jnp.square(y) + jnp.square(z))/2