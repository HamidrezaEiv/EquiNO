import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_ENABLE_X64'] = 'True'
import jax.numpy as jnp
import jax
import tensorflow as tf

class configs:
    rve = 'RVE1'
    method = 'EquiNO'
    name = f'{method}_{rve}'
    dtype = 'float64'

mdir = os.getcwd() + f'/../trained_models/{configs.name}'
model = tf.keras.models.load_model(mdir)
nodes = model.kinema.get_config()['nodeCoord']
scaling_bs = model.get_config()['scaling_bs']
scale_input = 0.04

_, _, t_s = model.trunk(nodes)
hom_t_s = model.homogenization(tf.transpose(t_s, (1, 0, 2))).numpy()

weights = model.branch.get_weights()
weights = [weights[i:i+2] for i in range(2, len(weights), 4)]
params = [(jnp.array(w[0]), jnp.array(w[1])) for w in weights]

# JAX MLP function
@jax.jit
def jax_branch(x):
    x_n = x
    for (w, b) in params[:-1]:
        x_n = jax.nn.swish(jnp.dot(x_n, w) + b)
    final_w, final_b = params[-1]
    return jnp.dot(x_n, final_w) + final_b

@jax.jit
def forward(x):
    x = x.reshape((1, 3))
    b_s = jax_branch(x / scale_input)
    b_s = b_s * scaling_bs[1] + scaling_bs[0]
    return jnp.einsum('ir,rn->n', b_s, hom_t_s)

jacobian = jax.jit(jax.jacrev(forward))

print('Model has been loaded successfully!')

@jax.jit
def prediction(*args):
    E = jnp.array(args)
    T = forward(E)
    C = jacobian(E).reshape((9,))

    tc = ()
    for t in T:
        tc = tc + (t,)
        
    for c in C:
        tc = tc + (c,)

    return tc

# print(prediction(0.02, 0.02, 0.02))