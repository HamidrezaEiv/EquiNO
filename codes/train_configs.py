import tensorflow as tf
class configs:
    n_cp = 1
    n_hl = 4
    n_nu = 64
    r = 16
    rve = 'RVE3'
    method = 'equino'
    name = f'{method}_{n_hl}_{n_nu}_{r}_{n_cp}_{rve}'
    dtype = 'float64'
    
    if dtype=='float32':
        floatx = tf.float32
    else:
        floatx = tf.float64
    