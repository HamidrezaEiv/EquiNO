import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, losses, activations
from nns import EquiNO
from utils import PreProcessing, plot_strain_stress, plot_disp
from train_configs import configs
from time import time
tf.keras.backend.set_floatx(configs.dtype)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

tf.config.set_visible_devices(gpus, 'GPU')

SEED = 24
random.seed(SEED)
np.random.seed(SEED)
#%% data
ddir = f'../data/{configs.rve}/data/'
gdir = f'../data/{configs.rve}/grid/'
prep = PreProcessing(data_dir=ddir, dtype=configs.dtype)
train_global, val_global, _ = prep.get_data_global()
train_local, val_local, _ = prep.get_data_local()
nodes, ips = prep.get_grid(gdir)
e_train = prep.gen_e(configs.n_cp, method='lhs')

d, e, s = train_local
e_hat, s_hat, c_hat = train_global

d_t, e_t, s_t = val_local
e_hat_t, s_hat_t, c_hat_t = val_global
#%% model
model = EquiNO((configs.n_hl, configs.n_nu,  3, configs.r, [1, 1], activations.swish), 
            (d, e_hat, nodes, s), 
            floatx=configs.floatx,
            gdir=gdir)

model.compile(
    optimizer = optimizers.Adam(learning_rate=1e-3),
    loss = losses.MeanSquaredError(),
    run_eagerly=False)
#%% Train
start_time = time()
model.fit(([nodes, e_train], None), epochs=1000)
model.fit_lbfgs(([nodes, e_train], None), minimizer='tfp')
end_time = time()
#%% save model
model.save(f'../checkpoints/{configs.name}', save_format='tf')
hist = [x.numpy() for x in model.hist] + [x.numpy() for x in model.hist_lbfgs]
np.savez_compressed(f'../checkpoints/hist_{configs.name}',
                                      history=np.array(hist), 
                                      training_time=end_time - start_time)
#%% eval
d_p, e_p, s_p, s_nn = model([nodes, e_hat_t])

plot_disp(nodes, d_t)
plot_disp(nodes, d_p)

plot_strain_stress(ips, e_t, s_t)
plot_strain_stress(ips, e_p, s_p)

print('error disp:', model.l2norm(d_t, d_p).numpy())
print('error local s', model.l2norm(s_t, s_p).numpy())