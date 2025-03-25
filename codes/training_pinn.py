import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, losses, activations
from nns import PINN, EquiNO
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

method = 'pinn' # pinn or equino
#%% data
ddir = f'../data/{configs.rve}/data/'
gdir = f'../data/{configs.rve}/grid/'
prep = PreProcessing(data_dir=ddir, dtype=configs.dtype)
train_global, val_global, _ = prep.get_data_global()
train_local, val_local, _ = prep.get_data_local()
nodes, ips = prep.get_grid(gdir)

d, e, s = train_local
e_hat, s_hat, c_hat = train_global

vglobal, vlocal = prep.get_sample(val_global, val_local, i=0)
d_t, e_t, s_t = vlocal
e_hat_t, s_hat_t, c_hat_t = vglobal
e_train = e_hat_t
    #%% model
if method == 'pinn':
    pinn = PINN((configs.n_hl, configs.n_nu,  3, 2, activations.tanh), 
                floatx=configs.floatx,
                gdir=gdir)
    gm = prep.get_gmatid(gdir)
    nodes = np.concatenate((nodes, gm), 1)
elif method == 'equino':
    pinn = EquiNO((configs.n_hl, configs.n_nu,  3, configs.r, [1, 1], activations.swish), 
                (d, e_hat, nodes, s), 
                floatx=configs.floatx,
                gdir=gdir)
else:
    print('Method not implemented! Set method to pinn or equino.')
    
pinn.compile(
    optimizer = optimizers.Adam(learning_rate=1e-3),
    loss = losses.MeanSquaredError(),
    run_eagerly=False)
#%% Train
start_time = time()
pinn.fit(([nodes, e_train], s_t), epochs=1000)
pinn.fit_lbfgs(([nodes, e_train], s_t), minimizer='tfp')
end_time = time()
#%% save model
pinn.save(f'../checkpoints/{configs.name}', save_format='tf')
hist = [x.numpy() for x in pinn.hist] + [x.numpy() for x in pinn.hist_lbfgs]
np.savez_compressed(f'../checkpoints/hist_{configs.name}',
                                      history=np.array(hist), 
                                      training_time=end_time - start_time)
#%% eval
if method == 'pinn':
    d_p, e_p, s_p = pinn([nodes, e_hat_t])
    nodes = nodes[:, :-1]
else:
    d_p, e_p, s_p, s_nn = pinn([nodes, e_hat_t])
    
plot_disp(nodes, d_t)
plot_disp(nodes, d_p)

plot_strain_stress(ips, e_t, s_t)
plot_strain_stress(ips, e_p, s_p)

print('error disp:', pinn.l2norm(d_t, d_p).numpy())
print('error local s', pinn.l2norm(s_t, s_p).numpy())