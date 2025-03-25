import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import random
import numpy as np
import tensorflow as tf
from utils import PreProcessing, plot_compare_errors

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

tf.config.set_visible_devices(gpus, 'GPU')

SEED = 24
random.seed(SEED)
np.random.seed(SEED)

class configs:
    rve = 'RVE3'
    dtype = 'float64'
#%% data
ddir = f'../data/{configs.rve}/data/'
gdir = f'../data/{configs.rve}/grid/'
prep = PreProcessing(data_dir=ddir, dtype=configs.dtype)
_, _, test_global = prep.get_data_global()
_, _, test_local = prep.get_data_local()
nodes, ips = prep.get_grid(gdir)

d_t, e_t, s_t = test_local
e_hat_t, s_hat_t, c_hat_t = test_global

gm = prep.get_gmatid(gdir)
if configs.rve != 'RVE1':
    gm = -gm
nodes_gm = np.concatenate((nodes, gm), 1)
#%% test VPIONet
model_name = f'VPIONet_{configs.rve}'
vpionet = tf.keras.models.load_model(f'../trained_models/{model_name}')

d_p1, e_p1, s_p1 = vpionet([nodes_gm, e_hat_t])
s_hat_p1 = vpionet.homogenization(s_p1)

def l2norm(a, b):
    return np.round(vpionet.l2norm(a, b), 2)

print('\n## errors VPIONet ##')
print('error disp:', l2norm(d_t, d_p1))
print('error local s', l2norm(s_t, s_p1))
print('mean error local s', np.round(l2norm(s_t, s_p1).mean(), 2))

print('error global s', l2norm(s_hat_t[:,None,:], s_hat_p1[:,None,:]))
#%% test EquiNO
model_name = f'EquiNO_{configs.rve}'
equino = tf.keras.models.load_model(f'../trained_models/{model_name}')

d_p2, e_p2, s_p2, _ = equino([nodes, e_hat_t])
s_hat_p2 = equino.homogenization(s_p2)

def l2norm(a, b):
    return np.round(equino.l2norm(a, b), 2)

print('\n## errors EquiNO ##')
print('error disp:', l2norm(d_t, d_p2))
print('error local s', l2norm(s_t, s_p2))
print('mean error local s', np.round(l2norm(s_t, s_p2).mean(), 2))

print('error global s', l2norm(s_hat_t[:,None,:], s_hat_p2[:,None,:]))

l2_error = (np.linalg.norm(s_t - s_p2, axis=(1)) / np.linalg.norm(s_t, axis=(1))).mean(-1)
sample = np.argsort(l2_error)[(len(l2_error) - 1)//2]
#%%
plot_compare_errors(
    (d_t[sample, :, 0], d_t[sample, :, 1], s_t[sample, :, 0],  s_t[sample, :, 1],  s_t[sample, :, 2]),
    (d_p1[sample, :, 0], d_p1[sample, :, 1], s_p1[sample, :, 0],  s_p1[sample, :, 1],  s_p1[sample, :, 2]),
    (d_p2[sample, :, 0], d_p2[sample, :, 1], s_p2[sample, :, 0],  s_p2[sample, :, 1],  s_p2[sample, :, 2]),
    nodes,
    ips,
    save=True,
    name=f'NOs_{configs.rve}')