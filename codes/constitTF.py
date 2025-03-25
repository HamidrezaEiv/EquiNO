import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from train_configs import configs
tfdtype = configs.floatx

class ConstitutiveLayer(layers.Layer):
    def __init__(self, data_dir='../data/grid/', floatx=tf.float32, Nips=9, *args, **kwargs):
        super(ConstitutiveLayer, self).__init__(*args, **kwargs)
        self.floatx=floatx
        elemInc = pd.read_csv(data_dir + 'elementIncidences.csv',header=None).to_numpy() 
        matid = np.tile(elemInc[:, 1:2], [1, Nips]).reshape((-1, 1)) - 1
        self.matid = tf.convert_to_tensor(matid, dtype=self.floatx)

        
        self.matparNonlinElas = tf.constant([4.78e03, 5.0e01, 6.0e-02], dtype=self.floatx)
        self.matparLinElas = tf.constant([4.34523810e04, 2.99180328e+04], dtype=self.floatx)
        self.one = tf.constant([1.0, 1.0, 0.0], dtype=self.floatx)[None, ...]
    
    @tf.function
    def linElasticity(self, strain):
        trE = tf.reduce_sum(strain[:, :2], axis = 1, keepdims=True)
        strainDev = strain - trE/3.0 @ self.one
        stress = self.matparLinElas[0]*trE@self.one + 2*self.matparLinElas[1]*strainDev
        return stress
    
    @tf.function
    def scalarProdVoigt(self, a):
        mask = tf.constant([1.0, 1.0, 2.0], dtype=self.floatx)
        scalProd = tf.reduce_sum(mask * a**2, axis = 1, keepdims=True)
        return scalProd
    
    @tf.function
    def nonlinElasticity(self, strain):
        trE = tf.reduce_sum(strain[:, :2], axis = 1, keepdims=True)
        strainDev = strain - trE/3.0 @ self.one
        normStrainDev = tf.sqrt(self.scalarProdVoigt(strainDev))
        stress = self.matparNonlinElas[0]*trE@self.one +\
            strainDev * (self.matparNonlinElas[1]/(self.matparNonlinElas[2]+normStrainDev))
        return stress
    
    @tf.function
    def apply_fn(self, e):
        s = (1.0 - self.matid) * self.nonlinElasticity(e)\
              + self.matid * self.linElasticity(e)
        return s 
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 3], dtype=tfdtype)])
    def call(self, x):
        s = tf.vectorized_map(self.apply_fn, x, fallback_to_while_loop=False)
        return s
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'matid': self.matid.numpy(),
            'matparNonlinElas': self.matparNonlinElas.numpy(),
            'matparLinElas': self.matparLinElas.numpy(),
            'floatx': self.floatx,
            'one': self.one.numpy(),
        })
        return config
