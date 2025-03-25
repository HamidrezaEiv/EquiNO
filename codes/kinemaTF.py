import tensorflow as tf
import pandas as pd
from space_integration_mod import space_int

class KinematicLayer(tf.keras.layers.Layer):
    def __init__(self, data_dir='../data/grid/', floatx=tf.float32, **kwargs):
        super(KinematicLayer, self).__init__(**kwargs)
        self.floatx=floatx
        self.nodeCoord = pd.read_csv(data_dir + 'nodeCoordinates.csv',header=None).to_numpy()
        self.elemInc = pd.read_csv(data_dir + 'elementIncidences.csv',header=None).to_numpy()
        
        self.elemNodes = self.elemInc[:, 2:]
        self.elemNodeCoord = self.nodeCoord[self.elemNodes - 1, :]
        
        self.detj, self.shpdx, self.wt, self.shp, self.we = self.get_shp()
        self.reshape = tf.keras.layers.Reshape((-1, 3))

    def inv_jac(self, shpGP):
        jacobian = tf.einsum('ijk,lin->lknj', shpGP, self.elemNodeCoord)
        detj = tf.linalg.det(jacobian)
        invJac = tf.linalg.inv(jacobian)
        shpdx = tf.einsum('ijk,lkjn->lkin', shpGP, invJac)
        return detj, shpdx

    def get_shp(self):
        elemType = 'quad8'
        [weight, shp] = space_int(elemType)
        
        weight = tf.constant(weight, dtype=self.floatx)
        shp = tf.constant(shp, dtype=self.floatx)
        
        detj, shpdx = self.inv_jac(shp[:, 1:])
        
        wt = tf.reshape((detj * weight[None, ...]), (-1,))
        return detj, shpdx, wt, shp, weight
    
    @tf.function
    def diff(self, d):
        dd = tf.einsum('ijkl,bikn->bijnl', self.shpdx, d)
        e_xx = dd[:, :, :, 0, 0]
        e_yy = dd[:, :, :, 1, 1]
        e_xy = 0.5 * (dd[:, :, :, 0, 1] + dd[:, :, :, 1, 0])
        e = tf.stack([e_xx, e_yy, e_xy], axis=-1)
        return e
    
    @tf.function
    def call(self, inputs):
        disp = tf.gather(inputs, self.elemNodes - 1, axis=1)
        dd = self.diff(disp)
        return self.reshape(dd)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'nodeCoord': self.nodeCoord,
            'elemInc': self.elemInc,
            'elemNodes': self.elemNodes,
            'elemNodeCoord': self.elemNodeCoord,
            'floatx': self.floatx,
            'reshape': self.reshape.get_config(),
            'detj': self.detj.numpy(),
            'shpdx': self.shpdx.numpy(),
            'wt': self.wt.numpy(),
            'shp': self.shp.numpy(),
            'we': self.we.numpy(),
        })
        return config

