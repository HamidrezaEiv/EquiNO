import numpy as np
import tensorflow as tf

class PeriodicLayer3D(tf.keras.layers.Layer):
    def __init__(self, m, n, activation=tf.keras.activations.tanh, period=(1, 1), bias=True, *args, **kwargs):
        super(PeriodicLayer3D, self).__init__(*args, **kwargs)
        self.units = m
        self.n = n
        self.omg1 = 2 * np.pi / period[0]
        self.omg2 = 2 * np.pi / period[1]
        self.bias = bias
        
        self.activation = activation
        self.dense = tf.keras.layers.Dense(self.n, 
                                            activation=self.activation, 
                                            use_bias=self.bias)
        
        self.A1 = self.add_weight(shape=(self.units,),
                                 initializer="random_normal",
                                 trainable=True, name="A1")
        self.A2 = self.add_weight(shape=(self.units,),
                                 initializer="random_normal",
                                 trainable=True, name="A2")
        self.A3 = self.add_weight(shape=(self.units,),
                                 initializer="random_normal",
                                 trainable=True, name="A3")
        if self.bias:
            self.phi1 = self.add_weight(shape=(self.units,),
                                    initializer="random_normal",
                                    trainable=True, name="phi1")
            self.phi2 = self.add_weight(shape=(self.units,),
                                    initializer="random_normal",
                                    trainable=True, name="phi2")

            
            self.c1 = self.add_weight(shape=(self.units,),
                                    initializer="random_normal",
                                    trainable=True, name="c1")
            self.c2 = self.add_weight(shape=(self.units,),
                                    initializer="random_normal",
                                    trainable=True, name="c2")
            self.c3 = self.add_weight(shape=(self.units,),
                                    initializer="random_normal",
                                    trainable=True, name="c3")
        else:
            self.phi1 = 0.0
            self.phi2 = 0.0
            self.c1 = 0.0
            self.c2 = 0.0

    def call(self, inputs):
        # Define periodic functions
        v1 = tf.math.cos(self.omg1 * inputs[..., 0:1] + self.phi1) 
        v2 = tf.math.cos(self.omg2 * inputs[..., 1:2] + self.phi2)
        v3 = inputs[..., 2:3]
        
        v = self.activation(self.A1 * v1 + self.c1) + self.activation(self.A2 * v2 + self.c2) + self.activation(self.A3 * v3 + self.c3)
        # Compute the output of the layer
        q = self.dense(v)
        
        return q
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n)
    
