import numpy as np 
import tensorflow as tf
from tensorflow.keras import models, metrics, layers
from constitTF import ConstitutiveLayer
from kinemaTF import KinematicLayer
from periodicTF3D import PeriodicLayer3D
import scipy.optimize as sopt
import tensorflow_probability as tfp
from tensorflow.data import Dataset
from train_configs import configs
tfdtype = configs.floatx

class pod():
    def __init__(self, r = 16, floatx=tf.float32, gdir='../data/grid/'):
        self.r = r
        self.floatx = floatx
        self.kinema = KinematicLayer(floatx=self.floatx, data_dir=gdir)
    def periodic_disp(self, inputs, e_hat):
        cd = tf.constant([[2.0, 0.0], [0.0, 2.0]], dtype=self.floatx)
        matrix = inputs[...,None] * cd[None,...]
        matrix = 0.5 * tf.concat([matrix, tf.reverse(inputs, (1,))[:,None,:]], 1)
        return tf.einsum('bi,nim->bnm', e_hat, matrix)
    
    def modes(self, data):
        X, e_hat, nodes, s = data
        n_nodes = len(nodes)
        periodic_mat = self.periodic_disp(nodes, e_hat)
        snapshots = X - periodic_mat
        snapshots = snapshots.numpy().reshape((snapshots.shape[0], -1))
        phi, sigma, vh = np.linalg.svd(snapshots.T, full_matrices=False)
        cum_energy = np.cumsum(sigma) / np.sum(sigma)
        print('cumulative energy:', cum_energy[self.r])
        phi = phi[:, :self.r].reshape((n_nodes, -1, self.r))
        phi = np.transpose(phi, (0, 2, 1))
        cu = np.diag(sigma) @ vh
        scaling_bu = [cu.mean(), cu.std()]

        phi_e = tf.transpose(self.kinema(np.transpose(phi, (1, 0, 2))), (1, 0, 2))

        ns = s.shape[1]
        snapshots = s.reshape((s.shape[0], -1))
        phi_s, sigma, vh = np.linalg.svd(snapshots.T, full_matrices=False)
        cum_energy = np.cumsum(sigma) / np.sum(sigma)
        print('cumulative energy:', cum_energy[self.r])
        phi_s = phi_s[:, :self.r].reshape((ns, -1, self.r))
        phi_s = np.transpose(phi_s, (0, 2, 1))
        cs = np.diag(sigma) @ vh
        scaling_bs = [cs.mean(), cs.std()]
        return tf.convert_to_tensor(phi, dtype=self.floatx),\
                phi_e,\
                tf.convert_to_tensor(phi_s, dtype=self.floatx),\
                [scaling_bu, scaling_bs]

class EquiNO(models.Model):
    def __init__(self, branch_arc, trunk_arc, batch_size=10, floatx=tf.float32, gdir='../data/grid/', *args, **kwargs):
        super(EquiNO, self).__init__(*args, **kwargs)
        
        self.phi, self.phi_e, self.phi_s, scaling = pod(r=branch_arc[3], floatx=floatx, gdir=gdir).modes(trunk_arc)
        self.trunk = self.basis_functions
        self.scaling_bu = scaling[0]
        self.scaling_bs = scaling[1]        

        self.branch = self.build_branch(branch_arc)
        
        self.loss_tracker = metrics.Mean(name="loss")
        self.floatx = floatx
        self.kinema = KinematicLayer(floatx=self.floatx, data_dir=gdir)
        self.constit = ConstitutiveLayer(floatx=self.floatx, data_dir=gdir)
        
        self.sopt = lbfgs_optimizer(self.trainable_variables, floatx=self.floatx)
        self.hist = []
        self.hist_lbfgs = []
        self.batch_size=batch_size
        
        self.scale_input = 0.04
            
    @tf.function
    def basis_functions(self, nodes):
        return self.phi, self.phi_e, self.phi_s
    
    def build_branch(self, nn_arc):
        num_hl, num_neu, input_dim, output_dim, _, act = nn_arc
        inp = layers.Input(shape = (input_dim,))
        outs = []
        for ii in range(2):
            hl = inp
            for i in range(num_hl):
                hl = layers.Dense(num_neu, activation=act)(hl)
        
            outs.append(layers.Dense(output_dim)(hl))
        
        model = models.Model(inp, outs)
        print(model.summary())
        return model
        
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tfdtype), 
                                  tf.TensorSpec(shape=[None, None, None], dtype=tfdtype)])
    def l2norm(self, ref, pred):
        return tf.reduce_mean(tf.norm(ref - pred, axis=(1)) / tf.norm(ref, axis=(1)) * 100, axis=0)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 3], dtype=tfdtype), 
                                  tf.TensorSpec(shape=[None, None, 3], dtype=tfdtype)])
    def dem_loss(self, e, s):
        strain_energy = 0.5 * (e[...,0]*s[...,0] + e[...,1]*s[...,1] + 2.0*e[...,2]*s[...,2])
        dems = tf.reduce_sum(strain_energy * self.kinema.wt, axis=-1)
        return tf.reduce_mean(dems)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 3], dtype=tfdtype),
                                  tf.TensorSpec(shape=[None, None, 3], dtype=tfdtype)])
    def strain_energy(self, e, s):
        return 0.5 * (e[...,0]*s[...,0] + e[...,1]*s[...,1] + 2.0*e[...,2]*s[...,2]) * self.kinema.wt
    
    @tf.function()
    def loss_fn(self, s, s_nn):
        return self.loss(s, s_nn)

    @tf.function
    def periodic_disp(self, inputs):
        nodes, e_hat = inputs
        cd = tf.constant([[2.0, 0.0], [0.0, 2.0]], dtype=self.floatx)
        matrix = nodes[...,None] * cd[None,...]
        matrix = 0.5 * tf.concat([matrix, tf.reverse(nodes, (1,))[:,None,:]], 1)
        return tf.einsum('ij,ljm->ilm', e_hat, matrix)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 3], dtype=tfdtype)])
    def homogenization(self, x):
        return tf.reduce_sum(x * self.kinema.wt[None, :, None], 1)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 3], dtype=tfdtype)])
    def shear_divide_2(self, x):
        return x * tf.constant([[1.0, 1.0, 0.5]], dtype=self.floatx)
    
    @tf.function
    def call(self, inputs):
        nodes, e_hat = inputs
        
        t, t_e, t_s = self.trunk(nodes)
        b, b_s = self.branch(e_hat / self.scale_input)
        
        b = b * self.scaling_bu[1] + self.scaling_bu[0]
        b_s = b_s * self.scaling_bs[1] + self.scaling_bs[0]
        
        uv = tf.einsum('im,lmn->iln', b, t)
        s_nn = tf.einsum('im,lmn->iln', b_s, t_s)

        uv = uv - uv[:, :1] + self.periodic_disp(inputs)
        e = tf.einsum('im,lmn->iln', b, t_e) + self.shear_divide_2(e_hat)[:, None, :]
        s = self.constit(e)
        
        return uv, e, s, s_nn
           
    @tf.function
    def train_step(self, data):
        inputs, outputs = data
        with tf.GradientTape() as tape:
            u, e, s, s_nn = self(inputs, training=True)

            loss = self.loss_fn(s, s_nn) 
  
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]
    
    @tf.function
    def test_step(self, data):
        inputs, outputs = data
        u, e, s, s_nn = self(inputs, training=False)

        loss = self.loss_fn(s, s_nn) 
        
        self.loss_tracker.update_state(loss)
    
        return {"loss": self.loss_tracker.result()}
    
    def fit(self, data, epochs):
        inputs, outputs = data
        nodes, e_hat = inputs
        e_hat = Dataset.from_tensor_slices(e_hat).batch(self.batch_size)
        iterator = iter(e_hat.shuffle(5, reshuffle_each_iteration=True).repeat())
        for epoch in range(epochs):
            hist = self.train_step(([nodes, iterator.get_next()], outputs))
            tf.print(f"epoch {epoch}/{epochs}:", "loss:", hist["loss"])
            self.hist.append(hist["loss"])

    @tf.function
    def train_step_sopt(self, data):
        inputs, outputs = data
        with tf.GradientTape() as tape:
            u, e, s, s_nn = self(inputs, training=True)

            loss = self.loss_fn(s, s_nn) 
  
        grads = tape.gradient(loss, self.trainable_variables)
        return loss, grads
    
    def fit_lbfgs(self, data, minimizer='scipy'):
        self.lbfgs_epoch = len(self.hist)
        if minimizer=='tfp':
            def func(params_1d):
                self.sopt.assign_params(params_1d)
                loss, grads = self.train_step_sopt(data)
                grads = tf.dynamic_stitch(self.sopt.idx, grads)
                
                tf.print(f"epoch {self.lbfgs_epoch}:", "loss:", loss)
                self.hist_lbfgs.append(loss)
                self.lbfgs_epoch +=1
                return loss, grads
                
            self.sopt.tfp_minimize(func)
        else:
            def func(params_1d):
                self.sopt.assign_params(params_1d)
                loss, grads = self.train_step_sopt(data)
                grads = tf.dynamic_stitch(self.sopt.idx, grads)
                
                tf.print(f"epoch {self.lbfgs_epoch}:", "loss:", loss)
                self.hist_lbfgs.append(loss)
                self.lbfgs_epoch +=1
                return loss.numpy().astype(np.float64), grads.numpy().astype(np.float64)
                
            self.sopt.scipy_minimize(func)
        return np.array(self.hist_lbfgs)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'floatx': self.floatx,
            'scaling_bu': self.scaling_bu,
            'scaling_bs': self.scaling_bs,
        })
        return config

class VPIONet(models.Model):
    def __init__(self, branch_arc, trunk_arc, batch_size=10, floatx=tf.float32, gdir='../data/grid/', *args, **kwargs):
        super(VPIONet, self).__init__(*args, **kwargs)
        
        self.trunk = self.build_trunk(trunk_arc)

        self.branch = self.build_branch(branch_arc)
        
        self.loss_tracker = metrics.Mean(name="loss")
        self.floatx = floatx
        self.kinema = KinematicLayer(floatx=self.floatx, data_dir=gdir)
        self.constit = ConstitutiveLayer(floatx=self.floatx, data_dir=gdir)
        
        self.sopt = lbfgs_optimizer(self.trainable_variables, floatx=self.floatx)
        self.hist = []
        self.hist_lbfgs = []
        self.batch_size=batch_size
        
        self.scale_input = 0.04
        self.scale_output = 1e-4
        
    def build_trunk(self, nn_arc):
        num_hl, num_neu, input_dim, output_dim, no, act = nn_arc
        inps = layers.Input(shape = (input_dim,))

        hl = PeriodicLayer3D(num_neu, num_neu, activation=act)(inps)
        for i in range(num_hl):
            hl = layers.Dense(num_neu, activation=act)(hl)
        
        output_basis_functions = []
        for i in range(no):
            output_basis_functions.append(layers.Dense(output_dim)(hl))
        
        out = layers.Concatenate(-1)(output_basis_functions)
        out = layers.Reshape((output_dim, no))(out)
        
        model = models.Model(inps, out)
        print(model.summary())
        return model
    
    def build_branch(self, nn_arc):
        num_hl, num_neu, input_dim, output_dim, _, act = nn_arc
        inp = layers.Input(shape = (input_dim,))

        hl = inp
        for i in range(num_hl):
            hl = layers.Dense(num_neu, activation=act)(hl)
    
        out = layers.Dense(output_dim)(hl)
        
        model = models.Model(inp, out)
        print(model.summary())
        return model
        
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tfdtype), 
                                  tf.TensorSpec(shape=[None, None, None], dtype=tfdtype)])
    def l2norm(self, ref, pred):
        return tf.reduce_mean(tf.norm(ref - pred, axis=(1)) / tf.norm(ref, axis=(1)) * 100, axis=0)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 3], dtype=tfdtype), 
                                  tf.TensorSpec(shape=[None, None, 3], dtype=tfdtype)])
    def dem_loss(self, e, s):
        strain_energy = 0.5 * (e[...,0]*s[...,0] + e[...,1]*s[...,1] + 2.0*e[...,2]*s[...,2])
        dems = tf.reduce_sum(strain_energy * self.kinema.wt, axis=-1)
        return tf.reduce_mean(dems)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 3], dtype=tfdtype),
                                  tf.TensorSpec(shape=[None, None, 3], dtype=tfdtype)])
    def strain_energy(self, e, s):
        return 0.5 * (e[...,0]*s[...,0] + e[...,1]*s[...,1] + 2.0*e[...,2]*s[...,2]) * self.kinema.wt
    
    @tf.function()
    def loss_fn(self, e, s):
        return self.dem_loss(e, s)

    @tf.function
    def periodic_disp(self, inputs):
        nodes, e_hat = inputs
        nodes = nodes[:, :-1]
        cd = tf.constant([[2.0, 0.0], [0.0, 2.0]], dtype=self.floatx)
        matrix = nodes[...,None] * cd[None,...]
        matrix = 0.5 * tf.concat([matrix, tf.reverse(nodes, (1,))[:,None,:]], 1)
        return tf.einsum('ij,ljm->ilm', e_hat, matrix)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 3], dtype=tfdtype)])
    def homogenization(self, x):
        return tf.reduce_sum(x * self.kinema.wt[None, :, None], 1)
    
    @tf.function
    def call(self, inputs):
        nodes, e_hat = inputs
        t = self.trunk(nodes)
        b = self.branch(e_hat / self.scale_input)

        uv = tf.einsum('im,lmn->iln', b, t)

        uv = (uv - uv[:, :1]) * self.scale_output + self.periodic_disp(inputs)
        e = self.kinema(uv)
        s = self.constit(e)
        
        return uv, e, s
           
    @tf.function
    def train_step(self, data):
        inputs, outputs = data
        with tf.GradientTape() as tape:
            u, e, s = self(inputs, training=True)

            loss = self.loss_fn(e, s) 
  
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]
    
    @tf.function
    def test_step(self, data):
        inputs, outputs = data
        u, e, s = self(inputs, training=False)

        loss = self.loss_fn(e, s) 
        
        self.loss_tracker.update_state(loss)
    
        return {"loss": self.loss_tracker.result()}
    
    def fit(self, data, epochs):
        inputs, outputs = data
        nodes, e_hat = inputs
        e_hat = Dataset.from_tensor_slices(e_hat).batch(self.batch_size)
        iterator = iter(e_hat.shuffle(5, reshuffle_each_iteration=True).repeat())
        for epoch in range(epochs):
            hist = self.train_step(([nodes, iterator.get_next()], outputs))
            tf.print(f"epoch {epoch}/{epochs}:", "loss:", hist["loss"])
            self.hist.append(hist["loss"])

    @tf.function
    def train_step_sopt(self, data):
        inputs, outputs = data
        with tf.GradientTape() as tape:
            u, e, s = self(inputs, training=True)

            loss = self.loss_fn(e, s) 
  
        grads = tape.gradient(loss, self.trainable_variables)
        return loss, grads
    
    def fit_lbfgs(self, data, minimizer='scipy'):
        self.lbfgs_epoch = len(self.hist)
        if minimizer=='tfp':
            def func(params_1d):
                self.sopt.assign_params(params_1d)
                loss, grads = self.train_step_sopt(data)
                grads = tf.dynamic_stitch(self.sopt.idx, grads)
                
                tf.print(f"epoch {self.lbfgs_epoch}:", "loss:", loss)
                self.hist_lbfgs.append(loss)
                self.lbfgs_epoch +=1
                return loss, grads
                
            self.sopt.tfp_minimize(func)
        else:
            def func(params_1d):
                self.sopt.assign_params(params_1d)
                loss, grads = self.train_step_sopt(data)
                grads = tf.dynamic_stitch(self.sopt.idx, grads)
                
                tf.print(f"epoch {self.lbfgs_epoch}:", "loss:", loss)
                self.hist_lbfgs.append(loss)
                self.lbfgs_epoch +=1
                return loss.numpy().astype(np.float64), grads.numpy().astype(np.float64)
                
            self.sopt.scipy_minimize(func)
        return np.array(self.hist_lbfgs)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'floatx': self.floatx,
        })
        return config

class PINN(models.Model):
    def __init__(self, net_arc, batch_size=10, floatx=tf.float32,
                 gdir='../data/grid/', *args, **kwargs):
        super(PINN, self).__init__(*args, **kwargs)

        self.loss_tracker = metrics.Mean(name="loss")
        self.floatx = floatx
        self.kinema = KinematicLayer(floatx=self.floatx, data_dir=gdir)
        self.constit = ConstitutiveLayer(floatx=self.floatx, data_dir=gdir)
        
        self.sopt = lbfgs_optimizer(self.trainable_variables, floatx=self.floatx)
        self.hist = []
        self.hist_lbfgs = []
        self.batch_size=batch_size
        
        self.scale_output = 1e-4

    def build_net(self, nn_arc):
        num_hl, num_neu, input_dim, output_dim, act = nn_arc
        inps = layers.Input(shape = (input_dim,))
        hl = PeriodicLayer3D(num_neu, num_neu, activation=act)(inps)
        for i in range(num_hl):
            hl = layers.Dense(num_neu, activation=act)(hl)
        out = layers.Dense(output_dim)(hl)
        
        model = models.Model(inps, out)
        print(model.summary())
        return model
        
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tfdtype), 
                                  tf.TensorSpec(shape=[None, None, None], dtype=tfdtype)])
    def l2norm(self, ref, pred):
        return tf.reduce_mean(tf.norm(ref - pred, axis=(1)) / tf.norm(ref, axis=(1)) * 100, axis=0)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 3], dtype=tfdtype), 
                                  tf.TensorSpec(shape=[None, None, 3], dtype=tfdtype)])
    def dem_loss(self, e, s):
        strain_energy = 0.5 * (e[...,0]*s[...,0] + e[...,1]*s[...,1] + 2.0*e[...,2]*s[...,2])
        dems = tf.reduce_sum(strain_energy * self.kinema.wt, axis=-1)
        return tf.reduce_mean(dems)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 3], dtype=tfdtype),
                                  tf.TensorSpec(shape=[None, None, 3], dtype=tfdtype)])
    def strain_energy(self, e, s):
        return 0.5 * (e[...,0]*s[...,0] + e[...,1]*s[...,1] + 2.0*e[...,2]*s[...,2]) * self.kinema.wt
    
    @tf.function()
    def loss_fn(self, e, s):
        return self.dem_loss(e, s)

    @tf.function
    def periodic_disp(self, inputs):
        nodes, e_hat = inputs
        nodes = nodes[:, :-1]
        cd = tf.constant([[2.0, 0.0], [0.0, 2.0]], dtype=self.floatx)
        matrix = nodes[...,None] * cd[None,...]
        matrix = 0.5 * tf.concat([matrix, tf.reverse(nodes, (1,))[:,None,:]], 1)
        return tf.einsum('ij,ljm->ilm', e_hat, matrix)
    
    @tf.function
    def call(self, inputs):
        nodes, e_hat = inputs
        uv = self.net(nodes)[None,...]
        uv = (uv - uv[:, :1]) * self.scale_output + self.periodic_disp(inputs)
        e = self.kinema(uv)
        s = self.constit(e)
        
        return uv, e, s
           
    @tf.function
    def train_step(self, data):
        inputs, outputs = data
        with tf.GradientTape() as tape:
            u, e, s = self(inputs, training=True)

            loss = self.loss_fn(e, s)
  
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]
    
    @tf.function
    def test_step(self, data):
        inputs, outputs = data
        u, e, s = self(inputs, training=False)

        loss = self.loss_fn(e, s) 
        
        self.loss_tracker.update_state(loss)
    
        return {"loss": self.loss_tracker.result()}
    
    def fit(self, data, epochs):
        inputs, outputs = data
        nodes, e_hat = inputs
        e_hat = Dataset.from_tensor_slices(e_hat).batch(self.batch_size)
        iterator = iter(e_hat.shuffle(5, reshuffle_each_iteration=True).repeat())
        for epoch in range(epochs):
            hist = self.train_step(([nodes, iterator.get_next()], outputs))
            tf.print(f"epoch {epoch}/{epochs}:", "loss:", hist["loss"])
            self.hist.append(hist["loss"])

    @tf.function
    def train_step_sopt(self, data):
        inputs, outputs = data
        with tf.GradientTape() as tape:
            u, e, s = self(inputs, training=True)

            loss = self.loss_fn(e, s) 
  
        grads = tape.gradient(loss, self.trainable_variables)
        return loss, grads
    
    def fit_lbfgs(self, data, minimizer='scipy'):
        self.lbfgs_epoch = len(self.hist)
        if minimizer=='tfp':
            def func(params_1d):
                self.sopt.assign_params(params_1d)
                loss, grads = self.train_step_sopt(data)
                grads = tf.dynamic_stitch(self.sopt.idx, grads)
                
                tf.print(f"epoch {self.lbfgs_epoch}:", "loss:", loss)
                self.hist_lbfgs.append(loss)
                self.lbfgs_epoch +=1
                return loss, grads
                
            self.sopt.tfp_minimize(func)
        else:
            def func(params_1d):
                self.sopt.assign_params(params_1d)
                loss, grads = self.train_step_sopt(data)
                grads = tf.dynamic_stitch(self.sopt.idx, grads)
                
                tf.print(f"epoch {self.lbfgs_epoch}:", "loss:", loss)
                self.hist_lbfgs.append(loss)
                self.lbfgs_epoch +=1
                return loss.numpy().astype(np.float64), grads.numpy().astype(np.float64)
                
            self.sopt.scipy_minimize(func)
        return np.array(self.hist_lbfgs)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'floatx': self.floatx,
        })
        return config

class lbfgs_optimizer():
    def __init__(self, trainable_vars, floatx=tf.float32, method = 'L-BFGS-B'):
        super(lbfgs_optimizer, self).__init__()
        self.trainable_variables = trainable_vars
        self.floatx=floatx
        self.method = method
        
        self.shapes = tf.shape_n(self.trainable_variables)
        self.n_tensors = len(self.shapes)

        count = 0
        idx = [] # stitch indices
        part = [] # partition indices
    
        for i, shape in enumerate(self.shapes):
            n = np.product(shape)
            idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
            part.extend([i]*n)
            count += n
    
        self.part = tf.constant(part)
        self.idx = idx
    
    def assign_params(self, params_1d):
        params_1d = tf.cast(params_1d, dtype = self.floatx)
        params = tf.dynamic_partition(params_1d, self.part, self.n_tensors)
        for i, (shape, param) in enumerate(zip(self.shapes, params)):
            self.trainable_variables[i].assign(tf.reshape(param, shape))       
    
    def scipy_minimize(self, func):
        init_params = tf.dynamic_stitch(self.idx, self.trainable_variables)
        sopt.minimize(fun = func, 
                        x0 = init_params, 
                        method = self.method,
                        jac = True, options = {'iprint' : 0,
                                                'maxiter': 50000,
                                                'maxfun' : 50000,
                                                'maxcor' : 50,
                                                'maxls': 50,
                                                'gtol': 1.0 * np.finfo(float).eps,
                                                'ftol' : 1.0 * np.finfo(float).eps})
        
    def tfp_minimize(self, func):
        init_params = tf.dynamic_stitch(self.idx, self.trainable_variables)
        tfp.optimizer.lbfgs_minimize(func, 
                                      init_params, 
                                      max_iterations=50000, 
                                      num_correction_pairs=50,
                                      parallel_iterations=1,
                                      max_line_search_iterations=100,
                                      tolerance=1.0 * np.finfo(float).eps)
        
