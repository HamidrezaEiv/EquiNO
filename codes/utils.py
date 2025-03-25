import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.stats import laplace
from matplotlib.ticker import FuncFormatter
import cmcrameri.cm as cmc

class PreProcessing():
    def __init__(self, data_dir = '../data/', dtype=np.float32):
        super(PreProcessing, self).__init__()
        self.data_dir = data_dir
        self.dtype = dtype
    def gen_e(self, N, method='laplace'):
        if method == 'lhs':
            e_min = -0.04
            e_max = 0.04
            
            lhs_samples = lhs(3, N, criterion = None)
            lb = np.array([e_min, e_min, e_min])
            ub = np.array([e_max, e_max, e_max])
            e = lb + (ub-lb) * lhs_samples
        elif method == 'laplace':
            lhs_samples = lhs(3, N, criterion = None)
            e = laplace.ppf(lhs_samples, loc=0, scale=0.01)
        
            ub = e > 0.04
            e[ub] = 0.08 - e[ub]
            
            lb = e < -0.04
            e[lb] = 0.08 + e[lb]
        else:
            print('Sampling method not implemented!')
        
        return e
    
    def get_data_global(self):
        train = joblib.load(self.data_dir + 'train_global.pkl')
        val = joblib.load(self.data_dir + 'val_global.pkl')
        test = joblib.load(self.data_dir + 'test_global.pkl')
        return (train, val, test)
    
    def get_data_local(self):
        train = joblib.load(self.data_dir + 'train_local.pkl')
        val = joblib.load(self.data_dir + 'val_local.pkl')
        test = joblib.load(self.data_dir + 'test_local.pkl')
        return (train, val, test)
    
    def get_grid(self, gdir):
        nodes = pd.read_csv(gdir + 'nodeCoordinates.csv',header=None).to_numpy().astype(self.dtype)
        ips = joblib.load(gdir + 'ips').astype(self.dtype)
        return nodes, ips

    def get_sample(self, data_global, data_local, i=0):
        e_hat, s_hat, c_hat = [x[i:i+1] for x in data_global]
        d, e, s = [x[i:i+1] for x in data_local]
        return (e_hat, s_hat, c_hat), (d, e, s)
    
    def get_gmatid(self, gdir):
        elemInc = pd.read_csv(gdir + 'elementIncidences.csv',header=None).to_numpy()
        nnode = elemInc.max()
        gmatid = np.zeros((nnode,), dtype=self.dtype)
        for i in range(len(elemInc)):
            inc = elemInc[i, 2:]
            for j in inc:
                if gmatid[j-1] == 0:
                    gmatid[j-1] = elemInc[i, 1]
                if gmatid[j-1] != elemInc[i, 1]:
                    gmatid[j-1] = 1.5
                
        return (gmatid[...,None] - 1.5)*2.0
    

def plot_strain_stress(ips, strain=None, stress=None):
    if strain is not None:
        fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    
        c0 = ax[0].tricontourf(ips[:, 0], ips[:, 1], strain[-1, :, 0], levels=50)
        c1 = ax[1].tricontourf(ips[:, 0], ips[:, 1], strain[-1, :, 1], levels=50)
        c2 = ax[2].tricontourf(ips[:, 0], ips[:, 1], strain[-1, :, 2], levels=50)
    
        plt.colorbar(c0, ax = ax[0], shrink=0.8)
        plt.colorbar(c1, ax = ax[1], shrink=0.8)
        plt.colorbar(c2, ax = ax[2], shrink=0.8)
    
        for axx in ax:
            axx.set_aspect('equal')
            axx.set_axis_off()
    
    if stress is not None:
        fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    
        c0 = ax[0].tricontourf(ips[:, 0], ips[:, 1], stress[-1, :, 0], levels=50)
        c1 = ax[1].tricontourf(ips[:, 0], ips[:, 1], stress[-1, :, 1], levels=50)
        c2 = ax[2].tricontourf(ips[:, 0], ips[:, 1], stress[-1, :, 2], levels=50)
    
        plt.colorbar(c0, ax = ax[0], shrink=0.8)
        plt.colorbar(c1, ax = ax[1], shrink=0.8)
        plt.colorbar(c2, ax = ax[2], shrink=0.8)
    
        for axx in ax:
            axx.set_aspect('equal')
            axx.set_axis_off()
        
def plot_disp(nodes, disp):
        
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))

    c0 = ax[0].tricontourf(nodes[:, 0], nodes[:, 1], disp[-1, :, 0], levels=15)
    c1 = ax[1].tricontourf(nodes[:, 0], nodes[:, 1], disp[-1, :, 1], levels=15)

    plt.colorbar(c0, ax = ax[0], shrink=0.8)
    plt.colorbar(c1, ax = ax[1], shrink=0.8)

    for axx in ax:
        axx.set_aspect('equal')
        axx.set_axis_off()

def l2error(ref, model):
    return np.linalg.norm(model - ref)/np.linalg.norm(ref) * 100

def plot_compare_errors(ref, pred1, pred2, nodes, ips, save = False, name='rve'):
    titles = ['$u_x$', '$u_y$', '$\sigma_{xx}$', '$\sigma_{yy}$', '$\sigma_{xy}$']
    fig, ax = plt.subplots(5, 5, sharex = True, sharey = True, figsize = (11, 8), gridspec_kw={'hspace': 0.5, 'wspace': 0.2})
    plt.set_cmap('cmc.berlin')
    
    SMALL_SIZE = 14
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 20
    try:
        plt.rc('text', usetex = True)
    except:
        print('No Tex!')
    plt.rc('font', size=SMALL_SIZE,family='Ubuntu')          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE,linewidth=1)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    for v in range(5):

        z = ref[v]
        zp1 = pred1[v]
        err1 = np.abs(z - zp1)
        l2err1 = l2error(z, zp1)
        
        zp2 = pred2[v]
        err2 = np.abs(z - zp2)
        l2err2 = l2error(z, zp2)
        
        def percent_formatter(x, pos):
            return f"{x:.0e}"
        
        if v < 2:
            mesh = nodes
            fmt = FuncFormatter(percent_formatter)
        else:
            mesh = ips
            fmt = '%.1f'
        
        l = 20
        c0 = ax[0, v].tricontourf(mesh[:, 0], mesh[:, 1], z, levels = l)
        c1 = ax[1, v].tricontourf(mesh[:, 0], mesh[:, 1], zp2, levels = l)
        c2 = ax[2, v].tricontourf(mesh[:, 0], mesh[:, 1], err2, levels = l)
        c3 = ax[3, v].tricontourf(mesh[:, 0], mesh[:, 1], zp1, levels = l)
        c4 = ax[4, v].tricontourf(mesh[:, 0], mesh[:, 1], err1, levels = l)
        

        for j, c in enumerate([c0, c1, c2, c3, c4]):
            
            cb = fig.colorbar(c, ax = ax[j, v], format = fmt, orientation = 'vertical', shrink = 1)
            cb.ax.yaxis.set_major_locator(plt.LinearLocator(numticks=3))
            cb.ax.set_position([ax[j, v].get_position().x1+0.01,ax[j, v].get_position().y0+0.02,0.02,ax[j, v].get_position().y1-ax[j, v].get_position().y0-0.04])
            
            # cb.ax.set_title(self.var_names_ref[v], pad = 20)
        
        ax[0, v].set_title(titles[v], pad = 10)
        ax[2, v].set_title(f' ({np.round(l2err2,2)}\%)', pad = 10)
        ax[4, v].set_title(f' ({np.round(l2err1,2)}\%)', pad = 10)
        
    for axx in ax.flatten():
        axx.set_aspect('equal')
        axx.set_xticks([])
        axx.set_yticks([])
    
    if save:
        plt.savefig(f'../figs/con_compare_{name}.png', dpi = 300, bbox_inches = 'tight')