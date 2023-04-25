from config import RND_SEED
import numpy as np
# setting a seed for the random state for reproducibility
np.random.seed(RND_SEED)
import tensorflow as tf
import tensorflow_probability as tfp
import scipy.stats as st
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8,6)
import multiprocessing
PROCESSORS = multiprocessing.cpu_count()
import os

import platform
if platform.processor() == 'arm':
    import logging
    # mac config
    tf.config.set_visible_devices([], 'GPU')
    tf.get_logger().setLevel(logging.ERROR)

# ml utils
def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)

def common_prior(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Normal(loc=t, scale=1))
    ])

def common_posterior(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    z = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2*n, dtype=dtype),
        tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.Normal(loc=t[..., :n],
                                    scale=1e-3 + 0.1*tf.nn.softplus(z + t[..., n:])
                                    )
                )
    ])

# plotting utils
def save_xy_graph(fig, x, y, figures_dir, filename, legend=True):
    plt.xlabel(x)
    plt.ylabel(y)
    if legend:
        plt.legend()
    plt.show()
    fig.savefig(figures_dir + filename, bbox_inches='tight')

def save_multiplot_graph(fig, x, y, figures_dir, filename, labels=[]):
    if labels:
        artists = (fig.legend(labels, loc='lower center', ncol=len(labels), bbox_transform=fig.transFigure, bbox_to_anchor=[0.5, -0.1]),)
    else:
        artists = ()
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
    fig.savefig(figures_dir + filename, bbox_extra_artists=artists, bbox_inches='tight')

def get_confidence_axis():
    return np.arange(0,101,5)

def get_confidence_area(percentages):
    confidences = get_confidence_axis()
    return np.trapz(np.abs(np.subtract(confidences, percentages)), confidences)

def unc_calibration(y, y_hat, sd_hat):
    # for each confidence level calculate the amount of points in CI
    confidences = get_confidence_axis()
    in_range = []
    for confidence in confidences:
        intervals = [st.norm.interval(confidence/100, loc=mean, scale=sd) for mean, sd in zip(y_hat, sd_hat)]
        points_in_range = [true >= ci[0] and true <= ci[1] for true, ci in zip(y, intervals)]
        in_range.append(sum(points_in_range)/len(points_in_range) * 100)
    return in_range