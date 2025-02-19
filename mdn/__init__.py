"""
A Mixture Density Layer for Keras
cpmpercussion: Charles Martin (University of Oslo) 2018
https://github.com/cpmpercussion/keras-mdn-layer

Hat tip to [Omimo's Keras MDN layer](https://github.com/omimo/Keras-MDN)
for a starting point for this code.

Provided under MIT License
"""
from .version import __version__

import numpy as np

import tensorflow as tf
from tensorflow_probability import distributions as tfd


def elu_plus_one_plus_epsilon(x):
    """ELU activation with a very small addition to help prevent
    NaN in loss."""
    return tf.keras.backend.elu(x) + 1 + tf.keras.backend.epsilon()


def create_mdn_layers(input_layer, output_dim, num_mixes, mu_activation=None):
    mdn_mus = tf.keras.layers.Dense(
        output_dim * num_mixes,
        activation=mu_activation)(input_layer)  # optional activation for means
    mdn_sigmas = tf.keras.layers.Dense(
        output_dim * num_mixes,
        activation=elu_plus_one_plus_epsilon)(input_layer)
    # softmax is applied to pi when sampling, so no need for activation here
    mdn_pi = tf.keras.layers.Dense(num_mixes)(input_layer)
    return tf.keras.layers.concatenate([mdn_mus, mdn_sigmas, mdn_pi])


def get_mixture_loss_func(output_dim, num_mixes):
    """Construct a loss functions for the MDN layer parametrised by number of mixtures."""
    # Construct a loss function with the right number of mixtures and outputs
    def loss_func(y_true, y_pred):
        # Reshape inputs in case this is used in a TimeDistribued layer
        y_pred = tf.reshape(
            y_pred,
            [-1, (2 * num_mixes * output_dim) + num_mixes]
        )
        y_true = tf.reshape(
            y_true,
            [-1, output_dim]
        )
        # Split the inputs into paramaters
        out_mu, out_sigma, out_pi = tf.split(
            y_pred,
            num_or_size_splits=[
                num_mixes * output_dim,
                num_mixes * output_dim,
                num_mixes
            ],
            axis=-1
        )
        # Construct the mixture models
        cat = tfd.Categorical(logits=out_pi)
        component_splits = [output_dim] * num_mixes
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
        coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                in zip(mus, sigs)]
        mixture = tfd.Mixture(cat=cat, components=coll)
        loss = mixture.log_prob(y_true)
        loss = tf.negative(loss)
        loss = tf.reduce_mean(loss)
        return loss

    # Actually return the loss_func
    return loss_func


def get_mixture_sampling_fun(output_dim, num_mixes):
    """Construct a TensorFlor sampling operation for the MDN layer parametrised
    by mixtures and output dimension. This can be used in a Keras model to
    generate samples directly."""

    def sampling_func(y_pred):
        # Reshape inputs in case this is used in a TimeDistribued layer
        y_pred = tf.reshape(
            y_pred,
            [-1, (2 * num_mixes * output_dim) + num_mixes]
        )
        out_mu, out_sigma, out_pi = tf.split(
            y_pred,
            num_or_size_splits=[
                num_mixes * output_dim,
                num_mixes * output_dim,
                num_mixes
            ],
            axis=1
        )
        cat = tfd.Categorical(logits=out_pi)
        component_splits = [output_dim] * num_mixes
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
        coll = [
            tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
            for loc, scale in zip(mus, sigs)
        ]
        mixture = tfd.Mixture(cat=cat, components=coll)
        samp = mixture.sample()
        # Todo: temperature adjustment for sampling function.
        return samp

    # Actually return the loss_func
    return sampling_func


def get_mixture_mse_accuracy(output_dim, num_mixes):
    """Construct an MSE accuracy function for the MDN layer
    that takes one sample and compares to the true value."""
    # Construct a loss function with the right number of mixtures and outputs
    def mse_func(y_true, y_pred):
        # Reshape inputs in case this is used in a TimeDistribued layer
        y_pred = tf.reshape(
            y_pred,
            [-1, (2 * num_mixes * output_dim) + num_mixes]
        )
        y_true = tf.reshape(
            y_true,
            [-1, output_dim]
        )
        out_mu, out_sigma, out_pi = tf.split(
            y_pred,
            num_or_size_splits=[
                num_mixes * output_dim,
                num_mixes * output_dim,
                num_mixes
            ],
            axis=1
        )
        cat = tfd.Categorical(logits=out_pi)
        component_splits = [output_dim] * num_mixes
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
        coll = [
            tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
            for loc, scale in zip(mus, sigs)
        ]
        mixture = tfd.Mixture(cat=cat, components=coll)
        samp = mixture.sample()
        mse = tf.reduce_mean(tf.square(samp - y_true), axis=-1)
        # Todo: temperature adjustment for sampling functon.
        return mse

    # Actually return the loss_func
    return mse_func


def split_mixture_params(params, output_dim, num_mixes):
    """Splits up an array of mixture parameters into mus, sigmas, and pis
    depending on the number of mixtures and output dimension.

    Arguments:
    params -- the parameters of the mixture model
    output_dim -- the dimension of the normal models in the mixture model
    num_mixes -- the number of mixtures represented
    """
    mus = params[:num_mixes * output_dim]
    sigs = params[num_mixes * output_dim:2 * num_mixes * output_dim]
    pi_logits = params[-num_mixes:]
    return mus, sigs, pi_logits


def softmax(w, t=1.0):
    """Softmax function for a list or numpy array of logits. Also adjusts temperature.

    Arguments:
    w -- a list or numpy array of logits

    Keyword arguments:
    t -- the temperature for to adjust the distribution (default 1.0)
    """
    e = np.array(w) / t  # adjust temperature
    e -= e.max()  # subtract max to protect from exploding exp values.
    e = np.exp(e)
    dist = e / np.sum(e)
    return dist


def sample_from_categorical(dist):
    """Samples from a categorical model PDF.

    Arguments:
    dist -- the parameters of the categorical model

    Returns:
    One sample from the categorical model, or -1 if sampling fails.
    """
    r = np.random.rand(1)  # uniform random number in [0,1]
    accumulate = 0
    for i in range(0, dist.size):
        accumulate += dist[i]
        if accumulate >= r:
            return i
    tf.logging.info('Error sampling categorical model.')
    return -1


def sample_from_output(params, output_dim, num_mixes, temp=1.0, sigma_temp=1.0):
    """Sample from an MDN output with temperature adjustment.
    This calculation is done outside of the Keras model using
    Numpy.

    Arguments:
    params -- the parameters of the mixture model
    output_dim -- the dimension of the normal models in the mixture model
    num_mixes -- the number of mixtures represented

    Keyword arguments:
    temp -- the temperature for sampling between mixture components (default 1.0)
    sigma_temp -- the temperature for sampling from the normal distribution (default 1.0)

    Returns:
    One sample from the the mixture model.
    """
    mus, sigs, pi_logits = split_mixture_params(params, output_dim, num_mixes)
    pis = softmax(pi_logits, t=temp)
    m = sample_from_categorical(pis)
    # Alternative way to sample from categorical:
    # m = np.random.choice(range(len(pis)), p=pis)
    mus_vector = mus[m * output_dim:(m + 1) * output_dim]
    sig_vector = sigs[m * output_dim:(m + 1) * output_dim] * sigma_temp  # adjust for temperature
    cov_matrix = np.identity(output_dim) * sig_vector
    sample = np.random.multivariate_normal(mus_vector, cov_matrix, 1)
    return sample
