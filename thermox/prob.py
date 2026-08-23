import jax.numpy as jnp
from jax.lax import fori_loop
from jax import Array, vmap

from thermox.utils import (
    handle_matrix_inputs,
    preprocess_drift_matrix,
    ProcessedDriftMatrix,
    ProcessedDiffusionMatrix,
)
from thermox.sampler import expm_vp, transition_cov_eigh


def log_prob(
    ts: Array,
    xs: Array,
    A: Array | ProcessedDriftMatrix,
    b: Array,
    D: Array | ProcessedDiffusionMatrix,
) -> Array:
    """Calculates log probability of samples from the Ornstein-Uhlenbeck process,
    defined as:

    dx = - A * (x - b) dt + sqrt(D) dW

    by using exact diagonalization.

    Assumes x(t_0) is given deterministically.

    Preprocessing (diagonalisation) costs O(d^3) and evaluation then costs O(T * d^2),
    where T=len(ts), or O(T * d^3) when D^-0.5 @ A @ D^0.5 is not a normal matrix.

    By default, this function does the preprocessing on A and D before the evaluation.
    However, the preprocessing can be done externally using thermox.preprocess
    the output of which can be used as A and D here, this will skip the preprocessing.

    Args:
        ts: Times at which samples are collected. Includes time for x0.
        xs: States of the process.
        A: Drift matrix (Array or thermox.ProcessedDriftMatrix).
            Note: If a thermox.ProcessedDriftMatrix instance is used as input,
            must be the transformed drift matrix, A_y, given by thermox.preprocess,
            not thermox.utils.preprocess_drift_matrix.
        b: Drift displacement vector.
        D: Diffusion matrix (Array or thermox.ProcessedDiffusionMatrix).

    Returns:
        Scalar log probability of given xs.
    """
    A_y, D = handle_matrix_inputs(A, D)

    ys = vmap(jnp.matmul, in_axes=(None, 0))(D.sqrt_inv, xs)
    b_y = D.sqrt_inv @ b
    log_prob_ys = log_prob_identity_diffusion(ts, ys, A_y, b_y)

    D_sqrt_inv_log_det = jnp.log(jnp.linalg.det(D.sqrt_inv))
    return log_prob_ys + D_sqrt_inv_log_det * (len(ts) - 1)


def log_prob_identity_diffusion(
    ts: Array,
    xs: Array,
    A: Array | ProcessedDriftMatrix,
    b: Array,
) -> float:
    if isinstance(A, Array):
        A = preprocess_drift_matrix(A)

    def transition_mean(y, dt):
        return b + expm_vp(A, y - b, dt)

    def logpt(yt, y0, dt):
        diff = yt - transition_mean(y0, dt)

        def mahalanobis_and_log_det(w, U):
            w = jnp.where(w < 1e-20, 1e-20, w)
            diff_val = (U.T @ diff) / jnp.sqrt(w)
            return jnp.dot(diff_val, diff_val), jnp.sum(jnp.log(w))

        # One factorization of the transition covariance per step serves both terms.
        quad, log_det = transition_cov_eigh(A, dt, mahalanobis_and_log_det)
        return -quad / 2 - log_det / 2 - jnp.log(2 * jnp.pi) * (yt.shape[0] / 2)

    log_prob_val = fori_loop(
        1,
        len(ts),
        lambda i, val: val + logpt(xs[i], xs[i - 1], ts[i] - ts[i - 1]),
        0.0,
    )

    return log_prob_val.real
