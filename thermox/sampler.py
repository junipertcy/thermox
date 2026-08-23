from functools import partial
import jax
import jax.numpy as jnp
from jax import Array

from thermox.utils import (
    handle_matrix_inputs,
    preprocess_drift_matrix,
    ProcessedDriftMatrix,
    ProcessedDiffusionMatrix,
)


def sample(
    key: Array,
    ts: Array,
    x0: Array,
    A: Array | ProcessedDriftMatrix,
    b: Array,
    D: Array | ProcessedDiffusionMatrix,
    associative_scan: bool = True,
) -> Array:
    """Collects samples from the Ornstein-Uhlenbeck process, defined as:

    dx = - A * (x - b) dt + sqrt(D) dW

    by using exact diagonalization.

    Preprocessing (diagonalization) costs O(d^3) and sampling costs O(T * d^2),
    where T=len(ts), or O(T * d^3) when D^-0.5 @ A @ D^0.5 is not a normal matrix.

    If associative_scan=True then jax.lax.associative_scan is used which will run in
    time O((T/p + log(T)) * d^2) on a GPU/TPU with p cores, still with
    O(d^3) preprocessing.

    By default, this function does the preprocessing on A and D before the evaluation.
    However, the preprocessing can be done externally using thermox.preprocess
    the output of which can be used as A and D here, this will skip the preprocessing.

    Args:
        key: Jax PRNGKey.
        ts: Times at which samples are collected. Includes time for x0.
        x0: Initial state of the process.
        A: Drift matrix (Array or thermox.ProcessedDriftMatrix).
            Note: If a thermox.ProcessedDriftMatrix instance is used as input,
            must be the transformed drift matrix, A_y, given by thermox.preprocess,
            not thermox.utils.preprocess_drift_matrix.
        b: Drift displacement vector.
        D: Diffusion matrix (Array or thermox.ProcessedDiffusionMatrix).
        associative_scan: If True, uses jax.lax.associative_scan.

    Returns:
        Array-like, desired samples.
            shape: (len(ts), ) + x0.shape
    """
    A_y, D = handle_matrix_inputs(A, D)

    y0 = D.sqrt_inv @ x0
    b_y = D.sqrt_inv @ b
    ys = sample_identity_diffusion(key, ts, y0, A_y, b_y, associative_scan)
    return jax.vmap(jnp.matmul, in_axes=(None, 0))(D.sqrt, ys)


def sample_identity_diffusion(
    key: Array,
    ts: Array,
    x0: Array,
    A: Array | ProcessedDriftMatrix,
    b: Array,
    associative_scan: bool = True,
) -> Array:
    if associative_scan:
        return _sample_identity_diffusion_associative_scan(key, ts, x0, A, b)
    else:
        return _sample_identity_diffusion_scan(key, ts, x0, A, b)


def expm_vp(A, v, dt):
    out = A.eigvecs_inv @ v
    out = jnp.exp(-A.eigvals * dt) * out
    out = A.eigvecs @ out
    return out.real


def transition_cov(A, dt):
    """Covariance of x_dt given x_0 for dx = -A x dt + dW, i.e.
    int_0^dt exp(-A s) exp(-A^T s) ds, computed in the eigenbasis of A.
    Exact for any stable, diagonalizable A.
    """
    eigvals_sum = A.eigvals[:, None] + A.eigvals.conj()[None, :]
    integral = -jnp.expm1(-eigvals_sum * dt) / eigvals_sum
    cov = A.eigvecs @ (A.noise_cov_eigbasis * integral) @ A.eigvecs.conj().T
    cov = cov.real
    return 0.5 * (cov + cov.T)


def transition_cov_eigh(A, dt, apply=lambda w, U: (w, U)):
    """Spectral factorization transition_cov(A, dt) = U diag(w) U^T, returned as
    apply(w, U).

    Branches on A.is_normal. Normal A: U = A.sym_eigvecs is precomputed and w is
    a closed-form function of dt, O(d^2) per step. Otherwise transition_cov(A, dt)
    is eigendecomposed at each step, O(d^3). apply is evaluated inside the branch
    so that only its result, not a d x d matrix per step, leaves the lax.cond.

    Gradients with respect to A follow the branch taken: at an exactly normal A
    they are those of the normal-branch formula, which depends on A only through
    (A + A^T)/2.
    """

    def normal(A, dt):
        w = (1 - jnp.exp(-2 * A.sym_eigvals * dt)) / (2 * A.sym_eigvals)
        return apply(w, A.sym_eigvecs)

    def general(A, dt):
        # eigh rather than Cholesky: stays well defined at dt = 0 (zero covariance).
        w, U = jnp.linalg.eigh(transition_cov(A, dt))
        return apply(w, U)

    # apply is evaluated inside the branches so the cond returns a small result;
    # returning (w, U) and applying it outside made the associative-scan path
    # measurably slower (vmap's cond batching rule broadcasts the d x d U over all
    # steps).
    return jax.lax.cond(A.is_normal, normal, general, A, dt)


def transition_cov_sqrt_vp(A, v, dt):
    return transition_cov_eigh(
        A, dt, lambda w, U: U @ (jnp.sqrt(jnp.maximum(w, 0.0)) * v)
    )


def _sample_identity_diffusion_scan(
    key: Array,
    ts: Array,
    x0: Array,
    A: Array | ProcessedDriftMatrix,
    b: Array,
) -> Array:
    if isinstance(A, Array):
        A = preprocess_drift_matrix(A)

    def transition_mean(x, dt):
        return b + expm_vp(A, x - b, dt)

    def next_x(x, dt, rv):
        return transition_mean(x, dt) + transition_cov_sqrt_vp(A, rv, dt)

    def scan_body(carry, dt_and_rv):
        x = carry
        dt, rv = dt_and_rv
        new_x = next_x(x, dt, rv)
        return new_x, new_x

    dts = jnp.diff(ts)
    gauss_samps = jax.random.normal(key, (len(dts),) + x0.shape)

    # Stack dts and gauss_samps along a new axis
    dt_and_rv = (dts, gauss_samps)

    _, xs = jax.lax.scan(scan_body, x0, dt_and_rv)
    xs = jnp.concatenate([jnp.expand_dims(x0, axis=0), xs], axis=0)
    return xs


def _sample_identity_diffusion_associative_scan(
    key: Array,
    ts: Array,
    x0: Array,
    A: Array | ProcessedDriftMatrix,
    b: Array,
) -> Array:
    if isinstance(A, Array):
        A = preprocess_drift_matrix(A)

    dts = jnp.diff(ts)

    # transition_mean(x, dt) = b + expm_vp(A, x - b, dt)

    gauss_samps = jax.random.normal(key, (len(dts),) + x0.shape)
    noise_terms = jax.vmap(lambda v, dt: transition_cov_sqrt_vp(A, v, dt))(
        gauss_samps, dts
    )

    @partial(jax.vmap, in_axes=(0, 0))
    def binary_associative_operator(elem_a, elem_b):
        t_a, x_a = elem_a
        t_b, x_b = elem_b
        return t_a + t_b, expm_vp(A, x_a, t_b) + x_b

    scan_times = jnp.concatenate([ts[:1], dts], dtype=float)  # [t0, dt1, dt2, ...]
    scan_input_values = jnp.concatenate(
        [x0[None] - b, noise_terms], axis=0
    )  # Shift input by b
    scan_elems = (scan_times, scan_input_values)

    scan_output = jax.lax.associative_scan(binary_associative_operator, scan_elems)
    return scan_output[1] + b  # Shift back by b
