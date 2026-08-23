"""Exactness checks against references (Van Loan, Lyapunov) that do not go
through thermox: drifts whose transformed form D^{-1/2} A D^{1/2} is not normal,
plus normal cases whose results must not change.
"""

import jax
import jax.numpy as jnp
import pytest

import thermox
from thermox.utils import preprocess_drift_matrix

jax.config.update("jax_enable_x64", True)

A_SYM = jnp.array([[3.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 5.0]])
A_TRI = jnp.array([[2.0, 1.5, 0.0], [0.0, 3.0, 1.5], [0.0, 0.0, 4.0]])
A_ROT = jnp.array([[1.0, 2.0], [-2.0, 1.0]])
D_DIAG = jnp.diag(jnp.array([1.0, 4.0, 9.0]))
D_DENSE = jnp.array([[1.0, 0.3, -0.1], [0.3, 1.0, 0.2], [-0.1, 0.2, 1.0]])

NONNORMAL_CASES = [
    pytest.param(A_SYM, D_DIAG, id="symmetric-A-diagonal-D"),
    pytest.param(A_SYM, D_DENSE, id="symmetric-A-dense-D"),
    pytest.param(A_TRI, jnp.eye(3), id="triangular-A-identity-D"),
]
NORMAL_CASES = [
    pytest.param(A_SYM, jnp.eye(3), id="symmetric-A-identity-D"),
    pytest.param(A_ROT, jnp.eye(2), id="rotation-A-identity-D"),
]


def van_loan_covariance(A, D, t):
    """int_0^t exp(-A s) D exp(-A^T s) ds via Van Loan (1978)."""
    d = A.shape[0]
    M = jnp.block([[-A, D], [jnp.zeros((d, d)), A.T]]) * t
    F = jax.scipy.linalg.expm(M)
    return F[:d, d:] @ jax.scipy.linalg.expm(-A.T * t)


def lyapunov_covariance(A, D):
    """Solve A S + S A^T = D by Kronecker vectorization (small d)."""
    d = A.shape[0]
    eye = jnp.eye(d)
    K = jnp.kron(eye, A) + jnp.kron(A, eye)
    return jnp.linalg.solve(K, D.reshape(-1, order="F")).reshape(d, d, order="F")


def reference_covariance(A, D, t):
    """Sigma_t = Sigma_inf - exp(-A t) Sigma_inf exp(-A^T t), valid for stable A.

    Numerically benign for large t, unlike the block exponential.
    """
    S_inf = lyapunov_covariance(A, D)
    E = jax.scipy.linalg.expm(-A * t)
    return S_inf - E @ S_inf @ E.T


def relerr(x, y):
    return jnp.linalg.norm(x - y) / jnp.linalg.norm(y)


@pytest.mark.parametrize("A,D", NONNORMAL_CASES + NORMAL_CASES)
def test_references_agree(A, D):
    # Guard the references themselves: two independent formulas for Sigma_t.
    assert (
        relerr(van_loan_covariance(A, D, 0.7), reference_covariance(A, D, 0.7)) < 1e-10
    )


@pytest.mark.parametrize("A,D", NONNORMAL_CASES + NORMAL_CASES)
@pytest.mark.parametrize("t", [0.05, 0.7, 5.0])
def test_conditional_covariance_matches_reference(A, D, t):
    cov = thermox.conditional.covariance(t, A, D)
    assert relerr(cov, reference_covariance(A, D, t)) < 1e-8


@pytest.mark.parametrize("A,D", NONNORMAL_CASES + NORMAL_CASES)
def test_stationary_covariance_matches_lyapunov(A, D):
    lam_min = jnp.min(jnp.linalg.eigvals(A).real)
    cov = thermox.conditional.covariance(60.0 / lam_min, A, D)
    assert relerr(cov, lyapunov_covariance(A, D)) < 1e-8


def reference_log_prob(ts, xs, A, b, D):
    """Sum of transition log-densities built from the Van Loan covariance.

    Independent of thermox and differentiable (jax.scipy.linalg.expm has a JVP).
    """

    def transition_logpdf(x1, x0, dt):
        mean = b + jax.scipy.linalg.expm(-A * dt) @ (x0 - b)
        cov = van_loan_covariance(A, D, dt)
        return jax.scipy.stats.multivariate_normal.logpdf(x1, mean, cov)

    return sum(
        transition_logpdf(xs[i], xs[i - 1], ts[i] - ts[i - 1])
        for i in range(1, len(ts))
    )


@pytest.mark.parametrize("A,D", NONNORMAL_CASES)
def test_log_prob_matches_reference_gaussian(A, D):
    d = A.shape[0]
    b = jnp.arange(1.0, d + 1.0)
    ts = jnp.array([0.0, 0.1, 0.5, 0.6, 1.4])
    xs = jax.random.normal(jax.random.PRNGKey(3), (len(ts), d))
    ref = reference_log_prob(ts, xs, A, b, D)
    assert jnp.isclose(thermox.log_prob(ts, xs, A, b, D), ref, rtol=1e-8)


@pytest.mark.parametrize("A,D", NONNORMAL_CASES)
def test_log_prob_grad_wrt_drift_matches_reference(A, D):
    d = A.shape[0]
    b = jnp.arange(1.0, d + 1.0)
    ts = jnp.array([0.0, 0.1, 0.5, 0.6, 1.4])
    xs = jax.random.normal(jax.random.PRNGKey(3), (len(ts), d))
    g = jax.grad(lambda A: thermox.log_prob(ts, xs, A, b, D))(A)
    g_ref = jax.grad(lambda A: reference_log_prob(ts, xs, A, b, D))(A)
    assert relerr(g, g_ref) < 1e-8


def test_log_prob_grad_symmetric_parametrization_matches_reference():
    # A = B B^T with D = I stays on the normal branch; gradients w.r.t. B are exact
    # there. (At an exactly normal A the derivative w.r.t. A itself in directions
    # that break normality is that of the symmetric-part formula -- see
    # thermox.sampler.transition_cov_eigh.)
    A, D = A_SYM, jnp.eye(3)
    B = jnp.linalg.cholesky(A)
    b = jnp.arange(1.0, 4.0)
    ts = jnp.array([0.0, 0.1, 0.5, 0.6, 1.4])
    xs = jax.random.normal(jax.random.PRNGKey(3), (len(ts), 3))
    g = jax.grad(lambda B: thermox.log_prob(ts, xs, B @ B.T, b, D))(B)
    g_ref = jax.grad(lambda B: reference_log_prob(ts, xs, B @ B.T, b, D))(B)
    assert relerr(g, g_ref) < 1e-8


def test_sample_covariance_matches_lyapunov_for_anisotropic_noise():
    A, D = A_SYM, D_DIAG
    ts = jnp.arange(0.0, 20000.0, 0.5)
    xs = thermox.sample(jax.random.PRNGKey(0), ts, jnp.zeros(3), A, jnp.zeros(3), D)
    emp = jnp.cov(xs[2000:].T)
    # Monte Carlo error of the sample covariance is ~1e-2 here; the symmetric-part
    # formula gives 0.13.
    assert relerr(emp, lyapunov_covariance(A, D)) < 0.05


@pytest.mark.parametrize("A,D", NORMAL_CASES)
def test_normal_case_equals_symmetric_part_formula(A, D):
    # For a normal transformed drift the symmetric-part formula is exact; the
    # general formula must reproduce it, so nothing changes for existing users.
    t = 0.7
    A_y, PD = thermox.preprocess(A, D)
    sym_eigvals, sym_eigvecs = jnp.linalg.eigh(0.5 * (A_y.val + A_y.val.T))
    old = sym_eigvecs @ jnp.diag(
        (1 - jnp.exp(-2 * sym_eigvals * t)) / (2 * sym_eigvals)
    )
    old = PD.sqrt @ old @ sym_eigvecs.T @ PD.sqrt.T
    assert relerr(thermox.conditional.covariance(t, A, D), old) < 1e-12


@pytest.mark.parametrize("A,D", NONNORMAL_CASES + NORMAL_CASES)
@pytest.mark.parametrize("t", [0.0, 0.7])
def test_transition_cov_eigh_factorizes_transition_cov(A, D, t):
    # (w, U) is the one object sample and log_prob read the covariance through;
    # both branches must return an orthonormal spectral factorization of
    # transition_cov, including at t = 0 (the linalg grids contain a zero step).
    from thermox.sampler import transition_cov, transition_cov_eigh

    A_y, _ = thermox.preprocess(A, D)
    cov = transition_cov(A_y, t)
    w, U = transition_cov_eigh(A_y, t)
    assert jnp.all(w > -1e-12)
    assert jnp.linalg.norm(U.T @ U - jnp.eye(len(w))) < 1e-10
    scale = max(1.0, float(jnp.linalg.norm(cov)))
    assert jnp.linalg.norm(U @ jnp.diag(w) @ U.T - cov) < 1e-10 * scale
    if bool(A_y.is_normal):
        # normal branch: precomputed eigenbasis of (A_y + A_y^T)/2, no per-step eigh
        assert jnp.array_equal(U, A_y.sym_eigvecs)
    # apply is evaluated inside the branch (so the normal branch never
    # materializes U per step)
    trace = transition_cov_eigh(A_y, t, lambda w, U: jnp.sum(w))
    assert jnp.isclose(trace, jnp.trace(cov), rtol=1e-10, atol=1e-14)


def test_preprocess_flags_normality():
    assert bool(preprocess_drift_matrix(A_SYM).is_normal)
    assert bool(preprocess_drift_matrix(A_ROT).is_normal)
    assert not bool(preprocess_drift_matrix(A_TRI).is_normal)
    # symmetric A becomes non-normal after transforming with anisotropic D
    A_y, _ = thermox.preprocess(A_SYM, D_DIAG)
    assert not bool(A_y.is_normal)
