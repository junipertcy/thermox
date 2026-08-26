"""Exactness checks against references (Van Loan, Lyapunov) that do not go
through thermox: drifts whose transformed form D^{-1/2} A D^{1/2} is not normal,
plus normal cases whose results must not change.
"""

import jax
import jax.numpy as jnp
import pytest

import thermox
from thermox.sampler import _scan_linear_recurrence, uniform_dt
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


GRIDS = [
    pytest.param(
        jnp.array([0.0, 0.1, 0.5, 0.6, 1.4]), id="non-uniform"
    ),  # per-step path
    pytest.param(jnp.arange(0.0, 1.5, 0.1), id="uniform"),  # factor-once path
]


@pytest.mark.parametrize("ts", GRIDS)
@pytest.mark.parametrize("A,D", NONNORMAL_CASES)
def test_log_prob_matches_reference_gaussian(A, D, ts):
    d = A.shape[0]
    b = jnp.arange(1.0, d + 1.0)
    xs = jax.random.normal(jax.random.PRNGKey(3), (len(ts), d))
    ref = reference_log_prob(ts, xs, A, b, D)
    assert jnp.isclose(thermox.log_prob(ts, xs, A, b, D), ref, rtol=1e-8)


@pytest.mark.parametrize("ts", GRIDS)
@pytest.mark.parametrize("A,D", NONNORMAL_CASES)
def test_log_prob_grad_wrt_drift_matches_reference(A, D, ts):
    d = A.shape[0]
    b = jnp.arange(1.0, d + 1.0)
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


@pytest.mark.parametrize("jitter", [0.0, 0.5], ids=["uniform-grid", "jittered-grid"])
def test_sample_covariance_matches_lyapunov_for_anisotropic_noise(jitter):
    A, D = A_SYM, D_DIAG
    ts = jnp.arange(0.0, 20000.0, 0.5)
    ts = jnp.sort(ts + jitter * jax.random.uniform(jax.random.PRNGKey(1), ts.shape))
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


def ill_conditioned_eigenvectors_case():
    # d = 12, diag(1..3) plus 6 x a strictly upper triangular Gaussian: eigenvector
    # condition number ~1e9, where a formula in the eigenbasis of A loses everything.
    d = 12
    upper = jnp.triu(jax.random.normal(jax.random.PRNGKey(0), (d, d)), 1)
    return jnp.diag(jnp.linspace(1.0, 3.0, d)) + 6.0 * upper


@pytest.mark.parametrize("t", [0.01, 0.3])
def test_covariance_accurate_for_ill_conditioned_eigenvectors(t):
    A = ill_conditioned_eigenvectors_case()
    D = jnp.eye(A.shape[0])
    assert jnp.linalg.cond(jnp.linalg.eig(A)[1]) > 1e8
    cov = thermox.conditional.covariance(t, A, D)
    assert relerr(cov, van_loan_covariance(A, D, t)) < 1e-12


def test_covariance_is_zero_at_t_zero():
    A = ill_conditioned_eigenvectors_case()
    cov = thermox.conditional.covariance(0.0, A, jnp.eye(A.shape[0]))
    assert jnp.all(cov == 0.0)


def linalg_grid(burnin, num_samples=100, dt=0.1):
    # The grid thermox.linalg builds: x0 at time 0, one gap of burnin * dt, then dt.
    ts = jnp.arange(burnin, burnin + num_samples + 1) * dt
    return jnp.concatenate([jnp.array([0]), ts])


@pytest.mark.parametrize(
    "ts",
    [
        pytest.param(jnp.arange(0, 1, 0.01), id="readme-arange"),
        pytest.param(linalg_grid(0), id="linalg-burnin-0"),
        pytest.param(linalg_grid(1), id="linalg-burnin-1"),
        pytest.param(linalg_grid(5), id="linalg-burnin-5"),
        pytest.param((jnp.arange(0, 10001) * 0.1).astype(jnp.float32), id="float32"),
        pytest.param(jnp.linspace(0, 100, 300), id="linspace"),
    ],
)
def test_uniform_dt_accepts_grids_uniform_up_to_rounding(ts):
    is_uniform, dt = uniform_dt(ts)
    assert bool(is_uniform)
    assert jnp.isclose(dt, ts[2] - ts[1], rtol=1e-6)


def test_uniform_dt_rejects_jittered_grid():
    ts = jnp.arange(0, 100, 0.1)
    ts = jnp.sort(ts + jax.random.uniform(jax.random.PRNGKey(0), ts.shape) * 0.1)
    assert not bool(uniform_dt(ts)[0])


def contracting_matrix(key, d=4):
    return jax.scipy.linalg.expm(
        -(jax.random.normal(key, (d, d)) / d**0.5 + 3 * jnp.eye(d))
    )


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 1000, 1001])
def test_scan_linear_recurrence_matches_sequential_scan(n):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(n), 3)
    E = contracting_matrix(k1)
    y0 = jax.random.normal(k2, (4,))
    u = jax.random.normal(k3, (n, 4))
    _, ys = jax.lax.scan(lambda y, u_k: (E @ y + u_k,) * 2, y0, u)
    assert jnp.allclose(_scan_linear_recurrence(E, y0, u), ys, rtol=1e-12, atol=1e-12)


def test_uniform_grid_engines_agree_for_ill_conditioned_eigenvectors():
    # On a uniform grid both engines apply one exp(-A dt); propagated through the
    # eigenbasis of A (per-step path) they disagree at ~1e-10 for this family.
    A = ill_conditioned_eigenvectors_case()
    d = A.shape[0]
    ts = jnp.arange(0.0, 1.0, 0.05)
    key = jax.random.PRNGKey(0)
    x0, b, D = jnp.ones(d), jnp.zeros(d), jnp.eye(d)
    xa = thermox.sample(key, ts, x0, A, b, D, associative_scan=True)
    xs = thermox.sample(key, ts, x0, A, b, D, associative_scan=False)
    assert relerr(xa, xs) < 1e-12


def test_log_prob_on_uniform_grid_exact_for_ill_conditioned_eigenvectors():
    A = ill_conditioned_eigenvectors_case()
    d = A.shape[0]
    ts = jnp.arange(0.0, 1.0, 0.05)
    x0, b, D = jnp.ones(d), jnp.zeros(d), jnp.eye(d)
    xs = thermox.sample(jax.random.PRNGKey(0), ts, x0, A, b, D)
    lp = thermox.log_prob(ts, xs, A, b, D)
    ref = reference_log_prob(ts, xs, A, b, D)
    assert jnp.abs(lp - ref) / jnp.abs(ref) < 1e-11


def test_linalg_expm_of_nonsymmetric_matrix():
    # expnegm's whitened drift is non-normal for a non-symmetric input: on
    # upstream main this estimate is off by 0.6; Monte Carlo noise is ~0.01.
    M = jnp.array([[-1.0, 3.0], [0.0, -2.0]])
    est = thermox.linalg.expm(M, num_samples=100000, dt=0.1, burnin=0, alpha=1.0)
    assert jnp.allclose(est, jax.scipy.linalg.expm(M), atol=1e-1)
