from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import LevenbergMarquardt

jax.config.update("jax_enable_x64", True)


def infer_Rbound_batched(
    L0: np.ndarray,  # n_samples
    KxStar: np.ndarray,  # n_samples
    Rtot: np.ndarray,  # n_samples x n_R
    Cplx: np.ndarray,  # n_samples x n_cplx x n_L
    Ctheta: np.ndarray,  # n_samples x n_cplx
    Ka: np.ndarray,  # n_samples x n_L x n_R
) -> np.ndarray:
    validate_inputs(L0, KxStar, Rtot, Cplx, Ctheta, Ka)
    return np.array(
        infer_Rbound_batched_jax(
            jnp.array(L0, dtype=jnp.float64),
            jnp.array(KxStar, dtype=jnp.float64),
            jnp.array(Rtot, dtype=jnp.float64),
            jnp.array(Cplx, dtype=jnp.float64),
            jnp.array(Ctheta, dtype=jnp.float64),
            jnp.array(Ka, dtype=jnp.float64),
        )
    )


@jax.jit
def infer_Rbound_batched_jax(
    L0: jnp.ndarray,  # n_samples
    KxStar: jnp.ndarray,  # n_samples
    Rtot: jnp.ndarray,  # n_samples x n_R
    Cplx: jnp.ndarray,  # n_samples x n_cplx x n_L
    Ctheta: jnp.ndarray,  # n_samples x n_cplx
    Ka: jnp.ndarray,  # n_samples x n_L x n_R
) -> jnp.ndarray:
    def process_sample(i):
        return infer_Req(Rtot[i], L0[i], KxStar[i], Cplx[i], Ctheta[i], Ka[i])

    Req, losses = jax.vmap(process_sample)(jnp.arange(Ka.shape[0]))

    _ = jax.lax.cond(
        jnp.any(losses > 1e-3),
        lambda _: jax.debug.print(
            "Losses exceeding threshold: {}", jnp.sum(losses > 1e-3)
        ),
        lambda _: None,
        operand=None,
    )

    return Rtot - Req


def infer_Req(
    Rtot: jnp.ndarray,
    L0: jnp.ndarray,
    KxStar: jnp.ndarray,
    Cplx: jnp.ndarray,
    Ctheta: jnp.ndarray,
    Ka: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    L0_Ctheta_KxStar = L0 * jnp.sum(Ctheta) / KxStar
    Ka_KxStar = Ka * KxStar
    Cplxsum = Cplx.sum(axis=0)

    def residual_log(log_Req: jnp.ndarray) -> jnp.ndarray:
        Req = jnp.exp(log_Req)
        Rbound = infer_Rbound_from_Req(Req, Cplxsum, L0_Ctheta_KxStar, Ka_KxStar)
        return jnp.log(Rtot) - jnp.log(Req + Rbound)

    solver = LevenbergMarquardt(
        residual_log,
        damping_parameter=5e-4,
        maxiter=75,
        tol=1e-14,
        xtol=1e-14,
        gtol=1e-14,
        implicit_diff=False,
        jit=True,
    )
    solution = solver.run(jnp.log(Rtot / 100.0))
    Req_opt = jnp.exp(solution.params)

    loss = solution.state.value**2
    return Req_opt, loss


def infer_Rbound_from_Req(
    Req: jnp.ndarray,
    Cplxsum: jnp.ndarray,
    L0_Ctheta_KxStar: jnp.ndarray,
    Ka_KxStar: jnp.ndarray,
) -> jnp.ndarray:
    Psi = Req * Ka_KxStar
    Psirs = Psi.sum(axis=1) + 1
    Psinorm = Psi / Psirs[:, None]
    return L0_Ctheta_KxStar * jnp.prod(Psirs**Cplxsum) * jnp.dot(Cplxsum, Psinorm)


def validate_inputs(
    L0: np.ndarray,
    KxStar: np.ndarray,
    Rtot: np.ndarray,
    Cplx: np.ndarray,
    Ctheta: np.ndarray,
    Ka: np.ndarray,
) -> None:
    assert L0.dtype == np.float64
    assert KxStar.dtype == np.float64
    assert Rtot.dtype == np.float64
    assert Ka.dtype == np.float64
    assert Ctheta.dtype == np.float64
    # assert each input has the correct shape
    assert L0.ndim == 1
    assert KxStar.ndim == 1
    assert Rtot.ndim == 2
    assert Ka.ndim == 3
    assert L0.shape[0] == KxStar.shape[0]
    assert L0.shape[0] == Rtot.shape[0]
    assert Ctheta.shape == (L0.shape[0], Cplx.shape[1])
    assert Cplx.shape == (L0.shape[0], Ctheta.shape[1], Ka.shape[1])
    assert L0.shape[0] == Ka.shape[0]
