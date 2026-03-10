import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh
import time


def estimate_spectrum_bounds(S, tol=1e-5):
    lam_max = eigsh(S, k=1, which='LA', tol=tol, return_eigenvectors=False)[0]
    lam_min = 1e-12  
    return float(lam_min), float(lam_max)

def cheb_coeff_sqrt_on_interval(lam_min, lam_max, deg, m_quad=2048):
    a = (lam_max - lam_min) / 2.0
    b = (lam_max + lam_min) / 2.0
    i = np.arange(m_quad)
    theta = (i + 0.5) * np.pi / m_quad
    fx = np.sqrt(a * np.cos(theta) + b)
    coskt = np.cos(np.outer(np.arange(deg + 1), theta))
    ck = (2.0 / np.pi) * (coskt @ fx) * (np.pi / m_quad)
    ck[0] *= 0.5
    return ck, a, b

def apply_sqrt_chebyshev(S, A, lam_min, lam_max, deg=80):
    if lam_min <= 0:
        raise ValueError("lam_min must be > 0 for sqrt; got %.3e" % lam_min)

    ck, a, b = cheb_coeff_sqrt_on_interval(lam_min, lam_max, deg)

    def S_tilde_mv(X):
        Y = (S @ X) if issparse(S) else S.dot(X)
        Y -= b * X
        Y *= (1.0 / a)
        return Y

    Bkp1 = np.zeros_like(A)
    Bk   = np.zeros_like(A)
    for k in range(deg, 0, -1):
        Bkm1 = 2.0 * S_tilde_mv(Bk) - Bkp1 + ck[k] * A
        Bkp1, Bk = Bk, Bkm1
    Y = S_tilde_mv(Bk) - Bkp1 + ck[0] * A
    return Y

def apply_S_half_adaptive(S, A, lam_min=None, lam_max=None,
                          deg0=40, step=10, tol=5e-4, maxdeg=200,
                          bound_tol=1e-3, verbose=True):

    if lam_min is None or lam_max is None:
        lam_min, lam_max = estimate_spectrum_bounds(S, tol=bound_tol)
        if verbose:
            print(f"[bounds] lam_min={lam_min:.6e}, lam_max={lam_max:.6e}")

    deg = deg0
    Y_prev = None
    while True:
        Y = apply_sqrt_chebyshev(S, A, lam_min, lam_max, deg=deg)
        if Y_prev is not None:
            num = np.linalg.norm(Y - Y_prev, ord='fro')
            den = np.linalg.norm(Y, ord='fro') + 1e-10
            rel = num / den
            if verbose:
                print(f"[deg {deg:3d}] rel_change = {rel:.3e}")
            if rel < tol:
                return Y, (lam_min, lam_max), deg
        if deg >= maxdeg:
            if verbose:
                print(f"[warn] reached maxdeg={maxdeg}, rel_change={rel:.3e}")
            return Y, (lam_min, lam_max), deg
        Y_prev = Y
        deg += step

if __name__ == "__main__":
    from scipy.sparse import random as sprand, csr_matrix

    n, m = 2000, 64
    R = sprand(n, n, density=0.01, format='csr', dtype=np.float64)
    S = (R + R.T) * 0.5
    d = np.abs(S).sum(axis=1).A.ravel() + 1e-3
    S = S + csr_matrix((d, (np.arange(n), np.arange(n))), shape=(n, n))

    A = np.random.randn(n, m)

    Y, (lam_min, lam_max), used_deg = apply_S_half_adaptive(
        S, A, deg0=40, step=10, tol=1e-8, maxdeg=120,
        bound_tol=1e-3, use_shift_invert=False, verbose=True
    )
    print(f"done. used_deg={used_deg}, bounds=({lam_min:.3e}, {lam_max:.3e}), ||Y||_F={np.linalg.norm(Y):.3e}")

