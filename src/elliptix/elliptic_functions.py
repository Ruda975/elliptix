#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Elliptix Source

File Name: elliptic_functions.py
Description: This module handles the numerical computation of all elliptic functions included in elliptix.
Author: Rudolf Rosendorf
Email: gdruda975@gmail.com
Date: 31/05/2026 (DD/MM/YYYY)
Version: 1.0.0

"""


from cmath import (exp, log as zlog, sqrt, sin, cos, tan, sinh, cosh, tanh, asin as arcsin, acos as arccos, 
                   atan as arctan, asinh as arcsinh, acosh as arccosh, atanh as arctanh, inf, nan, nanj)
from math import floor, ceil, remainder, factorial
from typing import TypeAlias, Literal, Sequence

from numpy import asarray, exp as np_exp, arange, roots, empty, complex128, int32, abs as np_abs
from scipy.special import elliprf, elliprg, comb, zeta


JacobiThetaVariants: TypeAlias = Literal[1, 2, 3, 4]
WeierstrassEIndex: TypeAlias = Literal[1, 2, 3]
WeierstrassGIndex: TypeAlias = Literal[2, 3]
WeierstrassWIndex: TypeAlias = Literal[1, 2, 3]
WeierstrassEtaIndex: TypeAlias = Literal[1, 2, 3]
JacobiEllipfunVariants: TypeAlias = Literal["am", 
                                            "cc", "cd", "cn", "cs", 
                                            "dc", "dd", "dn", "ds", 
                                            "nc", "nd", "nn", "ns", 
                                            "sc", "sd", "sn", "ss"]

PI = 3.141592653589793
""" π """
PI_I = 3.141592653589793j
""" πi """
PI_2 = 1.5707963267948966
""" π/2 """
TWO_PI = 6.283185307179586
""" 2π """
TWO_PI_I = 6.283185307179586j
""" 2πi """
THREE_PI_I = 9.42477796076938j
""" 3πi """
PI_I_2 = 1.5707963267948966j
""" πi/2 """
PI_I_4 = 0.7853981633974483j
""" πi/4 """
PI_I_12 = 0.26179938779914946j
""" πi/12 """
PI_SQ_12 = 0.8224670334241132
""" π²/12 """
PI_INV = 0.3183098861837907
""" 1/π """
I_PI_INV = 0.3183098861837907j
""" i/π """
SIX_OVER_PI_I = 1.909859317102744j
""" 6i/π """
TWO_ZETA_2 = 3.289868133696453
""" 2ζ(2) = π²/3 """
TWO_ZETA_4 = 2.1646464674222763
""" 2ζ(4) = π⁴/45 """
TWO_ZETA_6 = 2.0346861239688985
""" 2ζ(6) = 2π⁶/945 """
SQRT_PI_2 = 1.2533141373155003
""" √(π/2) """
ONE_54 = 0.018518518518518517
""" 1/54 """
I_SQRT_3_2 = 0.8660254037844386j
""" i √3/2 """

LEM_W = 1.1981402347355923
""" π / (√2 K(1/2)) """
LEM_Q = 0.04321391826377225
""" e^-π """
LEMH_W = 0.847213084793979
""" π / (2 K(1/2)) """
HALF_LEM = 1.3110287771460598
""" ϖ/2 ; ϖ — Lemniscate Constant. """

TAU_MIN = 0.4
""" Minimum value of Im(τ), for which we evaluate the infinite series. """
MAXITER = 100
""" Maximum number of allowed iterations. """
EPS = 2.220446049250313e-16
""" ε = 2^-52 """
LOG_EPS = -36.04365338911715
""" log(ε) = -52log(2) """
TAUEPS = 1.0e-8
""" Tolerance for τ in the inverse J invariant computation. """
TINYJ = 5.0e-324j
""" The smallest possible imaginary quantity. """
NANJ = nan + nanj
""" Complex NaN """

def wrong_ellipfun_variant(pq: str) -> str:
    return f"""Invalid variant ({pq}). The variant must be: 
1) A two character string containing "c", "d", "n", or "s". For example: "sn". 
2) "am" for the Jacobi amplitude."""


def log(z: complex) -> complex:
    if z == 0.0:
        return -inf + 0.0j
    return zlog(z + 0.0j)


def csc(z: complex) -> complex:
    s = sin(z)
    return NANJ if s == 0.0 else 1.0 / s


def sec(z: complex) -> complex:
    c = cos(z)
    return NANJ if c == 0.0 else 1.0 / c


def cot(z: complex) -> complex:
    t = tan(z)
    return NANJ if t == 0.0 else 1.0 / t


def csch(z: complex) -> complex:
    s = sinh(z)
    return NANJ if s == 0.0 else 1.0 / s


def sech(z: complex) -> complex:
    c = cosh(z)
    return NANJ if c == 0.0 else 1.0 / c


def coth(z: complex) -> complex:
    t = tanh(z)
    return NANJ if t == 0.0 else 1.0 / t


def arccsc(z: complex) -> complex:
    return NANJ if z == 0.0 else arcsin(1.0 / z)


def arcsec(z: complex) -> complex:
    return NANJ if z == 0.0 else arccos(1.0 / z)


def arccot(z: complex) -> complex:
    return PI_2 if z == 0.0 else arctan(1.0 / z)


def arccsch(z: complex) -> complex:
    return NANJ if z == 0.0 else arcsinh(1.0 / z)


def arcsech(z: complex) -> complex:
    return +inf if z == 0.0 else arccosh(1.0 / z)


def arccoth(z: complex) -> complex:
    return PI_I_2 if z == 0.0 else arctanh(1.0 / z)


def zatan2(y: complex, x: complex) -> complex:

    s = sqrt(x * x + y * y)
    if s == 0.0:
        return inf

    return -1.0j * log((x + 1.0j * y) / s)


def carlson_rf(x: complex, y: complex, z: complex) -> complex:
    """
    We add the smallest possible imaginary quantity (5.0e-324j), because scipy.special.elliprf doesn't 
    allow a 0 and a negative number in it's arguments.
    """
    if x.imag == 0.0 and x.real < 0.0:
        x += TINYJ
    if y.imag == 0.0 and y.real < 0.0:
        y += TINYJ
    if z.imag == 0.0 and z.real < 0.0:
        z += TINYJ

    return elliprf(x, y, z).item()


def carlson_rg(x: complex, y: complex, z: complex) -> complex:
    """
    We add the smallest possible imaginary quantity (5.0e-324j), because scipy.special.elliprg doesn't 
    allow a 0 and a negative number in it's arguments.
    """
    if x.imag == 0.0 and x.real < 0.0:
        x += TINYJ
    if y.imag == 0.0 and y.real < 0.0:
        y += TINYJ
    if z.imag == 0.0 and z.real < 0.0:
        z += TINYJ

    return elliprg(x, y, z).item()


def elliptick(m: complex) -> complex:
    """
    The Complete Elliptic Integral of the First Kind for complex arguments.

    K(m) = RF(0, 1 - m, 1) 

    https://dlmf.nist.gov/19.25#i
    """
    return carlson_rf(0.0, 1.0 - m, 1.0)


def elliptice(m: complex) -> complex:
    """
    The Complete Elliptic Integral of the Second Kind for complex arguments.

    E(m) = 2 RG(0, 1 - m, 1) 
    
    https://dlmf.nist.gov/19.25#i
    """
    return 2.0 * carlson_rg(0.0, 1.0 - m, 1.0)


def ellipticf(phi: complex, m: complex) -> complex:
    """
    https://mpmath.org/doc/0.19/functions/elliptic.html#ellipf
    https://en.wikipedia.org/wiki/Carlson_symmetric_form
    """
    n = round(PI_INV * phi.real)

    if n != 0:
        phi -= n * PI
        nkm = 2 * n * elliptick(m)
    else:
        nkm = 0.0
    
    s, c = sin(phi), cos(phi)
    return nkm + s * carlson_rf(1.0, c * c, 1.0 - m * s * s)


def normalize_imag(z: complex) -> complex:
    """ Normalizes the Im(z) to (-π, π), to correspond to the principal branch of log(z). """
    return z.real + 1.0j * remainder(z.imag, TWO_PI)


def periodicity(z: complex, tau: complex, u: complex) -> tuple[complex, complex]:
    """ 
    Uses the identity:

    θ₃(z, τ) = exp(iN(πNτ - 2z)) θ₃(z - (M + Nτ)π, τ)

    log(θ₃(z, τ)) = iN(πNτ - 2z) + log(θ₃(z - (M + Nτ)π, τ))

    to minimize both |Re(z)| and |Im(z)|.

    https://dlmf.nist.gov/20.2#ii
    """
    # Goal is to minimize Im(z) here.
    # Im(z) - π N Im(τ) = 0
    # N = floor(Im(z) / (π Im(τ)))
    n = z.imag // (PI * tau.imag)

    if n != 0:
        v = PI * n * tau
        u += 1.0j * n * (v - 2.0 * z)
        z -= v

    # Normalize Re(z) to (-π/2, π/2).
    return remainder(z.real, PI) + 1.0j * z.imag, u


def log_jacobi_theta_3_tau_series(z: complex, tau: complex, mu: int, nu: int) -> complex:
    """
    Compute the truncated infinite series for log(θ₃(z, τ)):

    log(θ₃(z, τ)) = log(Σ exp(2in(π/2 τ n + z)) (n = μ, ν))

    The sum is computed using the LogSumExp (LSE) trick:

    LSE(x1, ..., xn) = log(exp(x1) + ... + exp(xn))
    
    r = max(x1, ..., xn)

    LSE(x1, ..., xn) = r + log(exp(x1 - r) + ... + exp(xn - r))

    This is necessary for critical parameters, for example: (z = 50.05 + 2.0j, q = 0.99)

    https://mathworld.wolfram.com/JacobiThetaFunctions.html
    """
    x = 2.0j * (n := arange(mu, nu + 1)) * (PI_2 * tau * n + z)
    r = x.real.max()
    return r.item() + log(np_exp(x - r).sum().item())


def jacobi_theta_1_prime_fourier(z: complex, q: complex) -> complex: 
    """
    Computes derivative of θ₁(z, q) with respect to z: θ₁'(z, q) using the Fourier series:

    θ₁'(z, q) = 2q^(1/4) Σ (-1)^n q^(n (n + 1)) (2n + 1) cos((2n + 1) z) (n = 0, ∞)

    For the computation of Q(n) = (-1)^n q^(n (n + 1)) for successive n, the following recurrence is used:

    Q(n + 1) = -Q(n) q^(2n) q^2

    https://mathworld.wolfram.com/JacobiThetaFunctions.html
    """
    s = 0.0
    q2 = q * q
    q2n = qnn = 1.0
    two_z = 2.0 * z
    zn = z

    for k in range(1, 2 * MAXITER + 1, 2):

        t = k * qnn * cos(zn)
        s += t

        if abs(t) < max(EPS, EPS * abs(s)):
            break

        zn += two_z
        q2n *= q2
        qnn *= -q2n

    return 2.0 * q ** 0.25 * s
      

def log_jacobi_theta(n: JacobiThetaVariants, z: complex, q: complex) -> complex:

    """
    The Logarithm of the Jacobi Theta function, log(θ(n, z, q)).

    Parameters
    ----------
    n : integer, 1, 2, 3, or 4.
        The variant of the Jacobi Theta function.
    z : float or complex.
        The argument of the Theta function.
    q : float or complex.
        The elliptic nome, |q| < 1.

    Returns
    -------
    out : complex.
        The value of log(θ(n, z, q)).

    See Also
    --------
    jacobi_theta
            
    Implementation
    --------------
    First, the half-period ratio tau is computed from q. 

    The actual implementation computes log(θ₃(z, tau)), for n = 1, 2, 4, the logarithmic variants of the connection 
    formulas are used [1], allowing the computation of θ₁, θ₂ and θ₄ in terms of θ₃.

    Then, z is transformed by exploiting the Theta function's periodicity and quasi-periodicity [2], with the goal of 
    minimizing Re(z) and Im(z) for numerical stability.
    
    After that, the objective is to maximize Im(tau), for faster series convergence. We use the transformation of the
    lattice parameter identities [3] and [4]. 
    Then, we again perform the periodicity transformation of z, for faster convergence and numerical stability.

    Finally, we evaluate the truncated infinite series [5], using the LogSumExp trick, which improves
    numerical stability for critical parameters.

    References
    ----------
    [1]  https://dlmf.nist.gov/20.2.iii

    [2]  https://dlmf.nist.gov/20.2.E8

    [3]  https://dlmf.nist.gov/20.7.E28

    [4]  https://dlmf.nist.gov/20.7.E32

    [5]  https://mathworld.wolfram.com/JacobiThetaFunctions.html, formula (4)

    Examples
    --------

    Special Values:

    >>> log_jacobi_theta(3, 0.0, exp(-pi))
    (0.08290152003105485+0j)
    >>> 0.25 * log(pi) - loggamma(3/4)
    (0.08290152003105467+0j)

    >>> log_jacobi_theta(3, 0.0, exp(-sqrt(3) * pi))
    (0.008629500252467294+0j)
    >>> 3/2 * loggamma(4/3) - 2/3 * log(2) + 13/8 * log(3) - log(pi)
    (0.008629500252467448+0j)

    Landen Transformations:

    >>> z0, q0 = 1.0 - 0.5j, 0.1 - 0.2j

    >>> log_jacobi_theta(1, 2 * z0, q0 ** 2)
    (0.32420985740674046-0.21254079042010665j)
    >>> log_jacobi_theta(1, z0, q0) + log_jacobi_theta(2, z0, q0) - log_jacobi_theta(4, 0.0, q0 ** 2)
    (0.32420985740674046-0.21254079042010673j)

    >>> log_jacobi_theta(4, 2 * z0, q0 ** 2)
    (0.12364619632237805-0.3253198317374242j)
    >>> log_jacobi_theta(3, z0, q0) + log_jacobi_theta(4, z0, q0) - log_jacobi_theta(4, 0.0, q0 ** 2)
    (0.12364619632237796-0.3253198317374242j)

    """

    if q == 0.0:
        if n == 1 or n == 2:
            # θ₁(z, 0) = θ₂(z, 0) = 0 -> log(θ₁(z, 0)) = log(θ₂(z, 0)) = -∞
            return -inf
        elif n == 3 or n == 4:
            # θ₃(z, 0) = θ₄(z, 0) = 1 -> log(θ₃(z, 0)) = log(θ₄(z, 0)) = 0
            return 0.0
        else:
            error_message: str = f"Invalid variant ({n}). Jacobi Theta functions are only defined for n = 1, 2, 3, 4. "
            raise ValueError(error_message)
        
    if abs(q) >= 1.0:
        # θ(n, z, q) is only defined for |q| < 1
        return NANJ

    tau = -I_PI_INV * log(q)  # τ = log(q) / (iπ)

    """
    Transform θ₁, θ₂, θ₄ to θ₃, which is the easiest to compute.
    https://dlmf.nist.gov/20.2#iii
    """
    if n == 1:
        """
        θ₁(z, τ) = -i exp(iz + iπτ/4) θ₃(z + π/2 (1 + τ), τ)
        log(θ₁(z, τ)) = -iπ/2 + iz + iπτ/4 + log(θ₃(z + π/2 (1 + τ), τ))
        """
        u = -PI_I_2 + 1.0j * z + PI_I_4 * tau
        z += PI_2 * (1.0 + tau)

    elif n == 2:
        """
        θ₂(z, τ) = exp(iz + iπτ/4) θ₃(z + πτ/2, τ)
        log(θ₂(z, τ)) = iz + iπτ/4 + log(θ₃(z + πτ/2, τ))
        """
        u = 1.0j * z + PI_I_4 * tau
        z += PI_2 * tau

    elif n == 3:
        u = 0.0

    elif n == 4:
        """
        θ₄(z, τ) = θ₃(z + π/2, τ)
        log(θ₄(z, τ)) = log(θ₃(z + π/2, τ))
        """
        u = 0.0
        z += PI_2

    else:
        error_message: str = f"Invalid variant ({n}). Jacobi Theta functions are only defined for n = 1, 2, 3, 4. "
        raise ValueError(error_message)

    # Minimize |z|
    z, u = periodicity(z, tau, u)

    """
    Transform τ to a region, where the series converges in just a few terms.
    https://dlmf.nist.gov/20.7#viii
    """
    while tau.imag < TAU_MIN:
        """
        First, we want to transform Re(τ), to (-1/2, 1/2), because minimizing |Re(τ)| maximizes Im(τ).
        Apply the first transformation n times:
        θ₃(z, τ) = θ₄(z, τ - 1) = θ₃(z - π/2, τ - 1)
        """
        n = round(tau.real)

        if n & 1:
            """
            When it is applied an even number of times (n is even), we have:
            θ₃(z, τ) = θ₃(z - π, τ + 2) = θ₃(z, τ + 2)
            """
            z -= PI_2

        tau -= n

        """
        Then, we perform the τ' = -1/τ transform:
        θ₃(z, τ) = (-iτ)^(-1/2) exp(iτ'z²/π) θ₃(z τ', τ')
        log(θ₃(z, τ)) = iπ/4 - log(τ)/2 + iτ'z²/π + log(θ₃(z τ', τ'))
        """

        u += PI_I_4 - 0.5 * log(tau)

        tau = -1.0 / tau
        ztau = z * tau

        u += I_PI_INV * z * ztau
        z = ztau
    
    # Minimize |Im(z)| again, because the series converges faster, when |Im(z)| is small.
    z, u = periodicity(z, tau, u)
    
    """
    To find the series bounds, we want to find n, such that:

    |exp(2in(π/2 τ n + z))| = ε
    exp(-π Im(τ) n² - 2 Im(z) n) = ε
    -π Im(τ) n² - 2 Im(z) n - log(ε) = 0

    -π Im(τ) n² - 2 Im(z) n - log(ε) is quadratic in n.
    The vertex of the parabola: v = -Im(z) / (π Im(τ))
    Its roots are:
    d = √(v² - log(ε) / (π Im(τ)))
    n₁ = v - d ; n₂ = v + d
    
    floor(n₁) is the lower summation bound and ceil(n₂) is the upper bound.
    """
    t = -PI_INV / tau.imag
    v =  t * z.imag
    d = sqrt(v * v + LOG_EPS * t).real

    return normalize_imag(u + log_jacobi_theta_3_tau_series(z, tau, floor(v - d), ceil(v + d)))


def jacobi_theta(n: JacobiThetaVariants, z: complex, q: complex) -> complex:

    """
    The Jacobi Theta function, θ(n, z, q).

    Parameters
    ----------
    n : integer, 1, 2, 3, or 4.
        The variant of the Jacobi Theta function.
    z : float or complex.
        The argument of the Theta function.
    q : float or complex.
        The elliptic nome, |q| < 1.

    Returns
    -------
    out : complex.
        The value of θ(n, z, q).

    See Also
    --------
    log_jacobi_theta

    Implementation
    --------------
    The Jacobi Theta function is computed directly as:

    jacobi_theta(n, z, q) = exp(log_jacobi_theta(n, z, q))

    See the documentation of log_jacobi_theta for the implementation details.

    References
    ----------
    See: log_jacobi_theta

    Examples
    --------

    Special Values:

    >>> jacobi_theta(3, 0.0, exp(-2 * pi))
    (1.0037348854877393+0j)
    >>> pi ** (1/4) / gamma(3/4) * sqrt(2 + sqrt(2))/2
    (1.0037348854877393+0j)

    >>> jacobi_theta(3, 0.0, exp(-sqrt(2) * pi))
    (1.023523999341006+0j)
    >>> gamma(9/8) * 2 ** (7/8) / sqrt(pi * gamma(5/4))
    (1.0235239993410057+0j)

    Duplication Formula:

    >>> z0, q0 = -3.1 + 0.2j, 0.5 - 0.1j

    >>> jacobi_theta(1, 2 * z0, q0)
    (-0.16055069582413764+0.18174882521957836j)
    >>> 2 * (jacobi_theta(1, z0, q0) * jacobi_theta(2, z0, q0) * jacobi_theta(3, z0, q0) * jacobi_theta(4, z0, q0) / 
    ... (jacobi_theta(2, 0.0, q0) * jacobi_theta(3, 0.0, q0) * jacobi_theta(4, 0.0, q0)))
    (-0.16055069582413783+0.18174882521957852j)

    """

    return exp(log_jacobi_theta(n, z, q))


def weierstrass_e_from_g23(g2: complex, g3: complex) -> tuple[complex, complex, complex]:

    """
    Ordering is based on:
    https://dlmf.nist.gov/23.22
    """

    if g2 == 0.0 and g3 == 0.0:
        return 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j

    delta = g2 * g2 * g2 - 27.0 * g3 * g3

    if delta == 0.0:
        """
        Δ = 0 means double roots, which are imprecise when computed using numpy.roots.
        Example where this happens would be g₂ = 3 and g₃ = 1.

        Using Vieta's, D is the double root, S is the single root:
        2D + S = 0
        D² + 2DS = -g₂/4
        D²S = g₃/4

        Dividing D²S = g₃/4 and D² + 2DS = -g₂/4, and substituting S = -2D we get:
        2/3 D = -g₃/g₂  =>  D = -3/2 g₃/g₂
        S = -2D = 3 g₃/g₂
        """
        r = g3 / g2
        e123 = asarray((d := -1.5 * r, d, 3.0 * r), dtype=complex128)

    else:
        e123 = roots((4.0, 0.0, -g2, -g3)).astype(complex128)

    if g2.imag == 0.0 and g3.imag == 0.0 and delta.real >= 0.0:
        # If the Δ > 0, the order is e1 > e2 > e3
        return tuple(e123[(-e123.real).argsort()].tolist())

    a, b, c = e123.tolist()

    # s > 0: positively oriented (ccw), s == 0: collinear, s < 0: negatively oriented (cw)
    s = (b.real - a.real) * (c.imag - b.imag) - (b.imag - a.imag) * (c.real - b.real)
    if s < 0.0:
        # This swap should guarantee positive orientation (or collinearity, but that doesn't matter, right?).
        b, c = c, b

    # Sort the points (a, b, c) -> (alpha, beta, gamma) such that the side [alpha, gamma] is the longest side.
    ab, ac, bc = abs(b - a), abs(c - a), abs(c - b)

    if (abac := ab > ac) and ab > bc:  # ab > ac and ab > bc: ab max
        return b, c, a  # +(a, b, c) -> -(a, c, b) -> +(b, c, a)
    
    elif not abac and ac > bc:  # ac > ab and ac > bc: ac max
        return a, b, c  # +(a, b, c)

    # bc > ab and bc > ac: bc max
    return c, a, b  # +(a, b, c) -> -(b, a, c) -> +(c, a, b)


def weierstrass_e_from_w13(w1: complex, w3: complex) -> tuple[complex, complex, complex]:
    
    # Based on DLMF 23.6.1 — 23.6.4

    w1i = 1.0 / w1
    c = PI_SQ_12 * w1i * w1i
    q = exp(PI_I * w3 * w1i)

    t24, t44 = exp(4.0 * log_jacobi_theta(2, 0.0, q)), exp(4.0 * log_jacobi_theta(4, 0.0, q))

    e1, e2 = c * (t24 + 2.0 * t44), c * (t24 - t44)

    return e1, e2, -e1 - e2  # e1 + e2 + e3 = 0


def weierstrass_p(z: complex, g2: complex, g3: complex) -> complex:

    """
    The Weierstrass Elliptic function for the invariants g₂ and g₃, ℘(z; g₂, g₃).

    Parameters
    ----------
    z  : float or complex.
         The argument of the Weierstrass ℘ function.
    g2 : float or complex.
         The Weierstrass invariant g₂.
    g3 : float or complex.
         The Weierstrass invariant g₃.

    Returns
    -------
    out : complex.
        The value of ℘(z; g₂, g₃).

    See Also
    --------
    weierstrass_p_prime, inverse_weierstrass_p, weierstrass_sigma, weierstrass_zeta, weierstrass_e, weierstrass_g,
    weierstrass_w, weierstrass_eta
            
    Implementation
    --------------
    The Weierstrass ℘ function can be expressed in terms of the Jacobi elliptic function ns [1].

    The ns function is computed directly from its relation to the Jacobi Theta function [2].

    The lattice roots e₁, e₂, e₃ are the roots of the cubic polynomial 4x³ - g₂ x - g₃. 
    For their computation, see weierstrass_e.

    References
    ----------
    [1] https://functions.wolfram.com/EllipticFunctions/WeierstrassP/27/02/02/0001/

    [2] https://functions.wolfram.com/EllipticFunctions/JacobiNS/27/02/08/0001/

    Examples
    --------

    Special Values:

    >>> z0 = 2.1 - 0.4j

    >>> weierstrass_p(z0, 0, 0)
    (0.20349630594352858+0.08044089270238307j)
    >>> 1 / z0 ** 2
    (0.20349630594352858+0.08044089270238307j)

    >>> weierstrass_p(z0, 3, 1) 
    (0.4011495886948573-2.570469274892414j)
    >>> 3 / (2 * tan(sqrt(3/2) * z0) ** 2) + 1
    (0.40114958869485684-2.5704692748924134j)

    Lemniscatic Case:

    >>> weierstrass_p(gamma(1/4) ** 2 / (4 * sqrt(2 * pi)), 4, 0)
    (1.0000000000000004-8.881784197001254e-16j)
    >>> 1 + 0j
    (1+0j)
    
    Equianharmonic Case:

    >>> weierstrass_p(gamma(1/3) ** 3 / (2 ** (7/3) * pi), 0, 4)
    (1.0000000000000013+1.887379141862766e-15j)
    >>> 1 + 0j
    (1+0j)

    """

    e1, e2, e3 = weierstrass_e_from_g23(g2, g3)
    e13 = e1 - e3

    if e13 == 0.0:
        return 1.0 / (z * z)
    
    m = (e2 - e3) / e13
    
    if m == 0.0:
        # e₃ + (e₁ - e₃) ns²(z √(e₁ - e₃), 0) = e₃ + (e₁ - e₃) csc²(z √(e₁ - e₃))
        return e3 + e13 * csc(z * sqrt(e13)) ** 2
    
    elif m == 1.0:
        # e₃ + (e₁ - e₃) ns²(z √(e₁ - e₃), 1) = e₃ + (e₁ - e₃) coth²(z √(e₁ - e₃))
        return e3 + e13 * coth(z * sqrt(e13)) ** 2

    kmi = PI / elliptick(m)
    q = exp(-kmi * elliptick(1.0 - m))
 
    w = 0.5 * z * kmi * sqrt(e13)
    return e3 + e13 * sqrt(m) * exp(2.0 * (log_jacobi_theta(4, w, q) - log_jacobi_theta(1, w, q)))


def weierstrass_p_prime(z: complex, g2: complex, g3: complex) -> complex:
    
    """
    The derivative of the Weierstrass Elliptic function with respect to z for the invariants g₂ and g₃, ℘'(z; g₂, g₃).

    Parameters
    ----------
    z  : float or complex.
         The argument of the Weierstrass ℘' function.
    g2 : float or complex.
         The Weierstrass invariant g₂.
    g3 : float or complex.
         The Weierstrass invariant g₃.

    Returns
    -------
    out : complex.
        The value of the derivative of ℘'(z; g₂, g₃).

    See Also
    --------
    weierstrass_p, inverse_weierstrass_p, weierstrass_sigma, weierstrass_zeta, weierstrass_e, weierstrass_g,
    weierstrass_w, weierstrass_eta
            
    Implementation
    --------------
    The derivative of the Weierstrass ℘ function can be expressed in terms of the Jacobi elliptic functions [1].

    The Jacobi elliptic functions sn, cn and dn can be computed directly from its relation to the Jacobi 
    Theta function [2], [3], [4].

    The lattice roots e₁, e₂, e₃ are the roots of the cubic polynomial 4x³ - g₂ x - g₃. 
    For their computation, see weierstrass_e.

    References
    ----------
    [1] https://functions.wolfram.com/EllipticFunctions/WeierstrassPPrime/27/02/02/0001/

    [2] https://functions.wolfram.com/EllipticFunctions/JacobiSN/27/02/08/0001/

    [3] https://functions.wolfram.com/EllipticFunctions/JacobiCN/27/02/08/0001/

    [4] https://functions.wolfram.com/EllipticFunctions/JacobiDN/27/02/08/0001/

    Examples
    --------

    Special Values:

    >>> z0 = 2.1 - 0.4j

    >>> weierstrass_p_prime(z0, 0, 0)
    (-0.17293911833718018-0.10955115844744677j)
    >>> -2 / z0 ** 3
    (-0.17293911833718018-0.10955115844744677j)

    >>> weierstrass_p_prime(z0, 3, 1) 
    (-4.721048235851306-7.48597214507722j)
    >>> -3 * sqrt(3/2) / (tan(sqrt(3/2) * z0) * sin(sqrt(3/2) * z0) ** 2)
    (-4.721048235851307-7.48597214507722j)

    Lemniscatic Case:

    >>> weierstrass_p_prime(gamma(1/4) ** 2 / (4 * sqrt(2 * pi)), 4, 0)
    (9.45722172159951e-16-9.557875874462315e-31j)
    >>> 0j
    0j
    
    Equianharmonic Case:

    >>> weierstrass_p_prime(gamma(1/3) ** 3 / (2 ** (7/3) * pi), 0, 4)
    (-7.048943009773035e-15+2.0191504841620116e-15j)
    >>> 0j
    0j

    """

    e1, e2, e3 = weierstrass_e_from_g23(g2, g3)
    e13 = e1 - e3

    if e13 == 0.0:
        return -2.0 / (z * z * z)

    s = sqrt(e13)
    u = z * s

    m = (e2 - e3) / e13
    
    if m == 0.0:
        """
        -2 (e₁ - e₃)^(3/2) cn(z √(e₁ - e₃), 0) dn(z √(e₁ - e₃), 0) / sn³(z √(e₁ - e₃), 0) = 
        -2 (e₁ - e₃)^(3/2) cot(z √(e₁ - e₃)) csc²(z √(e₁ - e₃))
        """
        return -2.0 * e13 * s * cos(u) / sin(u) ** 3
    
    elif m == 1.0:
        """
        -2 (e₁ - e₃)^(3/2) cn(z √(e₁ - e₃), 1) dn(z √(e₁ - e₃), 1) / sn³(z √(e₁ - e₃), 1) = 
        -2 (e₁ - e₃)^(3/2) coth(z √(e₁ - e₃)) csch²(z √(e₁ - e₃))
        """
        return -2.0 * e13 * s * cosh(u) / sinh(u) ** 3

    kmi = PI / elliptick(m)
    q = exp(-kmi * elliptick(m1 := 1.0 - m))
    w = 0.5 * kmi * u

    return -2.0 * e13 * s * sqrt(m * m1) * exp(log_jacobi_theta(2, w, q) + log_jacobi_theta(3, w, q) + 
                                               log_jacobi_theta(4, w, q) - 3.0 * log_jacobi_theta(1, w, q))


def inverse_weierstrass_p(w: complex, g2: complex, g3: complex) -> complex:

    """
    The Inverse of the Weierstrass Elliptic function for the invariants g₂ and g₃, ℘⁻¹(w; g₂, g₃).

    Parameters
    ----------
    w  : float or complex.
         The argument of the Inverse Weierstrass ℘⁻¹ function.
    g2 : float or complex.
         The Weierstrass invariant g₂.
    g3 : float or complex.
         The Weierstrass invariant g₃.

    Returns
    -------
    out : complex.
        The value z, such that w = ℘(z; g₂, g₃).

    See Also
    --------
    weierstrass_p, weierstrass_p_prime, weierstrass_sigma, weierstrass_zeta, weierstrass_e, weierstrass_g,
    weierstrass_w, weierstrass_eta
            
    Implementation
    --------------
    The inverse Weierstrass-℘ function can be represented as the Carlson elliptic integral RF(x, y, z):

    From the differential equation for the Weierstrass ℘ function [1], we have the following integral representation:
    ℘⁻¹(w) = ∫(w, ∞) 1 / √(4s³ - g₂s - g₃) ds

    Using the lattice roots e₁, e₂, e₃, the polynomial can be written as:
    4s³ - g₂s - g₃ = 4(s - e₁)(s - e₂)(s - e₃)

    u = s - w ; du = ds ; s = u + w

    ℘⁻¹(w) = 1/2 ∫(0, ∞) 1 / √((u + w - e₁)(u + w - e₂)(u + w - e₃)) du

    Directly from the definition of Carlson RF [2]:
    ℘⁻¹(w) = RF(w - e₁, w - e₂, w - e₃)

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Weierstrass_elliptic_function#Differential_equation

    [2] https://en.wikipedia.org/wiki/Carlson_symmetric_form

    https://math.stackexchange.com/questions/4381226/elliptic-integral-as-inverse-of-weierstrass-elliptic-function

    Examples
    --------

    Special Values:

    >>> z0 = -3 - 0.8j

    >>> inverse_weierstrass_p(z0, 0, 0)
    (0.07373938465413428+0.5627084647943774j)
    >>> 1 / sqrt(z0)
    (0.07373938465413427+0.5627084647943774j)

    >>> inverse_weierstrass_p(z0, 3, 1) 
    (0.07626223860994176+0.5660637360487425j)
    >>> sqrt(2/3) * arctan(1 / sqrt(2/3 * (z0 - 1)))
    (0.07626223860994179+0.5660637360487426j)

    Differentiation:

    >>> g20, g30 = 1 - 2j, -2.1 + 1.1j

    >>> 1 / weierstrass_p_prime(inverse_weierstrass_p(z0, g20, g30), g20, g30)
    (0.03803618597134445+0.08308606998718632j)
    >>> 1 / sqrt(4 * z0 ** 3 - g20 * z0 - g30)
    (0.03803618597134442+0.08308606998718614j)

    Inverse Property:

    >>> weierstrass_p(inverse_weierstrass_p(z0, g20, g30), g20, g30)
    (-2.9999999999999982-0.7999999999999989j)
    >>> z0
    (-3-0.8j)
    """

    e1, e2, e3 = weierstrass_e_from_g23(g2, g3)
    return carlson_rf(w - e1, w - e2, w - e3)


def weierstrass_w_from_e123(e1: complex, e2: complex, e3: complex) -> tuple[complex, complex]:

    e13i = 1.0 / (e1 - e3)
    s = sqrt(e13i)
    m = e13i * (e2 - e3)

    return s * elliptick(m), 1.0j * s * elliptick(1.0 - m)


def weierstrass_eta_from_e123(e1: complex, e2: complex, e3: complex) -> tuple[complex, complex, complex]:

    # http://functions.wolfram.com/09.21.27.0004.01

    e13 = e1 - e3
    e13i = 1.0 / e13
    s = sqrt(e13)
    
    m = e13i * (e2 - e3)

    eta1 = s * (elliptice(m) - e1 * e13i * elliptick(m))
    eta3 = -1.0j * s * (elliptice(m1 := 1.0 - m) + e3 * e13i * elliptick(m1))

    return eta1, -eta1 - eta3, eta3  # eta1 + eta2 + eta3 = 0


def weierstrass_sigma(z: complex, g2: complex, g3: complex) -> complex:

    """
    The Weierstrass Sigma function for the invariants g₂ and g₃, σ(z; g₂, g₃).

    Parameters
    ----------
    z  : float or complex.
         The argument of the Weierstrass σ function.
    g2 : float or complex.
         The Weierstrass invariant g₂.
    g3 : float or complex.
         The Weierstrass invariant g₃.

    Returns
    -------
    out : complex.
        The value of σ(z; g₂, g₃).

    See Also
    --------
    weierstrass_p, weierstrass_p_prime, inverse_weierstrass_p, weierstrass_zeta, weierstrass_e, weierstrass_g,
    weierstrass_w, weierstrass_eta
            
    Implementation
    --------------
    The Weierstrass Sigma function is computed using the Jacobi Theta function [1].

    θ₁'(0, q) is computed using the lattice roots and the half-period ω₁ [2], ω₁ and η₁ are computed 
    using K(m), E(m) and e₁, e₂, e₃ from [3] and [4].

    The lattice roots e₁, e₂, e₃ are the roots of the cubic polynomial 4x³ - g₂ x - g₃. 
    For their computation, see weierstrass_e.

    References
    ----------
    [1] https://functions.wolfram.com/EllipticFunctions/WeierstrassSigma/27/01/02/0002/

    [2] https://functions.wolfram.com/EllipticFunctions/EllipticThetaPrime1/03/01/02/0002/

    [3] http://functions.wolfram.com/09.18.27.0003.01

    [4] http://functions.wolfram.com/09.21.27.0004.01

    Examples
    --------

    Special Values:

    >>> z0 = 2.1 - 0.4j

    >>> weierstrass_sigma(z0, 0, 0)
    (2.1-0.4j)
    >>> z0
    (2.1-0.4j)

    >>> weierstrass_sigma(z0, 3, 1) 
    (1.7194146635160368+0.3428241744183744j)
    >>> sqrt(2/3) * exp(z0 ** 2 / 4) * sin(sqrt(3/2) * z0)
    (1.7194146635160368+0.3428241744183744j)

    Weierstrass Constant:

    >>> 1/2 * weierstrass_sigma(1, *weierstrass_g(w13=(1, 1j)))
    (0.47494937998792075-1.8071429272298379e-38j)
    >>> 2 ** (5/4) * sqrt(pi) * exp(pi/8) / gamma(1/4) ** 2
    (0.4749493799879207+0j)

    """

    e1, e2, e3 = weierstrass_e_from_g23(g2, g3)
    e13, e23 = e1 - e3, e2 - e3

    if e13 == 0.0:
        return z

    e13i = 1.0 / e13
    s = sqrt(e13)

    m = e23 * e13i
    
    if m == 1.0:
        """
        From the integral relation: 
        σ(z; g₂, g₃) = z exp(∫(0, z) ζ(t; g₂, g₃) - 1/t dt)

        ∫ ζ(t; g₂, g₃) - 1/t dt = 
        ∫ √(e₁ - e₃) coth(t √(e₁ - e₃)) - e₁t - 1/t dt = 
        log(sinh(t √(e₁ - e₃))) - e₁ t²/2 - log(t) + C

        ∫(0, z) ζ(t; g₂, g₃) - 1/t dt = 
        log(sinh(z √(e₁ - e₃))) - e₁ z²/2 - log(z) - log(s)

        σ(z; g₂, g₃) = z exp(log(sinh(z √(e₁ - e₃))) - e₁ z²/2 - log(z) - log(s))
        = exp(-e₁ z²/2) sinh(z √(e₁ - e₃)) / √(e₁ - e₃)
        Nice!
        """
        return exp(-0.5 * e1 * z * z) * sinh(z * s) / s

    km = elliptick(m)

    w1 = 2.0 * km / s  # :)
    w1p = PI_INV * w1
    eta1 = s * (elliptice(m) - e1 * e13i * km)
    u = z / w1

    if m == 0.0:
        """
        From the Fourier series expansions for θ₁:
        θ₁(z, q) ~ 2 q^(1/4) sin(z) ; q -> 0
        θ₁'(0, q) ~ 2 q^(1/4) ; q -> 0
        lim(q -> 0) θ₁(w, q)/θ₁'(0, q) = sin(w)
        lim(q -> 0) 2ω₁/π exp(η₁ z² / (2ω₁)) θ₁(π z / (2ω₁), q) / θ₁'(0, q) = 2ω₁/π exp(η₁ z² / (2ω₁)) sin(π z / (2ω₁))
        """
        return w1p * exp(eta1 * z * u) * sin(PI * u)

    # https://functions.wolfram.com/EllipticFunctions/EllipticThetaPrime1/03/01/02/0002/
    t1q = (w1p ** 6 * e23 * (e1 - e2) * e13) ** -0.25
    
    return w1p * t1q * exp(eta1 * z * u + log_jacobi_theta(1, PI * u, exp(-PI * elliptick(1.0 - m) / km)))


def weierstrass_zeta(z: complex, g2: complex, g3: complex) -> complex:

    """
    The Weierstrass Zeta function for the invariants g₂ and g₃, ζ(z; g₂, g₃).

    Parameters
    ----------
    z  : float or complex.
         The argument of the Weierstrass ζ function.
    g2 : float or complex.
         The Weierstrass invariant g₂.
    g3 : float or complex.
         The Weierstrass invariant g₃.

    Returns
    -------
    out : complex.
        The value of ζ(z; g₂, g₃).

    See Also
    --------
    weierstrass_p, weierstrass_p_prime, inverse_weierstrass_p, weierstrass_sigma, weierstrass_e, weierstrass_g,
    weierstrass_w, weierstrass_eta
            
    Implementation
    --------------
    The Weierstrass Zeta function is computed using the Jacobi Theta function [1].

    θ₁'(z, q) is computed by evaluating its Fourier series, which is obtained by differentiating the Fourier
    series of θ₁(z, q) [2] term by term. 
    
    ω₁ and η₁ are computed using K(m), E(m) and e₁, e₂, e₃ from [3] and [4].

    The lattice roots e₁, e₂, e₃ are the roots of the cubic polynomial 4x³ - g₂ x - g₃. 
    For their computation, see weierstrass_e.

    References
    ----------
    [1] http://functions.wolfram.com/09.17.27.0004.01

    [2] https://dlmf.nist.gov/20.2.E1

    [3] http://functions.wolfram.com/09.18.27.0003.01

    [4] http://functions.wolfram.com/09.21.27.0004.01

    Examples
    --------

    Special Values:

    >>> z0 = 1.3 - 0.5j

    >>> weierstrass_zeta(z0, 0, 0)
    (0.6701030927835051+0.25773195876288657j)
    >>> 1 / z0
    (0.6701030927835051+0.25773195876288657j)

    >>> weierstrass_zeta(z0, 3, 1) 
    (0.6316218506730146+0.41867398832776637j)
    >>> z0/2 + sqrt(3/2) / tan(sqrt(3/2) * z0)
    (0.6316218506730146+0.41867398832776626j)

    Transformations:

    >>> g20, g30 = 0.4j, 1.5

    >>> weierstrass_zeta(1j * z0, g20, g30)
    (0.19402683245604282-0.6395531671131188j)
    >>> -1j * weierstrass_zeta(z0, g20, -g30)
    (0.19402683245604266-0.6395531671131184j)

    >>> weierstrass_zeta(2 * z0, g20, g30)
    (1.0085592411687425+0.9830098859684033j)
    >>> (2 * weierstrass_zeta(z0, g20, g30) + (3 * weierstrass_p(z0, g20, g30) ** 2 - g20/4) / 
    ...  weierstrass_p_prime(z0, g20, g30))
    (1.0085592411687392+0.9830098859684011j)

    """

    e1, e2, e3 = weierstrass_e_from_g23(g2, g3)
    e13 = e1 - e3

    if e13 == 0.0:
        return 1.0 / z

    e13i = 1.0 / e13
    s = sqrt(e13)

    m = (e2 - e3) * e13i

    if m == 1.0:
        """
        From the integral relation: 
        ζ(z; g₂, g₃) = 1/z - ∫(0, z) ℘(t; g₂, g₃) - 1/t² dt
        When m = 1, ℘(t; g₂, g₃) = e₃ + (e₁ - e₃) coth²(t √(e₁ - e₃))
        ζ(z; g₂, g₃) = 1/z - ∫(0, z) e₃ + (e₁ - e₃) coth²(t √(e₁ - e₃)) - 1/t² dt

        Integrating coth²(t) dt:
        Substitute u = tanh(t) ; du = (1 - u²) dt and do partial fractions to obtain ∫ 1/u² + 1/(1 - u²) du
        = -1/u + arctanh(u) + C = x - coth(x) + C

        ∫(0, z) e₃ + (e₁ - e₃) coth²(t √(e₁ - e₃)) - 1/t² dt
        = [e₃t + (e₁ - e₃) (t - coth(t √(e₁ - e₃)) / √(e₁ - e₃)) + 1/t](0, z)
        = e₁z - √(e₁ - e₃) coth(z √(e₁ - e₃)) + 1/z

        ζ(z; g₂, g₃) = √(e₁ - e₃) coth(z √(e₁ - e₃)) - e₁z
        Awesome!
        """
        return s * coth(s * z) - e1 * z

    km = elliptick(m)
    kmi = 1.0 / km
    q = exp(-PI * kmi * elliptick(1.0 - m))
    eta1 = s * (elliptice(m) - e1 * e13i * km)

    w1i = s * kmi
    zw = z * w1i

    if m == 0.0:
        """
        From the Fourier series expansions for θ₁:
        θ₁(z, q) ~ 2 q^(1/4) sin(z) ; q -> 0
        θ₁'(z, q) ~ 2 q^(1/4) cos(z) ; q -> 0
        lim(q -> 0) θ₁'(w, q)/θ₁(w, q) = cot(w)
        lim(q -> 0) z η₁ / ω₁ + π θ₁'(π z / (2ω₁), q) / (2ω₁ θ₁(π z / (2ω₁), q)) = z η₁ / ω₁ + π/(2ω₁) cot(π z / (2ω₁))
        """
        return zw * eta1 + PI_2 * w1i * cot(PI_2 * zw)
    
    return zw * eta1 + PI_2 * w1i * jacobi_theta_1_prime_fourier(u := PI_2 * zw, q) * exp(-log_jacobi_theta(1, u, q))


def neville_theta_c(u: complex, m: complex) -> complex:

    """
    The Neville Theta function θc(u, m).

    Parameters
    ----------
    u : float or complex.
        The argument of the Theta function.
    m : float or complex.
        The elliptic parameter.

    Returns
    -------
    out : complex.
        The value of θc(u, m).

    See Also
    --------
    neville_theta_d, neville_theta_n, neville_theta_s, jacobi_ellipfun, inverse_jacobi_ellipfun

    Implementation
    --------------
    The Neville Theta functions are computed using the connection formulas to the Jacobi Theta function [1].

    References
    ----------
    [1] http://functions.wolfram.com/09.09.27.0008.01
    
    Examples
    --------

    Connection to Neville θs(u, m):

    >>> u0, m0 = -0.2 + 0.6j, 2.2 - 2.9j

    >>> neville_theta_c(u0 + elliptick(m0), m0)
    (0.6973651663708349-0.48787645046120476j)
    >>> -(1 - m0) ** (1/4) * neville_theta_s(u0, m0)
    (0.6973651663708351-0.48787645046120476j)

    """

    if m == 0.0:
        return cos(u)
    elif m == 1.0:
        return 1.0 + 0.0j

    kmi = 1.0 / elliptick(m)
    q = exp(-PI * kmi * elliptick(1.0 - m))

    return SQRT_PI_2 * sqrt(kmi / sqrt(m)) * jacobi_theta(2, PI_2 * u * kmi, q)
 

def neville_theta_d(u: complex, m: complex) -> complex:

    """
    The Neville Theta function θd(u, m).

    Parameters
    ----------
    u : float or complex.
        The argument of the Theta function.
    m : float or complex.
        The elliptic parameter.

    Returns
    -------
    out : complex.
        The value of θd(u, m).

    See Also
    --------
    neville_theta_c, neville_theta_n, neville_theta_s, jacobi_ellipfun, inverse_jacobi_ellipfun

    Implementation
    --------------
    The Neville Theta functions are computed using the connection formulas to the Jacobi Theta function [1].

    References
    ----------
    [1] http://functions.wolfram.com/09.10.27.0008.02
    
    Examples
    --------

    Connection to Neville θn(u, m):

    >>> u0, m0 = -2.6 - 3.2j, 0.2 - 1.7j

    >>> neville_theta_d(u0 + elliptick(m0), m0)
    (-2392.8231651113133-1605.9208977722667j)
    >>> (1 - m0) ** (1/4) * neville_theta_n(u0, m0)
    (-2392.8231651113083-1605.9208977722565j)

    """

    if m == 0.0:
        return 1.0 + 0.0j
    elif m == 1.0:
        return 1.0 + 0.0j

    kmi = 1.0 / elliptick(m)
    q = exp(-PI * kmi * elliptick(1.0 - m))

    return SQRT_PI_2 * sqrt(kmi) * jacobi_theta(3, PI_2 * u * kmi, q)


def neville_theta_n(u: complex, m: complex) -> complex:

    """
    The Neville Theta function θn(u, m).

    Parameters
    ----------
    u : float or complex.
        The argument of the Theta function.
    m : float or complex.
        The elliptic parameter.

    Returns
    -------
    out : complex.
        The value of θn(u, m).

    See Also
    --------
    neville_theta_c, neville_theta_d, neville_theta_s, jacobi_ellipfun, inverse_jacobi_ellipfun

    Implementation
    --------------
    The Neville Theta functions are computed using the connection formulas to the Jacobi Theta function [1].

    References
    ----------
    [1] http://functions.wolfram.com/09.11.27.0008.01
    
    Examples
    --------

    Connection to Neville θd(u, m):

    >>> u0, m0 = 0.9 - 1.2j, 0.4 + 0.4j

    >>> neville_theta_n(u0 + elliptick(m0), m0)
    (0.6596259505914532+0.38557067279438734j)
    >>> 1 / (1 - m0) ** (1/4) * neville_theta_d(u0, m0)
    (0.6596259505914535+0.38557067279438767j)

    """

    if m == 0.0:
        return 1.0 + 0.0j
    elif m == 1.0:
        return cosh(u)

    kmi = 1.0 / elliptick(m)
    q = exp(-PI * kmi * elliptick(1.0 - m))

    return SQRT_PI_2 * sqrt(kmi / sqrt(1.0 - m)) * jacobi_theta(4, PI_2 * u * kmi, q)


def neville_theta_s(u: complex, m: complex) -> complex:

    """
    The Neville Theta function θs(u, m).

    Parameters
    ----------
    u : float or complex.
        The argument of the Theta function.
    m : float or complex.
        The elliptic parameter.

    Returns
    -------
    out : complex.
        The value of θs(u, m).

    See Also
    --------
    neville_theta_c, neville_theta_d, neville_theta_n, jacobi_ellipfun, inverse_jacobi_ellipfun

    Implementation
    --------------
    The Neville Theta functions are computed using the connection formulas to the Jacobi Theta function [1].

    References
    ----------
    [1] http://functions.wolfram.com/09.12.27.0008.01
    
    Examples
    --------

    Connection to Neville θc(u, m):

    >>> u0, m0 = 1.7 - 0.1j, 0.3 - 2.7j

    >>> neville_theta_s(elliptick(m0) - u0, m0)
    (-0.5961935591031594-0.2771970799563592j)
    >>> 1 / (1 - m0) ** (1/4) * neville_theta_c(u0, m0)
    (-0.596193559103159-0.2771970799563599j)

    """

    if m == 0.0:
        return sin(u)
    elif m == 1.0:
        return sinh(u)

    kmi = 1.0 / elliptick(m)
    q = exp(-PI * kmi * elliptick(1.0 - m))

    return SQRT_PI_2 * sqrt(kmi / sqrt(m * (1.0 - m))) * jacobi_theta(1, PI_2 * u * kmi, q)


def dedekind_eta(tau: complex) -> complex:

    """
    The Dedekind Eta function, η(τ).

    Parameters
    ----------
    tau : float or complex.
          The modular parameter, Im(τ) > 0.

    Returns
    -------
    out : complex.
        The value of η(τ).

    See Also
    --------
    modular_lambda, klein_j, inverse_klein_j, euler_phi, eisenstein_e, eisenstein_g

    Implementation
    --------------
    The Dedekind Eta function is computed using the connection formula to the Jacobi Theta function [1].

    References
    ----------
    [1] https://dlmf.nist.gov/23.15.E9

    Examples
    --------

    Special Values:

    >>> dedekind_eta(1j)
    (0.7682254223260566-1.7569052324234992e-19j)
    >>> gamma(1/4) / (2 * pi ** (3/4))
    (0.7682254223260566+0j)

    >>> dedekind_eta(exp(2j * pi / 3))
    (0.7937303350476406-0.10449658101990238j)
    >>> exp(-1j * pi / 24) * 3 ** (1/8) * gamma(1/3) ** (3/2) / (2 * pi)
    (0.7937303350476402-0.10449658101990235j)

    Functional Equations:

    >>> tau0 = -2.7 + 1.1j

    >>> dedekind_eta(tau0 + 1)
    (0.6766410026006273-0.32352780417717425j)
    >>> exp(1j * pi / 12) * dedekind_eta(tau0)
    (0.6766410026006273-0.32352780417717425j)

    >>> dedekind_eta(-1 / tau0)
    (1.272039125360532-0.14802168887940895j)
    >>> sqrt(-1j * tau0) * dedekind_eta(tau0)
    (1.2720391253605303-0.14802168887940903j)

    """
    return exp(PI_I_12 * tau + log_jacobi_theta(3, PI_2 * (tau + 1.0), exp(THREE_PI_I * tau)))


def modular_lambda(tau: complex) -> complex:

    """
    The Modular Lambda function, λ(τ).

    Parameters
    ----------
    tau : float or complex.
          The modular parameter, Im(τ) > 0.

    Returns
    -------
    out : complex.
        The value of λ(τ).

    See Also
    --------
    dedekind_eta, klein_j, inverse_klein_j, euler_phi, eisenstein_e, eisenstein_g

    Implementation
    --------------
    The Modular Lambda function is computed using the connection formula to the Jacobi Theta function [1].

    References
    ----------
    [1] https://dlmf.nist.gov/23.15.E6

    Examples
    --------

    Special Values:

    >>> modular_lambda(1j)
    (0.49999999999999983+0j)
    >>> 1/2 + 0j
    (0.5+0j)

    >>> modular_lambda(exp(2j * pi / 3))
    (0.5000000000000003-0.8660254037844385j)
    >>> (1 + 1j * sqrt(3)) / 2
    (0.5+0.8660254037844386j)

    Functional Equations:

    >>> tau0 = -2.6 + 0.9j

    >>> modular_lambda(tau0 + 2)
    (0.1677236439740376-1.043868285586904j)
    >>> modular_lambda(tau0)
    (0.16772364397403783-1.043868285586904j)

    >>> modular_lambda(-1 / tau0)
    (0.8322763560259618+1.0438682855869041j)
    >>> 1 - modular_lambda(tau0)
    (0.8322763560259622+1.043868285586904j)

    """

    q = exp(PI_I * tau)
    return exp(4.0 * (log_jacobi_theta(2, 0.0, q) - log_jacobi_theta(3, 0.0, q)))


def klein_j(tau: complex) -> complex:

    """
    The Absolute Klein Invariant, J(τ).

    Parameters
    ----------
    tau : float or complex.
          The modular parameter, Im(τ) > 0.

    Returns
    -------
    out : complex.
        The value of J(τ).

    Notes
    -----
    This function defines the Absolute Invariant, J(i) = 1.
    For the standard definition of the j-invariant, use j(τ) = 1728 J(τ).

    See Also
    --------
    dedekind_eta, modular_lambda, inverse_klein_j, euler_phi, eisenstein_e, eisenstein_g

    Implementation
    --------------
    The Absolute Klein Invariant is computed using the connection formula to the Jacobi Theta function [1].

    References
    ----------
    [1] http://functions.wolfram.com/09.50.27.0004.01

    Examples
    --------

    Special Values:

    >>> klein_j(1j)
    (1.0000000000000009+3.1722178103523546e-37j)
    >>> 1 + 0j
    (1+0j)

    >>> klein_j(exp(2j * pi / 3))
    (8.635994148366454e-47-1.153663340430797e-48j)
    >>> 0j
    0j

    Functional Equations:

    >>> tau0 = -0.1 + 1.5j

    >>> klein_j(tau0 + 1)
    (6.239554861080322+4.209580603202556j)
    >>> klein_j(tau0)
    (6.239554861080319+4.209580603202552j)

    >>> klein_j(-1 / tau0)
    (6.239554861080338+4.20958060320256j)
    >>> klein_j(tau0)
    (6.239554861080319+4.209580603202552j)

    Heegner Numbers:

    >>> 1728 * klein_j((1 + 1j * sqrt(7)) / 2)
    (-3375.000000000007-4.1331829471223256e-13j)
    >>> (-15) ** 3 + 0j
    (-3375+0j)

    >>> 1728 * klein_j((1 + 1j * sqrt(43)) / 2)
    (-884735999.9999968-1.083489110490429e-07j)
    >>> (-960) ** 3 + 0j
    (-884736000+0j)

    >>> 1728 * klein_j((1 + 1j * sqrt(163)) / 2)
    (-2.6253741264076736e+17-32.15156020469436j)
    >>> (-640_320) ** 3 + 0j
    (-2.62537412640768e+17+0j)

    """    

    q = exp(PI_I * tau)
    lt2, lt3, lt4 = (8.0 * log_jacobi_theta(2, 0.0, q), 8.0 * log_jacobi_theta(3, 0.0, q), 
                     8.0 * log_jacobi_theta(4, 0.0, q))
    
    return ONE_54 * exp(-lt2 - lt3 - lt4) * (exp(lt2) + exp(lt3) + exp(lt4)) ** 3


def inverse_klein_j(w: complex) -> complex:
    
    """
    The Inverse of the Absolute Klein Invariant, J⁻¹(w).

    Parameters
    ----------
    w : float or complex.
        The argument of the Inverse of the Absolute Klein J⁻¹ function.

    Returns
    -------
    out : complex.
        The value τ, such that w = J(τ).

    Notes
    -----
    For the inverse of the standard j-invariant, use j⁻¹(w) = J⁻¹(w / 1728)

    See Also
    --------
    dedekind_eta, modular_lambda, klein_j, euler_phi, eisenstein_e, eisenstein_g

    Implementation
    --------------
    From the equation [1], we can obtain:

    Let u = 27/4 J - 3 ; t = λ(1 - λ):

    t³ + u t² + 3t - 1 = 0

    We numerically solve for t, and compute λ and then τ:

    λ(1 - λ) = t  ->  λ² - λ + t = 0

    τ = i K(1 - λ) / K(λ)

    Now, we have 6 different values of τ. The principal value of J⁻¹ is when: 

    -1/2 < Re(τ) < 1/2 ; |τ| > 1

    We filter the τ, and return the correct one.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/J-invariant#Inverse_function, Method 1

    Examples
    --------

    Special Values:

    >>> inverse_klein_j(1)
    1j
    >>> 1j
    1j

    >>> inverse_klein_j(0)
    (0.5+0.8660254037844386j)
    >>> (1 + 1j * sqrt(3)) / 2
    (0.5+0.8660254037844386j)

    >>> inverse_klein_j(1331/8)
    1.9999999999999982j
    >>> 2j
    2j

    >>> inverse_klein_j(-512)
    (0.4999999999999989+2.1794494717703485j)
    >>> (1 + 1j * sqrt(19)) / 2
    (0.5+2.179449471770337j)

    >>> inverse_klein_j(-151_931_373_056_000)
    (0.500000000443016+6.383572667401854j)
    >>> (1 + 1j * sqrt(163)) / 2
    (0.5+6.383572667401852j)

    """ 

    if w == 0.0:
        return 0.5 + I_SQRT_3_2

    elif w == 1.0:
        return 1.0j

    tau = empty((6,), dtype=complex128)
    i = 0

    for l1l in roots((1.0, 6.75 * w - 3.0, 3.0, -1.0)):

        lam1, lam2 = roots((1.0, -1.0, l1l))

        tau[i] = 1.0j * elliptick(1.0 - lam1) / elliptick(lam1)
        tau[i + 1] = 1.0j * elliptick(1.0 - lam2) / elliptick(lam2)
        i += 2

    # -1/2 < Re(τ) < 1/2 ; |τ| >= 1
    tau = tau[(-0.5 - TAUEPS < tau.real) & (tau.real < 0.5 + TAUEPS) & (np_abs(tau) > 1.0 - TAUEPS)]

    if tau.size == 0:
        return NANJ
    
    return tau[np_abs(tau.real).argmin()].item()


def euler_phi(q: complex) -> complex:

    """
    The Euler function, ϕ(q).

    Parameters
    ----------
    q : float or complex.
        The elliptic nome, |q| < 1.

    Returns
    -------
    out : complex.
        The value of ϕ(q).

    See Also
    --------
    dedekind_eta, modular_lambda, klein_j, inverse_klein_j, eisenstein_e, eisenstein_g

    Implementation
    --------------
    The Euler Phi function is computed using its relation to the Dedekind Eta function [1].

    When the Dedekind Eta is represented using the Jacobi Theta function [2], the identity simplifies to:

    ϕ(q) = θ₃(π/2 - i/4 log(q), q √q)

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Euler_function#Properties

    [2] https://dlmf.nist.gov/23.15.E9

    Examples
    --------

    Special Values:

    >>> euler_phi(exp(-pi))
    (0.9549187899876741-5.063446049288748e-18j)
    >>> exp(pi/24) * gamma(1/4) / (2 ** (7/8) * pi ** (3/4))
    (0.9549187899876741+0j)

    >>> euler_phi(exp(-8 * pi))
    (0.9999999999878384-1.4893611496646012e-27j)
    >>> exp(pi/3) * gamma(1/4) / (2 ** (29/16) * pi ** (3/4)) * (sqrt(2) - 1) ** (1/4)
    (0.9999999999878385+0j)

    Identity:

    >>> a = 3.6
    >>> b = pi ** 2 / a

    >>> a ** (1/4) * exp(-a/12) * euler_phi(exp(-2 * a))
    (1.0196771309633983-9.322953185418137e-20j)
    >>> b ** (1/4) * exp(-b/12) * euler_phi(exp(-2 * b))
    (1.0196771309633985-5.190331583118252e-19j)

    """
    if q == 1.0:
        return 0.0 + 0.0j

    return jacobi_theta(3, PI_2 - 0.25j * log(q), q * sqrt(q))


def eisenstein_e2_tau(tau: complex) -> complex:

    tau -= round(tau.real)

    if tau.imag < TAU_MIN:
        # E2(-1/τ) = τ^2 E2(τ) - 6iτ/pi
        # E2(τ) = E2(-1/τ)/τ^2 + 6i/(pi τ)
        tau = 1.0 / tau
        return tau * tau * eisenstein_e2_tau(-tau) + SIX_OVER_PI_I * tau

    s = 0.0
    q2 = qn = exp(TWO_PI_I * tau)

    for n in range(1, MAXITER + 1):

        t = n * qn / (1.0 - qn)
        s += t

        if abs(t) < max(EPS, EPS * abs(s)):
            break

        qn *= q2

    return 1.0 - 24.0 * s


def eisenstein_e2(q: complex) -> complex:

    if abs(q) >= 1.0:
        # θ(n, z, q) is only defined for |q| < 1
        return NANJ

    return eisenstein_e2_tau(-I_PI_INV * log(q))


def dedekind_eta_prime(tau: complex) -> complex:
    """
    The Derivative of the Dedekind Eta function, η'(τ).

    Parameters
    ----------
    tau : float or complex.
          The modular parameter, Im(τ) > 0.

    Returns
    -------
    out : complex.
        The value of η'(τ).

    See Also
    --------
    dedekind_eta, modular_lambda, klein_j, inverse_klein_j, euler_phi, eisenstein_e, eisenstein_g

    Implementation
    --------------
    A simple representation of η'(τ) in terms of η(τ) and E₂(τ) can be obtained from [1]:

    η'(τ) = πi/12 E₂(τ) η(τ) 

    References
    ----------
    [1] https://mathworld.wolfram.com/DedekindEtaFunction.html, formula (10)

    Examples
    --------

    Special Values:

    >>> dedekind_eta_prime(1j)
    (4.392263081058748e-20+0.19205635558151415j)
    >>> 1j * gamma(1/4) / (8 * pi ** (3/4))
    0.19205635558151415j

    >>> dedekind_eta_prime(exp(2j * pi / 3))
    (0.03016556459061806+0.2291302113018636j)
    >>> 1j * sqrt(3) * exp(-1j * pi / 24) * 3 ** (1/8) * gamma(1/3) ** (3/2) / (12 * pi)
    (0.030165564590618078+0.22913021130186345j)

    """
    return PI_I_12 * eisenstein_e2_tau(tau) * dedekind_eta(tau)


def eisenstein_e4(q: complex) -> complex:
    return 0.5 * (exp(8.0 * log_jacobi_theta(2, 0.0, q)) + exp(8.0 * log_jacobi_theta(3, 0.0, q)) + 
                  exp(8.0 * log_jacobi_theta(4, 0.0, q)))


def eisenstein_g4(q: complex) -> complex:
    return TWO_ZETA_4 * eisenstein_e4(q)


def eisenstein_e6(q: complex) -> complex:
    t24, t34, t44 = (exp(4.0 * log_jacobi_theta(2, 0.0, q)), exp(4.0 * log_jacobi_theta(3, 0.0, q)), 
                     exp(4.0 * log_jacobi_theta(4, 0.0, q)))
    return 0.5 * (t24 + t34) * (t34 + t44) * (t44 - t24)


def eisenstein_g6(q: complex) -> complex:
    return TWO_ZETA_6 * eisenstein_e6(q)


def eisenstein_g(n: int, q: complex) -> complex:

    """
    The Eisenstein Series, Gn(q).

    Parameters
    ----------
    n : positive even integer, 2, 4, 6, ...
        The weight of the Eisenstein Series.
    q : float or complex.
        The elliptic nome, |q| < 1.

    Returns
    -------
    out : complex.
        The value of Gn(q).

    See Also
    --------
    dedekind_eta, modular_lambda, klein_j, inverse_klein_j, euler_phi, eisenstein_e

    Implementation
    --------------
    For n = 2, 4 and 6, Gn(q) is computed using En(q) [1]. For the implementation of En(q),
    see eisenstein_e.

    For n > 6, Gn(q) is computed iteratively using the recurrence relation [2], with G₄ and G₆ 
    as the base cases.

    References
    ----------
    [1] https://mathworld.wolfram.com/EisensteinSeries.html, formula (5)

    [2] https://en.wikipedia.org/wiki/Eisenstein_series#Recurrence_relation

    Examples
    --------

    SL(2, ℤ) Invariance:

    >>> tau0 = 0.4 + 0.1j
    >>> tau1 = -1 / tau0
    
    >>> eisenstein_g(4, nome_q(tau=tau1))
    (-6.199953429133921-7.607439221069523j)
    >>> tau0 ** 4 * eisenstein_g(4, nome_q(tau=tau0))
    (-6.1999534291339184-7.607439221069527j)

    >>> eisenstein_g(6, nome_q(tau=tau1))
    (19.85844495273096+1.8283678216121886j)
    >>> tau0 ** 6 * eisenstein_g(6, nome_q(tau=tau0))
    (19.858444952730963+1.828367821612192j)

    >>> eisenstein_g(12, nome_q(tau=tau1))
    (173.8555540665175-42.31300541512066j)
    >>> tau0 ** 12 * eisenstein_g(12, nome_q(tau=tau0))
    (173.8555540665176-42.31300541512047j)

    Quasimodular property of G₂:

    >>> tau1 ** 2 * eisenstein_g(2, nome_q(tau=tau1))
    (27.33679420469431-5.076094083576979j)
    >>> eisenstein_g(2, nome_q(tau=tau0)) + 2j * pi * tau1
    (27.336794204694318-5.076094083576988j)

    """

    if n & 1 or n < 2:
        error_message: str = f"Invalid weight ({n}). The weight must be a positive, even integer, n = 2, 4, 6, 8, ..."
        raise ValueError(error_message)

    if n == 2:
        return TWO_ZETA_2 * eisenstein_e2(q)
    
    elif n == 4:
        return eisenstein_g4(q)
    
    elif n == 6:
        return eisenstein_g6(q)

    # https://en.wikipedia.org/wiki/Eisenstein_series#Recurrence_relation
    k = n // 2 - 2
    d = empty((k + 1,), dtype=complex128)
    d[0], d[1] = 3.0 * eisenstein_g4(q), 5.0 * eisenstein_g6(q)

    for m in range(k - 1):
        d[m + 2] = (3 * m + 6) * (comb(m, arange(m + 1, dtype=int32)) * d[:m + 1] * d[m::-1]).sum() / (2 * m + 9)

    return d[-1].item() / ((2 * k + 3) * factorial(k))


def eisenstein_e(n: int, q: complex) -> complex:

    """
    The Eisenstein Series, En(q).

    Parameters
    ----------
    n : positive even integer, 2, 4, 6, ...
        The weight of the Eisenstein Series.
    q : float or complex.
        The elliptic nome, |q| < 1.

    Returns
    -------
    out : complex.
        The value of En(q).

    See Also
    --------
    dedekind_eta, modular_lambda, klein_j, inverse_klein_j, euler_phi, eisenstein_g

    Implementation
    --------------
    For n = 4, 6 the identities for E₄ and E₆ in terms of the Jacobi Theta functions are used [1].

    For n = 2, when |q| is small, the q-expansion in [2] is used. Otherwise, we use the transformations derived
    from the identity in [3]:

    E₂(τ + 1) = E₂(τ)

    E₂(τ) = E₂(-1/τ) / τ² + 6 i / (π τ)

    to shift τ to a region, where the q-expansion converges fast, similarly to the implementation 
    of the logarithm of the Jacobi Theta function.

    For n > 6, the recurrence relation of Gn(q) is used [4]. Gn(q) is then converted to En(q) using [5].

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Eisenstein_series#Theta_functions

    [2] https://en.wikipedia.org/wiki/Eisenstein_series#Fourier_series

    [3] https://math.stackexchange.com/questions/3237800/weight-2-eisenstein-series-transformation

    [4] https://en.wikipedia.org/wiki/Eisenstein_series#Recurrence_relation

    [5] https://mathworld.wolfram.com/EisensteinSeries.html, formula (5)

    See also: https://mathematica.stackexchange.com/a/89682

    Examples
    --------

    SL(2, ℤ) Invariance:

    >>> tau0 = -0.7 + 0.2j
    >>> tau1 = -1 / tau0
    
    >>> eisenstein_e(4, nome_q(tau=tau1))
    (-15.598281754025983+5.481008219087166j)
    >>> tau0 ** 4 * eisenstein_e(4, nome_q(tau=tau0))
    (-15.598281754025932+5.481008219087118j)

    >>> eisenstein_e(6, nome_q(tau=tau1))
    (30.189838209085877+61.560209971655674j)
    >>> tau0 ** 6 * eisenstein_e(6, nome_q(tau=tau0))
    (30.189838209085607+61.56020997165541j)

    >>> eisenstein_e(12, nome_q(tau=tau1))
    (-2566.2422171441704+3792.963943315567j)
    >>> tau0 ** 12 * eisenstein_e(12, nome_q(tau=tau0))
    (-2566.2422171441594+3792.9639433155135j)

    Quasimodular property of E₂:

    >>> eisenstein_e(2, nome_q(tau=tau1))
    (2.286807516445588-1.5292653160493548j)
    >>> tau0 ** 2 * eisenstein_e(2, nome_q(tau=tau0)) - 6j * tau0 / pi
    (2.286807516445587-1.5292653160493563j)

    """

    if n & 1 or n < 2:
        error_message: str = f"Invalid weight ({n}). The weight must be a positive, even integer, n = 2, 4, 6, 8, ..."
        raise ValueError(error_message)

    if n == 2:
        return eisenstein_e2(q)
    
    elif n == 4:
        return eisenstein_e4(q)
    
    elif n == 6:
        return eisenstein_e6(q)

    return 0.5 * eisenstein_g(n, q) / zeta(n)


def jacobi_ellipfun(pq: JacobiEllipfunVariants, u: complex, m: complex) -> complex:

    """
    The Jacobi elliptic functions.

    Computes any of the Jacobi elliptic functions pq(u, m), where p and q are any of the symbols c, d, n or s,
    or alternatively, when am is specified, the Jacobi Amplitude is computed.

    Parameters
    ----------
    u : float or complex.
        The argument of the Jacobi elliptic function.
    m : float or complex.
        The elliptic parameter.

    Returns
    -------
    out : complex.
        The value of the specified Jacobi elliptic function.

    See Also
    --------
    inverse_jacobi_ellipfun, neville_theta_c, neville_theta_d, neville_theta_n, neville_theta_s

    Implementation
    --------------
    cc(u, m) = dd(u, m) = nn(u, m) = ss(u, m) = 1

    cd, cn, cs, dc, dn, ds, nc, nd, ns, sc, sd, sn are computed using their connection formulas to the 
    Jacobi Theta function. [cd] — [sn]

    Special values for m = 0 and m = 1 are given in [13].

    When Re(u) ∈ (-2 K(m), 2 K(m)), am(u, m) is computed like:
    
    sin(am(u, m)) = sn(u, m) ; cos(am(u, m)) = cn(u, m)  ->  tan(am(u, m)) = sc(u, m)  ->

    am(u, m) = arctan(sc(u, m))  ->  am(u, m) = arctan2(sn(u, m), cn(u, m))

    When outside this interval, the function is shifted by π for every interval Re(u) has passed
    to keep the function smooth.

    References
    ----------
    [cd] http://functions.wolfram.com/09.25.27.0024.02

    [cn] http://functions.wolfram.com/09.26.27.0026.02

    [cs] http://functions.wolfram.com/09.27.27.0025.01
    
    [dc] http://functions.wolfram.com/09.28.27.0025.01

    [dn] http://functions.wolfram.com/09.29.27.0027.01

    [ds] http://functions.wolfram.com/09.30.27.0023.01

    [nc] http://functions.wolfram.com/09.31.27.0024.02

    [nd] http://functions.wolfram.com/09.32.27.0025.02

    [ns] http://functions.wolfram.com/09.33.27.0025.01

    [sc] http://functions.wolfram.com/09.34.27.0025.02

    [sd] http://functions.wolfram.com/09.35.27.0023.02

    [sn] http://functions.wolfram.com/09.36.27.0027.02

    [13] https://en.wikipedia.org/wiki/Jacobi_elliptic_functions#Special_values

    Examples
    --------

    Half K Formula:

    >>> m0 = 1.4 - 0.6j
    >>> hk = elliptick(m0) / 2

    >>> jacobi_ellipfun("sn", hk, m0)
    (0.7697108009080443-0.1928061807497407j)
    >>> 1 / sqrt(1 + sqrt(1 - m0))
    (0.7697108009080446-0.1928061807497408j)

    >>> jacobi_ellipfun("dn", hk, m0)
    (0.7905300178000148+0.4735445073653465j)
    >>> (1 - m0) ** (1/4)
    (0.7905300178000145+0.47354450736534637j)

    Square Identities:

    >>> u0 = -4.9 + 1.1j

    >>> jacobi_ellipfun("cn", u0, m0) ** 2 + jacobi_ellipfun("sn", u0, m0) ** 2
    (0.9999999999999991-2.7478019859472624e-15j)
    >>> 1 + 0j
    (1+0j)

    >>> jacobi_ellipfun("dn", u0, m0) ** 2 + m0 * jacobi_ellipfun("sn", u0, m0) ** 2
    (1.0000000000000002-8.777700788442644e-16j)
    >>> 1 + 0j
    (1+0j)

    Amplitude:

    >>> sin(jacobi_ellipfun("am", u0, m0))
    (0.7658998799291391+0.14419270885252666j)
    >>> jacobi_ellipfun("sn", u0, m0)
    (0.7658998799291392+0.14419270885252658j)

    >>> cos(jacobi_ellipfun("am", u0, m0))
    (-0.6787227072554022+0.16271325125306313j)
    >>> jacobi_ellipfun("cn", u0, m0)
    (-0.6787227072554018+0.16271325125306518j)

    """

    if pq[0] == pq[1] and len(pq) == 2:

        # cc = dd = nn = ss = 1

        if pq[0] in ("c", "d", "n", "s"):
            return 1.0 + 0.0j
        else:
            raise ValueError(wrong_ellipfun_variant(pq))

    if m == 0.0:

        match pq:

            case "am":
                return u

            case "cd":
                return cos(u)
            case "cn":
                return cos(u)
            case "cs":
                return cot(u)
            
            case "dc":
                return sec(u)
            case "dn":
                return 1.0 + 0.0j
            case "ds":
                return csc(u)
            
            case "nc":
                return sec(u)
            case "nd":
                return 1.0 + 0.0j
            case "ns":
                return csc(u)
            
            case "sc":
                return tan(u)
            case "sd":
                return sin(u)
            case "sn":
                return sin(u)

            case _:
                raise ValueError(wrong_ellipfun_variant(pq))
    
    elif m == 1.0:

        match pq:
            
            case "am":
                return 2.0 * arctan(exp(u)) - PI_2

            case "cd":
                return 1.0 + 0.0j
            case "cn":
                return sech(u)
            case "cs":
                return csch(u)
            
            case "dc":
                return 1.0 + 0.0j
            case "dn":
                return sech(u)
            case "ds":
                return csch(u)
            
            case "nc":
                return cosh(u)
            case "nd":
                return cosh(u)
            case "ns":
                return coth(u)
            
            case "sc":
                return sinh(u)
            case "sd":
                return sinh(u)
            case "sn":
                return tanh(u)

            case _:
                raise ValueError(wrong_ellipfun_variant(pq))

    if pq == "am":
        
        km2 = 2.0 * elliptick(m)
        km2i = 1.0 / km2
        q = exp(-TWO_PI * km2i * elliptick(m1 := (1.0 - m)))

        # n = round(Re(u) / Re(2K(m)))
        n = round(u.real / km2.real)
        w = PI * (u - km2 * n) * km2i

        lt4 = log_jacobi_theta(4, w, q)
        m14 = m ** -0.25

        # arctan(sc(u - 2K(m)n, m)) + πn = arctan2(sn(u - 2K(m)n, m) ; cn(u - 2K(m)n, m)) + πn
        return zatan2(m14 * exp(log_jacobi_theta(1, w, q) - lt4), 
                      m1 ** 0.25 * m14 * exp(log_jacobi_theta(2, w, q) - lt4)) + PI * n

    kmi = PI / elliptick(m)
    q = exp(-kmi * elliptick(m1 := 1.0 - m))
    w = 0.5 * kmi * u

    match pq:
        
        case "cd":
            return m ** -0.25 * exp(log_jacobi_theta(2, w, q) - log_jacobi_theta(3, w, q))
        case "cn":
            return (m1 / m) ** 0.25 * exp(log_jacobi_theta(2, w, q) - log_jacobi_theta(4, w, q))
        case "cs":
            return m1 ** 0.25 * exp(log_jacobi_theta(2, w, q) - log_jacobi_theta(1, w, q))
        
        case "dc":
            return m ** 0.25 * exp(log_jacobi_theta(3, w, q) - log_jacobi_theta(2, w, q))
        case "dn":
            return m1 ** 0.25 * exp(log_jacobi_theta(3, w, q) - log_jacobi_theta(4, w, q))
        case "ds":
            return (m * m1) ** 0.25 * exp(log_jacobi_theta(3, w, q) - log_jacobi_theta(1, w, q))
        
        case "nc":
            return (m / m1) ** 0.25 * exp(log_jacobi_theta(4, w, q) - log_jacobi_theta(2, w, q))
        case "nd":
            return m1 ** -0.25 * exp(log_jacobi_theta(4, w, q) - log_jacobi_theta(3, w, q))
        case "ns":
            return m ** 0.25 * exp(log_jacobi_theta(4, w, q) - log_jacobi_theta(1, w, q))
        
        case "sc":
            return m1 ** -0.25 * exp(log_jacobi_theta(1, w, q) - log_jacobi_theta(2, w, q))
        case "sd":
            return (m * m1) ** -0.25 * exp(log_jacobi_theta(1, w, q) - log_jacobi_theta(3, w, q))
        case "sn":
            return m ** -0.25 * exp(log_jacobi_theta(1, w, q) - log_jacobi_theta(4, w, q))
        
        case _:
            raise ValueError(wrong_ellipfun_variant(pq))


def inverse_jacobi_ellipfun(pq: JacobiEllipfunVariants, w: complex, m: complex) -> complex:

    """
    The Inverse Jacobi elliptic functions.

    Computes any of the Inverse Jacobi elliptic functions pq⁻¹(u, m), where p and q are any of the 
    symbols c, d, n or s, or alternatively, when am is specified, the Inverse of the Jacobi Amplitude 
    is computed.

    Parameters
    ----------
    w : float or complex.
        The argument of the Inverse Jacobi elliptic function.
    m : float or complex.
        The elliptic parameter.

    Returns
    -------
    out : complex.
        The value u, such that w = pq(u, m).

    See Also
    --------
    jacobi_ellipfun, neville_theta_c, neville_theta_d, neville_theta_n, neville_theta_s

    Implementation
    --------------
    cd⁻¹, cn⁻¹, dn⁻¹, sc⁻¹, sd⁻¹, sn⁻¹ are computed using the Incomplete Elliptic integral of the 
    first kind, F(ϕ, m) [cd] — [sn].

    cs⁻¹, dc⁻¹, ds⁻¹, nc⁻¹, nd⁻¹, ns⁻¹ are computed using the other functions by:

    qp⁻¹(w, m) = pq⁻¹(1 / w, m)

    Special values for m = 0 and m = 1 are given by inverting [7].

    The inverse amplitude is directly computed from its definition [8]:

    am⁻¹(w, m) = F(w, m)

    References
    ----------
    [cd] http://functions.wolfram.com/09.37.27.0013.01

    [cn] http://functions.wolfram.com/09.38.27.0016.01

    [dn] http://functions.wolfram.com/09.41.27.0016.02

    [sc] http://functions.wolfram.com/09.46.27.0013.01

    [sd] http://functions.wolfram.com/09.47.27.0013.01

    [sn] http://functions.wolfram.com/09.48.27.0015.02

    [7]  https://en.wikipedia.org/wiki/Jacobi_elliptic_functions#Special_values

    [8]  http://functions.wolfram.com/09.24.02.0001.01

    Examples
    --------

    Inverse Property:
    >>> w0, m0 = -1.4 - 2j, 0.4j

    >>> w0
    (-1.4-2j)
    >>> jacobi_ellipfun("sn", inverse_jacobi_ellipfun("sn", w0, m0), m0)
    (-1.400000000000001-1.9999999999999993j)
    >>> jacobi_ellipfun("cn", inverse_jacobi_ellipfun("cn", w0, m0), m0)
    (-1.400000000000001-2.000000000000002j)
    >>> jacobi_ellipfun("dn", inverse_jacobi_ellipfun("dn", w0, m0), m0)
    (-1.4000000000000032-2.0000000000000013j)
    >>> jacobi_ellipfun("am", inverse_jacobi_ellipfun("am", w0, m0), m0)
    (-1.3999999999999977-1.9999999999999976j)

    """

    if pq[0] == pq[1] and len(pq) == 2:

        # cc = dd = nn = ss = 1

        if pq[0] in ("c", "d", "n", "s"):
            return 0.0 + 0.0j if w == 1.0 else NANJ
        else:
            raise ValueError(wrong_ellipfun_variant(pq))

    if m == 0.0:

        match pq:

            case "cd":
                return arccos(w)
            case "cn":
                return arccos(w)
            case "cs":
                return arccot(w)
            
            case "dc":
                return arcsec(w)
            case "dn":
                return 0.0 + 0.0j if w == 1.0 else NANJ
            case "ds":
                return arccsc(w)
            
            case "nc":
                return arcsec(w)
            case "nd":
                return NANJ
            case "ns":
                return arccsc(w)
            
            case "sc":
                return arctan(w)
            case "sd":
                return arcsin(w)
            case "sn":
                return arcsin(w)

            case _:
                raise ValueError(wrong_ellipfun_variant(pq))
    
    elif m == 1.0:

        match pq:

            case "cd":
                return NANJ
            case "cn":
                return arcsech(w)
            case "cs":
                return arccsch(w)
            
            case "dc":
                return NANJ
            case "dn":
                return arcsech(w)
            case "ds":
                return arccsch(w)
            
            case "nc":
                return arccosh(w)
            case "nd":
                return arccosh(w)
            case "ns":
                return arccoth(w)
            
            case "sc":
                return arcsinh(w)
            case "sd":
                return arcsinh(w)
            case "sn":
                return arctanh(w)

            case _:
                raise ValueError(wrong_ellipfun_variant(pq))

    match pq:

        case "am":
            return ellipticf(w, m)

        case "cd":
            return elliptick(m) - ellipticf(arcsin(w), m)
        case "cn":
            return ellipticf(arccos(w), m)
        case "cs":
            if w == 0.0:
                m1 = 1.0 / (m - 1.0)
                return sqrt(-m1) * elliptick(m * m1)
            return inverse_jacobi_ellipfun("sc", 1.0 / w, m)
        
        case "dc":
            if w == 0.0:
                m = 1.0 / m
                return sqrt(m) * elliptick(m)
            return inverse_jacobi_ellipfun("cd", 1.0 / w, m)
        case "dn":
            m = 1.0 / (1.0 - m)
            return -sqrt(-m) * (elliptick(m) - ellipticf(arcsin(w), m))
        case "ds":
            if w == 0.0:
                m = 1.0 / m
                return sqrt(m) * elliptick(m)
            return inverse_jacobi_ellipfun("sd", 1.0 / w, m)
        
        case "nc":
            if w == 0.0:
                mi = 1.0 / m
                return 1.0j * sqrt(mi) * elliptick(mi * (m - 1.0))
            return inverse_jacobi_ellipfun("cn", 1.0 / w, m)
        case "nd":
            if w == 0.0:
                return 1.0j * elliptick(1.0 - m)
            return inverse_jacobi_ellipfun("dn", 1.0 / w, m)
        case "ns":
            if w == 0.0:
                mi = 1.0 / m
                return elliptick(m) - sqrt(mi) * elliptick(mi)
            return inverse_jacobi_ellipfun("sn", 1.0 / w, m)

        case "sc":
            return -1.0j * ellipticf(1.0j * arcsinh(w), 1.0 - m)
        case "sd":
            s = sqrt(m)
            return -1.0j * ellipticf(1.0j * arcsinh(s * w), (m - 1.0) / m) / s
        case "sn":
            return ellipticf(arcsin(w), m)
        
        case _:
            raise ValueError(wrong_ellipfun_variant(pq))


def sinlem(z: complex) -> complex:

    """
    The Lemniscate Sine function, sinlem(z).

    Parameters
    ----------
    z : float or complex.
        The argument of the Lemniscate Sine function.

    Returns
    -------
    out : complex.
        The value of sinlem(z).

    See Also
    --------
    coslem, arcsinlem, arccoslem, sinhlem, coshlem, arcsinhlem, arccoshlem

    Implementation
    --------------
    The Lemniscate Sine is computed using its connection formula to the Jacobi sd function [1], which is
    then efficiently computed using the Jacobi Theta function [2]. 

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Lemniscate_elliptic_functions#Relation_to_other_functions

    [2] http://functions.wolfram.com/09.35.27.0023.02

    Examples
    --------

    sinlem is 2ϖ—periodic, with sinlem(0) = 0:

    >>> z0 = 5.1 - 2.5j

    >>> sinlem(z0)
    (0.14413749339680135-0.12206628709072101j)
    >>> sinlem(z0 + 2 * lem)
    (0.1441374933968007-0.1220662870907209j)

    >>> sinlem(0)
    (-5.1106154506931017e-17-6.258698853364314e-33j)
    >>> 0j
    0j

    Identity with coslem:

    >>> coslem(z0) ** 2 + sinlem(z0) ** 2 + coslem(z0) ** 2 * sinlem(z0) ** 2
    (1+0j)
    >>> 1 + 0j
    (1+0j)

    Special Values:

    >>> sinlem(lem / 2)
    (1+7.930544525880871e-38j)
    >>> 1 + 0j
    (1+0j)

    >>> sinlem(lem / 4)
    (0.6435942529055827+2.3314943767109854e-38j)
    >>> sqrt(sqrt(2) - 1)
    (0.6435942529055827+0j)

    """
    w = LEM_W * z
    return exp(log_jacobi_theta(1, w, LEM_Q) - log_jacobi_theta(3, w, LEM_Q))


def coslem(z: complex) -> complex:

    """
    The Lemniscate Cosine function, coslem(z).

    Parameters
    ----------
    z : float or complex.
        The argument of the Lemniscate Cosine function.

    Returns
    -------
    out : complex.
        The value of coslem(z).

    See Also
    --------
    sinlem, arcsinlem, arccoslem, sinhlem, coshlem, arcsinhlem, arccoshlem

    Implementation
    --------------
    The Lemniscate Cosine is computed using its connection formula to the Jacobi cn function [1], which is
    then efficiently computed using the Jacobi Theta function [2]. 

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Lemniscate_elliptic_functions#Relation_to_other_functions

    [2] http://functions.wolfram.com/09.26.27.0026.02

    Examples
    --------

    coslem is 2ϖ—periodic, with coslem(0) = 1:

    >>> z0 = -4.2 + 0.9j

    >>> coslem(z0)
    (0.19723012872146797-0.8868304574593707j)
    >>> coslem(z0 + 2 * lem)
    (0.197230128721467-0.886830457459371j)

    >>> coslem(0)
    (1+7.930544525880871e-38j)
    >>> 1 + 0j
    (1+0j)

    Duplication Formula:

    >>> coslem(2 * z0)
    (0.9030775477309373-0.995631030969559j)
    >>> (coslem(z0) ** 4 + 2 * coslem(z0) ** 2 - 1) / (1 + 2 * coslem(z0) ** 2 - coslem(z0) ** 4)
    (0.9030775477309331-0.9956310309695584j)

    Special Values:

    >>> coslem(lem / 2)
    (5.1106154506931017e-17+0j)
    >>> 0j
    0j

    >>> coslem(lem / 6)
    (0.8253787243642845+1.1049661505198986e-22j)
    >>> (2 * sqrt(3) - 3) ** (1/4)
    (0.8253787243642843+0j)

    """
    w = LEM_W * z
    return exp(log_jacobi_theta(2, w, LEM_Q) - log_jacobi_theta(4, w, LEM_Q))


def arcsinlem(w: complex) -> complex:

    """
    The Inverse Lemniscate Sine function, arcsinlem(z).

    Parameters
    ----------
    w : float or complex.
        The argument of the Inverse Lemniscate Sine function.

    Returns
    -------
    out : complex.
        The value z, such that w = sinlem(z).

    See Also
    --------
    sinlem, coslem, arccoslem, sinhlem, coshlem, arcsinhlem, arccoshlem

    Implementation
    --------------
    The Inverse Lemniscate Sine is computed by using its integral definition [1]:

    arcsinlem(z) = F(arcsin(z), -1)

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Lemniscate_elliptic_functions#Inverse_functions

    Examples
    --------

    Special Values:

    >>> arcsinlem((2 * sqrt(3) - 3) ** (1/4))
    (0.87401918476404+0j)
    >>> lem / 3
    0.8740191847640398

    >>> arcsinlem(-(sqrt(3) + 1 - 12 ** (1/4)) / 2)
    (-0.43700959238202003+0j)
    >>> -lem/6
    -0.4370095923820199

    Inverse Property:

    >>> z0 = -1.1 + 5.2j

    >>> sinlem(arcsinlem(z0))
    (-1.1000000000000028+5.200000000000003j)
    >>> z0
    (-1.1+5.2j)

    """
    return ellipticf(arcsin(w), -1.0)


def arccoslem(w: complex) -> complex:

    """
    The Inverse Lemniscate Cosine function, arccoslem(z).

    Parameters
    ----------
    w : float or complex.
        The argument of the Inverse Lemniscate Cosine function.

    Returns
    -------
    out : complex.
        The value z, such that w = coslem(z).

    See Also
    --------
    sinlem, coslem, arcsinlem, sinhlem, coshlem, arcsinhlem, arccoshlem

    Implementation
    --------------
    The Inverse Lemniscate Cosine is computed using arcsinlem(z) [1].

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Lemniscate_elliptic_functions#Inverse_functions

    Examples
    --------

    Special Values:

    >>> arccoslem(sqrt(sqrt(2) - 1))
    (0.6555143885730297+0j)
    >>> lem/4
    0.6555143885730299

    >>> arccoslem(-(2 * sqrt(3) - 3) ** (1/4))
    (2.1850479619101+0j)
    >>> 5/6 * lem
    2.1850479619101

    Inverse Property:

    >>> z0 = 2.7 - 0.8j

    >>> coslem(arccoslem(z0))
    (2.700000000000002-0.7999999999999998j)
    >>> z0
    (2.7-0.8j)

    """
    return HALF_LEM - arcsinlem(w)


def sinhlem(z: complex) -> complex:

    """
    The Hyperbolic Lemniscate Sine function, sinhlem(z).

    Parameters
    ----------
    z : float or complex.
        The argument of the Hyperbolic Lemniscate Sine function.

    Returns
    -------
    out : complex.
        The value of sinhlem(z).

    See Also
    --------
    sinlem, coslem, arcsinlem, arccoslem, coshlem, arcsinhlem, arccoshlem

    Implementation
    --------------
    The Hyperbolic Lemniscate Sine is computed using its connection formula to the Jacobi elliptic 
    functions sn and cd [1], which are then efficiently computed using the Jacobi Theta function [2], [3]. 

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Lemniscate_elliptic_functions#Fundamental_information

    [2] http://functions.wolfram.com/09.36.27.0027.02

    [3] http://functions.wolfram.com/09.25.27.0024.02

    Examples
    --------

    Connection formulas to sinlem and coslem:

    >>> z0 = 1.1 + 0.2j

    >>> sinhlem(sqrt(2) * z0)
    (1.7697296566176155+1.6782668253493926j)
    >>> (1 + coslem(z0) ** 2) * sinlem(z0) / (sqrt(2) * coslem(z0))
    (1.7697296566176148+1.6782668253493918j)

    >>> sinhlem(z0)
    (1.2050750325279704+0.3616416013455245j)
    >>> (1 - 1j) / sqrt(2) * sinlem((1 + 1j) / sqrt(2) * z0)
    (1.2050750325279704+0.3616416013455245j)

    Identity with coshlem:

    >>> sinhlem(z0) * coshlem(z0)
    (1+5.551115123125783e-17j)
    >>> 1 + 0j
    (1+0j)

    Special Values:

    >>> sinhlem(5 * lem / (6 * sqrt(2)))
    (3.2331653360238417+1.7947666758063533e-16j)
    >>> (sqrt(2 * sqrt(3) + 3) + 1) * (1 + (2 * sqrt(3) - 3) ** (1/4)) / 2
    (3.2331653360238444+0j)

    """
    w = LEMH_W * z
    return exp(log_jacobi_theta(1, w, LEM_Q) + log_jacobi_theta(3, w, LEM_Q) - log_jacobi_theta(2, w, LEM_Q) - 
               log_jacobi_theta(4, w, LEM_Q))


def coshlem(z: complex) -> complex:

    """
    The Hyperbolic Lemniscate Cosine function, coshlem(z).

    Parameters
    ----------
    z : float or complex.
        The argument of the Hyperbolic Lemniscate Cosine function.

    Returns
    -------
    out : complex.
        The value of coshlem(z).

    See Also
    --------
    sinlem, coslem, arcsinlem, arccoslem, sinhlem, arcsinhlem, arccoshlem

    Implementation
    --------------
    The Hyperbolic Lemniscate Cosine is computed using its connection formula to the Jacobi elliptic 
    functions cd and sn [1], which are then efficiently computed using the Jacobi Theta function [2], [3]. 

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Lemniscate_elliptic_functions#Fundamental_information

    [2] http://functions.wolfram.com/09.25.27.0024.02

    [3] http://functions.wolfram.com/09.36.27.0027.02

    Examples
    --------

    Connection formulas to sinlem and coslem:

    >>> z0 = -2.3 + 1.4j

    Identity with coshlem:

    >>> sinhlem(z0) * coshlem(z0)
    (1+7.771561172376096e-16j)
    >>> 1 + 0j
    (1+0j)

    Special Values:

    >>> coshlem(lem / sqrt(2))
    (3.3436328052638403e-16+1.226738397894171e-52j)
    >>> 0j
    0j

    >>> coshlem(3 * sqrt(2) * lem / 4)
    (-1.0000000000000009+1.2246467991473542e-16j)
    >>> -1 + 0j
    (-1+0j)

    """
    w = LEMH_W * z
    return exp(log_jacobi_theta(2, w, LEM_Q) + log_jacobi_theta(4, w, LEM_Q) - log_jacobi_theta(1, w, LEM_Q) - 
               log_jacobi_theta(3, w, LEM_Q))


def arcsinhlem(w: complex) -> complex:

    """
    The Inverse Hyperbolic Lemniscate Sine function, arcsinhlem(z).

    Parameters
    ----------
    w : float or complex.
        The argument of the Inverse Hyperbolic Lemniscate Sine function.

    Returns
    -------
    out : complex.
        The value z, such that w = sinhlem(z).

    See Also
    --------
    sinlem, coslem, arcsinlem, arccoslem, sinhlem, coshlem, arccoshlem

    Implementation
    --------------
    The Inverse Hyperbolic Lemniscate Sine is computed using F(ϕ, m) [1].

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Lemniscate_elliptic_functions#Inverse_functions

    Examples
    --------

    Special Values:

    >>> arcsinhlem((sqrt(sqrt(2) + 1) - 1) / 2 ** (1/4))
    (0.4635186693253431+0j)
    >>> lem / (4 * sqrt(2))
    (0.4635186693253429+0j)

    >>> arcsinhlem((sqrt(2 * sqrt(3) + 3) + 1) * (1 - (2 * sqrt(3) - 3) ** (1/4)) / 2)
    (0.3090124462168954+0j)
    >>> lem / (6 * sqrt(2))
    (0.3090124462168953+0j)

    Inverse Property:

    >>> z0 = -1.7 + 0.1j

    >>> sinhlem(arcsinhlem(z0))
    (-1.6999999999999997+0.10000000000000032j)
    >>> z0
    (-1.7+0.1j)
    
    """
    return 0.5 * ellipticf(2.0 * arctan(w), 0.5)


def arccoshlem(w: complex) -> complex:

    """
    The Inverse Hyperbolic Lemniscate Cosine function, arccoshlem(z).

    Parameters
    ----------
    w : float or complex.
        The argument of the Inverse Hyperbolic Lemniscate Cosine function.

    Returns
    -------
    out : complex.
        The value z, such that w = coshlem(z).

    See Also
    --------
    sinlem, coslem, arcsinlem, arccoslem, sinhlem, coshlem, arcsinhlem

    Implementation
    --------------
    The Inverse Hyperbolic Lemniscate Cosine is computed using F(ϕ, m) [1].

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Lemniscate_elliptic_functions#Inverse_functions

    Examples
    --------

    Special Values:

    >>> arccoshlem(1)
    (0.927037338650686+0j)
    >>> lem * sqrt(2) / 4
    (0.9270373386506859+0j)

    >>> arccoshlem(-1)
    (-0.927037338650686+0j)
    >>> -lem * sqrt(2) / 4
    (-0.9270373386506859+0j)

    Inverse Property:

    >>> z0 = 2.3 - 1.1j

    >>> coshlem(arccoshlem(z0))
    (2.2999999999999985-1.1000000000000003j)
    >>> z0
    (2.3-1.1j)
    
    """
    return 0.5 * ellipticf(2.0 * arccot(w), 0.5)


def modulus_k(
        *,
        k: complex | None = None, 
        m: complex | None = None, 
        tau: complex | None = None, 
        q: complex | None = None, 
        e123: Sequence[complex] | None = None, 
        g23: Sequence[complex] | None = None, 
        w13: Sequence[complex] | None = None
        ) -> complex:

    """
    The elliptic modulus, k.

    Parameters
    ----------
    k :    float or complex, optional.
           The elliptic modulus.
    m :    float or complex, optional.
           The elliptic parameter.
    tau :  float or complex, optional.
           The elliptic half-period ratio, Im(τ) > 0.
    q :    float or complex, optional.
           The elliptic nome, |q| < 1.
    e123 : sequence of 3 float or complex, optional.
           The lattice roots, e₁ + e₂ + e₃ = 0.
    g23 :  sequence of 2 float or complex, optional.
           The elliptic invariants, g₂ and g₃.
    w13 :  sequence of 2 float or complex, optional.
           The elliptic half-periods, ω₁ and ω₃.
    
    Returns
    -------
    out : complex.
        The elliptic modulus, k.

    See Also
    --------
    parameter_m, half_period_ratio_tau, nome_q, weierstrass_e, weierstrass_g, weierstrass_w, weierstrass_eta
            
    Implementation
    --------------
    Connection formulas to m, τ and q are listed in [1].

    From the lattice roots, k is computed using the relation in [2].

    When the invariants are provided, the lattice roots are computed first, and then k is computed from them.

    If the user specifies the half-periods, τ = ω₃ / ω₁, and k is computed from τ.

    References
    ----------
    [1] https://mpmath.org/doc/current/functions/elliptic.html
    
    [2] https://en.wikipedia.org/wiki/Weierstrass_elliptic_function#Relation_to_Jacobi's_elliptic_functions

    Examples
    --------

    >>> k0 = 1.5 - 0.1j

    >>> modulus_k(m=parameter_m(k=k0))
    (1.5-0.10000000000000002j)

    >>> modulus_k(tau=half_period_ratio_tau(k=k0))
    (1.5-0.10000000000000075j)

    >>> modulus_k(q=nome_q(k=k0))
    (1.5000000000000004-0.10000000000000078j)

    """

    if k is not None:
        return k
    if m is not None:
        return sqrt(m)
    
    if tau is not None:
        return modulus_k(q=exp(PI_I * tau))
    if q is not None:
        return exp(2.0 * (log_jacobi_theta(2, 0.0, q) - log_jacobi_theta(3, 0.0, q)))
    
    if e123 is not None:
        e1, e2, e3 = e123
        return sqrt((e2 - e3) / (e1 - e3))
    
    if g23 is not None:
        return modulus_k(e123=weierstrass_e_from_g23(*g23))

    if w13 is not None:
        w1, w3 = w13
        return modulus_k(tau=w3 / w1)

    return NANJ


def parameter_m(
        *,
        k: complex | None = None, 
        m: complex | None = None, 
        tau: complex | None = None, 
        q: complex | None = None, 
        e123: Sequence[complex] | None = None, 
        g23: Sequence[complex] | None = None, 
        w13: Sequence[complex] | None = None
        ) -> complex:

    """
    The elliptic parameter, m.

    Parameters
    ----------
    k :    float or complex, optional.
           The elliptic modulus.
    m :    float or complex, optional.
           The elliptic parameter.
    tau :  float or complex, optional.
           The elliptic half-period ratio, Im(τ) > 0.
    q :    float or complex, optional.
           The elliptic nome, |q| < 1.
    e123 : sequence of 3 float or complex, optional.
           The lattice roots, e₁ + e₂ + e₃ = 0.
    g23 :  sequence of 2 float or complex, optional.
           The elliptic invariants, g₂ and g₃.
    w13 :  sequence of 2 float or complex, optional.
           The elliptic half-periods, ω₁ and ω₃.
    
    Returns
    -------
    out : complex.
        The elliptic parameter, m.

    See Also
    --------
    modulus_k, half_period_ratio_tau, nome_q, weierstrass_e, weierstrass_g, weierstrass_w, weierstrass_eta
            
    Implementation
    --------------
    Connection formulas to k, τ and q are listed in [1].

    From the lattice roots, m is computed using the relation in [2].

    When the invariants are provided, the lattice roots are computed first, and then m is computed from them.

    If the user specifies the half-periods, τ = ω₃ / ω₁, and m is computed from τ.

    References
    ----------
    [1] https://mpmath.org/doc/current/functions/elliptic.html
    
    [2] https://en.wikipedia.org/wiki/Weierstrass_elliptic_function#Relation_to_Jacobi's_elliptic_functions

    Examples
    --------

    >>> m0 = 0.5 + 2.3j

    >>> parameter_m(k=modulus_k(m=m0))
    (0.49999999999999956+2.3j)

    >>> parameter_m(tau=half_period_ratio_tau(m=m0))
    (0.5000000000000001+2.3000000000000007j)
    
    >>> parameter_m(q=nome_q(m=m0))
    (0.5000000000000019+2.300000000000001j)

    """

    if k is not None:
        return k * k
    if m is not None:
        return m
    
    if tau is not None:
        return parameter_m(q=exp(PI_I * tau))
    if q is not None:
        return exp(4.0 * (log_jacobi_theta(2, 0.0, q) - log_jacobi_theta(3, 0.0, q)))
    
    if e123 is not None:
        e1, e2, e3 = e123
        return (e2 - e3) / (e1 - e3)
    
    if g23 is not None:
        return parameter_m(e123=weierstrass_e_from_g23(*g23))

    if w13 is not None:
        w1, w3 = w13
        return parameter_m(tau=w3 / w1)

    return NANJ


def half_period_ratio_tau(
        *,
        k: complex | None = None, 
        m: complex | None = None, 
        tau: complex | None = None, 
        q: complex | None = None, 
        e123: Sequence[complex] | None = None, 
        g23: Sequence[complex] | None = None, 
        w13: Sequence[complex] | None = None
        ) -> complex:

    """
    The elliptic half-period ratio, τ.

    Parameters
    ----------
    k :    float or complex, optional.
           The elliptic modulus.
    m :    float or complex, optional.
           The elliptic parameter.
    tau :  float or complex, optional.
           The elliptic half-period ratio, Im(τ) > 0.
    q :    float or complex, optional.
           The elliptic nome, |q| < 1.
    e123 : sequence of 3 float or complex, optional.
           The lattice roots, e₁ + e₂ + e₃ = 0.
    g23 :  sequence of 2 float or complex, optional.
           The elliptic invariants, g₂ and g₃.
    w13 :  sequence of 2 float or complex, optional.
           The elliptic half-periods, ω₁ and ω₃.
    
    Returns
    -------
    out : complex.
        The elliptic half-period ratio, τ.

    See Also
    --------
    modulus_k, parameter_m, nome_q, weierstrass_e, weierstrass_g, weierstrass_w, weierstrass_eta
            
    Implementation
    --------------
    Connection formulas to k, m and q are listed in [1].

    From the lattice roots, m is computed using the relation in [2], and τ is computed from m.

    When the invariants are provided, the lattice roots are computed first, and then τ is computed from them.

    If the user specifies the half-periods, τ = ω₃ / ω₁.

    References
    ----------
    [1] https://mpmath.org/doc/current/functions/elliptic.html
    
    [2] https://en.wikipedia.org/wiki/Weierstrass_elliptic_function#Relation_to_Jacobi's_elliptic_functions

    Examples
    --------

    >>> tau0 = -0.4 + 1.7j

    >>> half_period_ratio_tau(k=modulus_k(tau=tau0))
    (-0.4000000000000002+1.7j)

    >>> half_period_ratio_tau(m=parameter_m(tau=tau0))
    (-0.4000000000000002+1.7j)

    >>> half_period_ratio_tau(q=nome_q(tau=tau0))
    (-0.4+1.7j)

    """

    if k is not None:
        return half_period_ratio_tau(m=k * k)
    if m is not None:
        return 1.0j * elliptick(1.0 - m) / elliptick(m)
    
    if tau is not None:
        return tau
    if q is not None:
        return -I_PI_INV * log(q)

    if e123 is not None:
        return half_period_ratio_tau(m=parameter_m(e123=e123))

    if g23 is not None:
        return half_period_ratio_tau(m=parameter_m(g23=g23))

    if w13 is not None:
        w1, w3 = w13
        return w3 / w1
    
    return NANJ


def nome_q(
        *,
        k: complex | None = None, 
        m: complex | None = None, 
        tau: complex | None = None, 
        q: complex | None = None, 
        e123: Sequence[complex] | None = None, 
        g23: Sequence[complex] | None = None, 
        w13: Sequence[complex] | None = None
        ) -> complex:

    """
    The elliptic nome, q.

    Parameters
    ----------
    k :    float or complex, optional.
           The elliptic modulus.
    m :    float or complex, optional.
           The elliptic parameter.
    tau :  float or complex, optional.
           The elliptic half-period ratio, Im(τ) > 0.
    q :    float or complex, optional.
           The elliptic nome, |q| < 1.
    e123 : sequence of 3 float or complex, optional.
           The lattice roots, e₁ + e₂ + e₃ = 0.
    g23 :  sequence of 2 float or complex, optional.
           The elliptic invariants, g₂ and g₃.
    w13 :  sequence of 2 float or complex, optional.
           The elliptic half-periods, ω₁ and ω₃.
    
    Returns
    -------
    out : complex.
        The elliptic nome, q.

    See Also
    --------
    modulus_k, parameter_m, half_period_ratio_tau, weierstrass_e, weierstrass_g, weierstrass_w, weierstrass_eta
            
    Implementation
    --------------
    Connection formulas to k, m and τ are listed in [1].

    From the lattice roots, m is computed using the relation in [2], and q is computed from m.

    When the invariants are provided, the lattice roots are computed first, and then q is computed from them.

    If the user specifies the half-periods, τ = ω₃ / ω₁, and q is computed from τ.

    References
    ----------
    [1] https://mpmath.org/doc/current/functions/elliptic.html
    
    [2] https://en.wikipedia.org/wiki/Weierstrass_elliptic_function#Relation_to_Jacobi's_elliptic_functions

    Examples
    --------

    >>> q0 = -0.9

    >>> nome_q(k=modulus_k(q=q0))
    (-0.9000000000000001+1.102182119232618e-16j)

    >>> nome_q(m=parameter_m(q=q0))
    (-0.9000000000000001+1.102182119232618e-16j)

    >>> nome_q(tau=half_period_ratio_tau(q=q0))
    (-0.9+1.1021821192326179e-16j)

    """

    if k is not None:
        return nome_q(m=k * k)
    if m is not None:
        return exp(-PI * elliptick(1.0 - m) / elliptick(m))
    
    if tau is not None:
        return exp(PI_I * tau)
    if q is not None:
        return q

    if e123 is not None:
        return nome_q(m=parameter_m(e123=e123))

    if g23 is not None:
        return nome_q(m=parameter_m(g23=g23))

    if w13 is not None:
        w1, w3 = w13
        return exp(PI_I * w3 / w1)
    
    return NANJ


def weierstrass_e(
        i: WeierstrassEIndex | None = None,
        *,
        k: complex | None = None, 
        m: complex | None = None, 
        tau: complex | None = None, 
        q: complex | None = None, 
        e123: Sequence[complex] | None = None, 
        g23: Sequence[complex] | None = None, 
        w13: Sequence[complex] | None = None
        ) -> complex | tuple[complex, complex, complex]:

    """
    The lattice roots, e₁, e₂ and e₃.

    Parameters
    ----------
    i :    integer, 1, 2, 3, optional.
           The lattice root which should be returned. If i isn't specified, all 3 roots are returned.
    k :    float or complex, optional.
           The elliptic modulus.
    m :    float or complex, optional.
           The elliptic parameter.
    tau :  float or complex, optional.
           The elliptic half-period ratio, Im(τ) > 0.
    q :    float or complex, optional.
           The elliptic nome, |q| < 1.
    e123 : sequence of 3 float or complex, optional.
           The lattice roots, e₁ + e₂ + e₃ = 0.
    g23 :  sequence of 2 float or complex, optional.
           The elliptic invariants, g₂ and g₃.
    w13 :  sequence of 2 float or complex, optional.
           The elliptic half-periods, ω₁ and ω₃.
    
    Returns
    -------
    out : complex or tuple of 3 complex.
        The specified lattice root(s).

    See Also
    --------
    modulus_k, parameter_m, half_period_ratio_tau, nome_q, weierstrass_g, weierstrass_w, weierstrass_eta
            
    Implementation
    --------------
    If we want to compute e₁, e₂ and e₃ from the elliptic invariants g₂ and g₃, we need to find the roots
    of the characteristic polynomial [1], [2]:

    4x³ - g₂ x - g₃ = 0

    4(x - e₁)(x - e₂)(x - e₃) = 0

    The values e₁, e₂ and e₃ are obtained by numerically solving for the roots of the polynomial with careful
    handling of the double or triple root cases.

    The order of e₁, e₂ and e₃ is standardized according to [3]. Mathematica uses a different ordering,
    but we choose the DLMF standard.

    If the elliptic half-periods ω₁ and ω₃ are specified instead, the lattice roots are computed from [4] — [7].

    References
    ----------
    [1] https://dlmf.nist.gov/23.3.E10

    [2] https://dlmf.nist.gov/23.3.E11
    
    [3] https://dlmf.nist.gov/23.22.ii, Starting from Invariants, (a)

    [4] https://dlmf.nist.gov/23.6.E1

    [5] https://dlmf.nist.gov/23.6.E2

    [6] https://dlmf.nist.gov/23.6.E3

    [7] https://dlmf.nist.gov/23.3.E5

    Examples
    --------

    Compute the lattice roots:

    >>> weierstrass_e(g23=(5, 2))
    ((1.280776406404415+0j), (-0.5000000000000003+0j), (-0.7807764064044139+0j))

    >>> weierstrass_e(w13=(3, 0.1j))
    ((82.24670334241125+0j), (82.24670334241125+0j), (-164.4934066848225-0j))

    Double root case:

    >>> weierstrass_e(g23=(3, 1))
    ((1+0j), (-0.5+0j), (-0.5+0j))

    Triple root case:

    >>> weierstrass_e(g23=(0, 0))
    (0j, 0j, 0j)

    """

    if k is not None:
        e = weierstrass_e(tau=half_period_ratio_tau(m=k * k))

    elif m is not None:
        e = weierstrass_e(tau=half_period_ratio_tau(m=m))

    elif tau is not None:
        e = weierstrass_e_from_w13(1.0 + 0.0j, tau)

    elif q is not None:
        e = weierstrass_e(tau=-I_PI_INV * log(q))

    elif e123 is not None:
        e = e123

    elif g23 is not None:
        e = weierstrass_e_from_g23(*g23)

    elif w13 is not None:
        e = weierstrass_e_from_w13(*w13)

    else:
        e = 3 * (NANJ,)

    if i is None:
        return e
    elif i == 1 or i == 2 or i == 3:
        return e[i - 1]
    
    error_message: str = f"Invalid index ({i}). The Weierstrass lattice roots are only defined for i = 1, 2, 3. "
    raise ValueError(error_message)


def weierstrass_g(
        i: WeierstrassGIndex | None = None,
        *,
        k: complex | None = None, 
        m: complex | None = None, 
        tau: complex | None = None, 
        q: complex | None = None, 
        e123: Sequence[complex] | None = None, 
        g23: Sequence[complex] | None = None, 
        w13: Sequence[complex] | None = None
        ) -> complex:

    """
    The elliptic invariants, g₂ and g₃.

    Parameters
    ----------
    i :    integer, 2, 3, optional.
           The invariant which should be returned. If i isn't specified, both invariants are returned.
    k :    float or complex, optional.
           The elliptic modulus.
    m :    float or complex, optional.
           The elliptic parameter.
    tau :  float or complex, optional.
           The elliptic half-period ratio, Im(τ) > 0.
    q :    float or complex, optional.
           The elliptic nome, |q| < 1.
    e123 : sequence of 3 float or complex, optional.
           The lattice roots, e₁ + e₂ + e₃ = 0.
    g23 :  sequence of 2 float or complex, optional.
           The elliptic invariants, g₂ and g₃.
    w13 :  sequence of 2 float or complex, optional.
           The elliptic half-periods, ω₁ and ω₃.
    
    Returns
    -------
    out : complex or tuple of 2 complex.
        The specified invariant(s).

    See Also
    --------
    modulus_k, parameter_m, half_period_ratio_tau, nome_q, weierstrass_e, weierstrass_w, weierstrass_eta
            
    Implementation
    --------------
    The invariants can be trivially computed from the lattice roots e₁, e₂ and e₃ [1], [2].
    
    See the documentation of weierstrass_e for the implementation details.

    References
    ----------
    [1] https://dlmf.nist.gov/23.3.E6

    [2] https://dlmf.nist.gov/23.3.E7

    Examples
    --------

    Compute the invariants:

    From the half-periods:
    >>> weierstrass_g(w13=(1, 1j))
    ((11.81704500807712-3.748624064035754e-36j), (-2.1580803703252577e-15-2.147707250179042e-36j))

    From the lattice roots:
    >>> weierstrass_g(e123=(1, -1/2, -1/2))
    (3.0, 1.0)

    """

    if k is not None:
        g = weierstrass_g(tau=half_period_ratio_tau(m=k * k))

    elif m is not None:
        g = weierstrass_g(tau=half_period_ratio_tau(m=m))

    elif tau is not None:
        g = weierstrass_g(e123=weierstrass_e_from_w13(1.0 + 0.0j, tau))

    elif q is not None:
        g = weierstrass_g(tau=-I_PI_INV * log(q))

    elif e123 is not None:
        e1, e2, e3 = e123
        e12 = e1 * e2
        g = (-4.0 * (e12 + e1 * e3 + e2 * e3), 4.0 * e12 * e3)

    elif g23 is not None:
        g = g23

    elif w13 is not None:
        g = weierstrass_g(e123=weierstrass_e_from_w13(*w13))

    else:
        g = (NANJ, NANJ)

    if i is None:
        return g
    elif i == 2 or i == 3:
        return g[i - 2]
    
    error_message: str = f"Invalid index ({i}). The Weierstrass invariants are only defined for i = 2, 3. "
    raise ValueError(error_message)


def weierstrass_w(
        i: WeierstrassWIndex | None = None,
        *,
        k: complex | None = None, 
        m: complex | None = None, 
        tau: complex | None = None, 
        q: complex | None = None, 
        e123: Sequence[complex] | None = None, 
        g23: Sequence[complex] | None = None, 
        w13: Sequence[complex] | None = None
        ) -> complex:

    """
    The elliptic half-periods, ω₁, ω₂ and ω₃.

    Parameters
    ----------
    i :    integer, 1, 2, 3, optional.
           The half-period which should be returned. If i isn't specified, ω₁ and ω₃ are returned.
    k :    float or complex, optional.
           The elliptic modulus.
    m :    float or complex, optional.
           The elliptic parameter.
    tau :  float or complex, optional.
           The elliptic half-period ratio, Im(τ) > 0.
    q :    float or complex, optional.
           The elliptic nome, |q| < 1.
    e123 : sequence of 3 float or complex, optional.
           The lattice roots, e₁ + e₂ + e₃ = 0.
    g23 :  sequence of 2 float or complex, optional.
           The elliptic invariants, g₂ and g₃.
    w13 :  sequence of 2 float or complex, optional.
           The elliptic half-periods, ω₁ and ω₃.
    
    Returns
    -------
    out : complex or tuple of 2 complex.
        The specified half-period(s).

    See Also
    --------
    modulus_k, parameter_m, half_period_ratio_tau, nome_q, weierstrass_e, weierstrass_g, weierstrass_eta
            
    Implementation
    --------------
    The half-periods ω₁ and ω₃ can be computed from the lattice roots e₁, e₂ and e₃ [1].

    The half-period ω₂ can then be computed easily from ω₁ and ω₃: 
    
    ω₂ = -ω₁ - ω₃
    
    For the implementation details of the lattice roots, see the documentation of weierstrass_e.

    References
    ----------
    [1] http://functions.wolfram.com/09.18.27.0003.01

    [2] http://functions.wolfram.com/09.18.02.0002.01

    Examples
    --------

    Compute the half-periods:

    From the invariants:
    >>> weierstrass_w(g23=(gamma(1/4) ** 8 / (256 * pi ** 2), 0))
    ((0.9999999999999999+0j), 0.9999999999999999j)

    From the lattice roots:
    >>> weierstrass_w(e123=(1, -1/2 + 1j * sqrt(3)/2, -1/2 - 1j * sqrt(3)/2))
    ((1.2143253239437908-1.1102230246251565e-16j), (0.6071626619718953+1.051636578994091j))

    """

    if k is not None:
        w = weierstrass_w(tau=half_period_ratio_tau(m=k * k))

    elif m is not None:
        w = weierstrass_w(tau=half_period_ratio_tau(m=m))

    elif tau is not None:
        w = (1.0 + 0.0j, tau)

    elif q is not None:
        w = weierstrass_w(tau=-I_PI_INV * log(q))

    elif e123 is not None:
        w = weierstrass_w_from_e123(*e123)

    elif g23 is not None:
        w = weierstrass_w_from_e123(*weierstrass_e_from_g23(*g23))

    elif w13 is not None:
        w = w13

    else:
        w = (NANJ, NANJ)

    if i is None:
        return w
    elif i == 1:
        return w[0]
    elif i == 2:
        return -w[0] - w[1]
    elif i == 3:
        return w[1]
    
    error_message: str = f"Invalid index ({i}). The Weierstrass half-periods are only defined for i = 1, 2, 3. "
    raise ValueError(error_message)


def weierstrass_eta(
        i: WeierstrassEtaIndex | None = None,
        *,
        k: complex | None = None, 
        m: complex | None = None, 
        tau: complex | None = None, 
        q: complex | None = None, 
        e123: Sequence[complex] | None = None, 
        g23: Sequence[complex] | None = None, 
        w13: Sequence[complex] | None = None
        ) -> complex:

    """
    The Weierstrass ζ function values at the half-periods, η₁, η₂ and η₃.

    Parameters
    ----------
    i :    integer, 1, 2, 3, optional.
           The value which should be returned. If i isn't specified, all 3 values are returned.
    k :    float or complex, optional.
           The elliptic modulus.
    m :    float or complex, optional.
           The elliptic parameter.
    tau :  float or complex, optional.
           The elliptic half-period ratio, Im(τ) > 0.
    q :    float or complex, optional.
           The elliptic nome, |q| < 1.
    e123 : sequence of 3 float or complex, optional.
           The lattice roots, e₁ + e₂ + e₃ = 0.
    g23 :  sequence of 2 float or complex, optional.
           The elliptic invariants, g₂ and g₃.
    w13 :  sequence of 2 float or complex, optional.
           The elliptic half-periods, ω₁ and ω₃.
    
    Returns
    -------
    out : complex or tuple of 3 complex.
        The specified ζ half-period value(s).

    See Also
    --------
    modulus_k, parameter_m, half_period_ratio_tau, nome_q, weierstrass_e, weierstrass_g, weierstrass_w
            
    Implementation
    --------------
    η₁ and η₃ are computed from the lattice roots e₁, e₂ and e₃ using the relation [1].

    η₂ is then computed using the relation [2]:

    η₁ + η₂ + η₃ = 0

    For the implementation details of the lattice roots, see the documentation of weierstrass_e.

    References
    ----------
    [1] http://functions.wolfram.com/09.21.27.0004.01

    [2] https://dlmf.nist.gov/23.2.E13

    Examples
    --------

    Test the Identities:

    >>> z0 = 0.4 - 0.1j
    >>> g2, g3 = 0, 4
    >>> eta1, eta2, eta3 = weierstrass_eta(g23=(g2, g3))
    >>> w1, w3 = weierstrass_w(g23=(g2, g3))

    >>> weierstrass_zeta(w1, g2, g3)
    (0.37341710011109314-0.6467773898074471j)
    >>> eta1
    (0.3734171001110931-0.6467773898074471j)

    >>> weierstrass_zeta(w3, g2, g3)
    (-0.37341710011109397-0.6467773898074483j)
    >>> eta3
    (-0.373417100111093-0.6467773898074471j)

    >>> weierstrass_sigma(z0 + 2 * w1, g2, g3)
    (-2.006104313442618+2.224882584024451j)
    >>> -exp(2 * eta1 * (z0 + w1)) * weierstrass_sigma(z0, g2, g3)
    (-2.0061043134426204+2.224882584024449j)

    >>> eta1 + eta2 + eta3
    0j
    >>> 0j
    0j

    >>> eta1 * w3 - eta3 * w1
    1.5707963267948952j
    >>> 1j * pi / 2
    1.5707963267948966j

    """

    if k is not None:
        eta = weierstrass_eta(tau=half_period_ratio_tau(m=k * k))

    elif m is not None:
        eta = weierstrass_eta(tau=half_period_ratio_tau(m=m))

    elif tau is not None:
        eta = weierstrass_eta_from_e123(*weierstrass_e_from_w13(1.0 + 0.0j, tau))

    elif q is not None:
        eta = weierstrass_eta(tau=-I_PI_INV * log(q))

    elif e123 is not None:
        eta = weierstrass_eta_from_e123(*e123)

    elif g23 is not None:
        eta = weierstrass_eta_from_e123(*weierstrass_e_from_g23(*g23))

    elif w13 is not None:
        eta = weierstrass_eta_from_e123(*weierstrass_e_from_w13(*w13))

    else:
        eta = 3 * (NANJ,)

    if i is None:
        return eta
    elif i == 1 or i == 2 or i == 3:
        return eta[i - 1]
    
    error_message: str = (f"Invalid index ({i}). The Weierstrass zeta half-period values are only defined "
                          "for i = 1, 2, 3. ")
    raise ValueError(error_message)


if __name__ == "__main__":

    import doctest
    from scipy.special import gamma as zgamma, loggamma as zloggamma

    pi = PI
    lem = 2.6220575542921196  # ϖ — Lemniscate constant.


    def loggamma(z: complex) -> complex:
        return zloggamma(z).astype(complex128).item()


    def gamma(z: complex) -> complex:
        return zgamma(z).astype(complex128).item()


    doctest.testmod(verbose=True)
