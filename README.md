
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


# Elliptix

A library of algorithms for evaluating elliptic functions of a complex argument.


## Installation

To install Elliptix, run the following command in your terminal:

```pip install elliptix```


## Documentation

As of right now, the documentation is handled through function Docstrings in the source file. 


## Implemented Functions

**— The Jacobi Theta functions:** 
$\theta_n(z, q)\;;\;n=1, 2, 3, 4$

```python
jacobi_theta(n, z, q)
```

**— The natural logarithm of the Jacobi Theta functions:** 
$\log(\theta_n(z, q))\;;\;n=1, 2, 3, 4$

```python
log_jacobi_theta(n, z, q)
```

**— The Weierstrass elliptic functions:** 
$\wp(z; g_2, g_3), \wp'(z; g_2, g_3), \wp^{-1}(w; g_2, g_3), \sigma(z; g_2, g_3), \zeta(z; g_2, g_3)$

```python
weierstrass_p(z, g2, g3)
weierstrass_p_prime(z, g2, g3)
inverse_weierstrass_p(z, g2, g3)
weierstrass_sigma(z, g2, g3)
weierstrass_zeta(z, g2, g3)
```

**— The conversion functions between the modulus**
$k,$
**the parameter**
$m,$
**the half-period ratio**
$\tau,$
**the nome**
$q,$
**the lattice roots**
$e_1, e_2, e_3,$
**the invariants**
$g_2, g_3,$
**the half-periods**
$\omega_1, \omega_2, \omega_3,$
**and the values of the Weierstrass**
$\zeta$
**function at the half-periods**
$\eta_1, \eta_2, \eta_3$

```python
modulus_k(...)
parameter_m(...)
half_period_ratio_tau(...)
nome_q(...)
weierstrass_e(...)
weierstrass_g(...)
weierstrass_w(...)
weierstrass_eta(...)
```

**— Modular functions:**
$J(\tau)$, $J^{-1}(\tau)$, $\lambda(\tau)$, $\eta(\tau)$, $\eta'(\tau)$, $G_n(q)$, $E_n(q)$, $\phi(q)$

```python
klein_j(tau)
inverse_klein_j(tau)
modular_lambda(tau)
dedekind_eta(tau)
dedekind_eta_prime(tau)
eisenstein_g(n, q)
eisenstein_e(n, q)
euler_phi(q)
```

**— The Jacobi elliptic functions:**
$\text{sn}(u, m), \text{cn}(u, m), \text{dn}(u, m), \text{am}(u, m), ...$ 
**and their inverses:**
$\text{sn}^{-1}(w, m), \text{cn}^{-1}(w, m), ...$

```python
jacobi_ellipfun("sn", u, m)
jacobi_ellipfun("cn", u, m)
jacobi_ellipfun("dn", u, m)
jacobi_ellipfun("am", u, m)
...
inverse_jacobi_ellipfun("sn", u, m)
inverse_jacobi_ellipfun("cn", u, m)
...
```

**— The Neville Theta functions:**
$\theta_c(u, m), \theta_d(u, m), \theta_n(u, m), \theta_s(u, m)$

```python
neville_theta_c(u, m)
neville_theta_d(u, m)
neville_theta_n(u, m)
neville_theta_s(u, m)
```

**— The Lemniscate elliptic functions:**
$\text{sinlem}(z), \text{coslem}(z), \text{sinhlem}(z), \text{coshlem}(z)$
**and their inverses:**
$\text{arcsinlem}(z), \text{arccoslem}(z), \text{arcsinhlem}(z), \text{arccoshlem}(z)$

```python
sinlem(z)
coslem(z)
sinhlem(z)
arcsinlem(z)
arcsinhlem(z)
...
```

## Citation
```bibtex
@misc{elliptix,
  author       = {Rudolf Rosendorf},
  title        = {Elliptix: Algorithms for elliptic functions},
  month        = {June},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/Ruda975/elliptix}}
}
```


## References

- [DLMF] NIST Digital Library of Mathematical Functions. https://dlmf.nist.gov/, Release 1.2.6 of 2026-03-15. 
F. W. J. Olver, A. B. Olde Daalhuis, D. W. Lozier, B. I. Schneider, R. F. Boisvert, C. W. Clark, B. R. Miller, 
B. V. Saunders, H. S. Cohl, and M. A. McClain, eds.

- Weisstein, E. W. (n.d.). MathWorld. Wolfram Research. https://mathworld.wolfram.com/.

- The Mathematical Functions Site. Wolfram Research, Inc., https://functions.wolfram.com/.
