#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Complex Plotting

File Name: complex_plot.py
Description: This module handles the plotting of complex functions.
Author: Rudolf Rosendorf
Email: gdruda975@gmail.com
Published: 05/06/2026 (DD/MM/YYYY)
Last Updated: 05/06/2026 (DD/MM/YYYY)
Version: 1.1.0
License: GNU General Public License v3 (GPLv3)

"""


# ———————————————— IMPORTS ————————————————

from typing import Callable, TypeAlias

from matplotlib.axes import Axes
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

from numpy import (abs as np_abs, angle as arg, complex128, empty, empty_like, exp, float64, isinf, isnan, linspace, 
                   log, log2, logspace, meshgrid, nan, newaxis, ones, ones_like, percentile, stack, where, zeros, 
                   zeros_like)
from numpy.typing import NDArray

from scipy.optimize import brentq
from scipy.interpolate import RegularGridInterpolator

from ..elliptic_functions import PI


# ———————————————— TYPING ————————————————

ComplexFunction: TypeAlias = Callable[[complex,], complex]
ComplexArray: TypeAlias = NDArray[complex128]
RealArray: TypeAlias = NDArray[float64]


def hsl_to_rgb(hsl: RealArray) -> RealArray:

    """ https://en.wikipedia.org/wiki/HSL_and_HSV#HSL_to_RGB """

    h, s, l = hsl[..., 0], hsl[..., 1], hsl[..., 2]

    c = (1.0 - np_abs(2.0 * l - 1.0)) * s
    hp = h / 60.0

    x = c * (1.0 - np_abs(hp % 2.0 - 1.0))
    z = zeros_like(h)

    r, g, b = empty_like(h), empty_like(h), empty_like(h)
    
    m = hp < 1.0
    r[m], g[m], b[m] = c[m], x[m], z[m]
    m = (1.0 <= hp) & (hp < 2.0)
    r[m], g[m], b[m] = x[m], c[m], z[m]
    m = (2.0 <= hp) & (hp < 3.0)
    r[m], g[m], b[m] = z[m], c[m], x[m]
    m = (3.0 <= hp) & (hp < 4.0)
    r[m], g[m], b[m] = z[m], x[m], c[m]
    m = (4.0 <= hp) & (hp < 5.0)
    r[m], g[m], b[m] = x[m], z[m], c[m]
    m = 5.0 <= hp
    r[m], g[m], b[m] = c[m], z[m], x[m]

    rgb = stack((r, g, b), axis=-1) + (l - 0.5 * c)[..., newaxis]
    rgb[rgb < 0.0] = 0.0
    rgb[rgb > 1.0] = 1.0

    return rgb


class ComplexPlot:

    def __init__(self, res: tuple[int, int] = (500, 500), interp_res: tuple[int, int] | None = None, 
                 contours: int = 0, r_contours: int | None = None, arg_contours: int | None = None, 
                 interp_method: str = "linear") -> None:
        
        """
        The complex function plotting utility.

        Parameters
        ----------
        res : sequence of 2 integers.
            The resolution of the plot on the Re and Im axes, defaults to (500, 500).
        interp_res : sequence of 2 integers, optional.
            If specified, the function will be interpolated to this resolution.
        contours : integer.
            The number of magnitude and angle contour lines, defaults to 0.
        r_contours : integer, optional.
            The number of magnitude contour lines, defaults to the value: contours.
        arg_contours : integer, optional.
            The number of angle contour lines, defaults to the value: contours.
        interp_method : string.
            The method for the RegularGridInterpolator engine, defaults to "linear".

        """

        self._m, self._n = res
        self._interpolate = interp_res is not None
        self._interp_method = interp_method

        if self._interpolate:
            self._interpolate = True
            mi, ni = interp_res
            self._resolution = self._mi, self._ni = max(self._m, mi), max(self._n, ni)
        else:
            self._resolution = res

        self._r_contours, self._arg_contours = (contours if r_contours is None else r_contours, 
                                                contours if arg_contours is None else arg_contours)
        
    def _get_function_values(self, func: ComplexFunction, no_bounds: bool = False) -> None:

        if no_bounds:
            x, y = (linspace(self._x_min, self._x_max, num=self._m + 2)[1:-1], 
                    linspace(self._y_min, self._y_max, num=self._n + 2)[1:-1])
        else:
            x, y = linspace(self._x_min, self._x_max, num=self._m), linspace(self._y_min, self._y_max, num=self._n)

        self._x, self._y = meshgrid(x, y)
        z = self._x + 1.0j * self._y

        self._w = empty_like(z)

        for i in range(self._n):
            for j in range(self._m):
                self._w[i, j] = func(z[i, j])

        self._w[isnan(self._w)] = 0.0
        self._w[isinf(self._w)] = 1.7e+308

        if self._interpolate:
            
            interp_re, interp_im = (RegularGridInterpolator((y, x), self._w.real, method=self._interp_method), 
                                    RegularGridInterpolator((y, x), self._w.imag, method=self._interp_method))
    
            if no_bounds:
                x, y = (linspace(self._x_min, self._x_max, num=self._mi + 2)[1:-1], 
                        linspace(self._y_min, self._y_max, num=self._ni + 2)[1:-1])
            else:
                x, y = (linspace(self._x_min, self._x_max, num=self._mi), 
                        linspace(self._y_min, self._y_max, num=self._ni))
            
            self._x, self._y = meshgrid(x, y)

            interp_z = empty((self._mi * self._ni, 2), dtype=float64)
            interp_z[:, 0] = self._y.ravel()
            interp_z[:, 1] = self._x.ravel()

            self._w = (interp_re(interp_z).reshape(self._resolution) + 
                       1.0j * interp_im(interp_z).reshape(self._resolution))

        self._r, self._theta = np_abs(self._w), arg(self._w)
        self._rmin, self._rmax = percentile(self._r, 0.005), percentile(self._r, 99.995)

        r0min, rinfmax = max(self._rmin, self._r[self._r != 0.0].min()), min(self._rmax, self._r[~isinf(self._r)].max())
        self._lrmin, self._lrmax = log2(r0min), log2(rinfmax)

    @staticmethod
    def _hue_function(theta: RealArray) -> RealArray:

        # https://dlmf.nist.gov/help/vrml/aboutcolor

        hue = empty_like(theta)
        q = 2.0 * theta / PI
        q[q < 0.0] = q[q < 0.0] + 4.0

        m = q < 1.0
        hue[m] = q[m]

        m = (1.0 <= q) & (q < 2.0)
        hue[m] = 2.0 * q[m] - 1.0

        m = (2.0 <= q) & (q < 3.0)
        hue[m] = q[m] + 1.0

        m = 3.0 <= q
        hue[m] = 2.0 * (q[m] - 1.0)

        return 60.0 * hue

    def _get_transformation_exponent(self, lt: RealArray) -> None:
        
        def objective(gamma: float) -> float:
            return (lt ** gamma).mean() - 0.5

        mu = lt.mean()
        pg = log(0.5) / log(mu)
        a = b = pg

        if mu < 0.5:
            while objective(a) < 0.0:
                b = a
                a *= 0.1
        else:
            while objective(b) > 0.0:
                a = b
                b *= 10.0
        
        try:
            self._gamma = brentq(objective, a, b, xtol=1.0e-5, rtol=1.0e-5)
        except ValueError:
            self._gamma = 1.0

    def _lightness_function(self, r: RealArray) -> RealArray:

        l = (r - self._rmin) / (self._rmax - self._rmin)
        l[l < 0.0] = 0.0
        l[l > 1.0] = 1.0

        if self._gamma is None:
            self._get_transformation_exponent(l)

        return l ** self._gamma

    def _inverse_lightness_function(self, l: RealArray) -> RealArray:
        return (self._rmax - self._rmin) * l ** (1.0 / self._gamma) + self._rmin

    def _get_plot_image(self) -> RealArray:

        hsl = ones((*self._resolution, 3), dtype=float64)

        hsl[..., 0] = self._hue_function(self._theta)
        hsl[..., 2] = self._lightness_function(self._r)

        return hsl_to_rgb(hsl)

    def _get_argbar_image(self) -> RealArray:

        argbar = empty((self._n, 1, 3), dtype=float64)

        argbar[..., 0] = self._hue_function(theta=linspace(-PI, PI, num=self._n)[:, newaxis])
        argbar[..., 1] = 1.0
        argbar[..., 2] = 0.5

        return hsl_to_rgb(argbar)

    def _get_rbar_image(self) -> RealArray:

        rbar = zeros((self._n, 1, 3), dtype=float64)
        rbar[..., 2] = linspace(0.0, 1.0, self._n)[:, newaxis]

        return hsl_to_rgb(rbar)

    def _plot_image(self, axes: Axes, title: str = "Complex Plot", xlabel: str = "Re(z)", 
                   ylabel: str = "Im(z)", alpha: float = 0.9) -> None:

        axes.imshow(self._get_plot_image(), extent=(self._x_min, self._x_max, self._y_min, self._y_max), origin="lower",
                    aspect="auto", alpha=alpha)
        
        axes.set_title(title)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)

    def _contours(self, axes: Axes, linewidths: float = 0.7, alpha: float = 1.0) -> None:

        if self._r_contours:
            
            # Constant |z| contours.
            r_contour_levels = logspace(self._lrmin, self._lrmax, num=self._r_contours + 2, base=2.0)[1:-1]

            r_contour_l = self._lightness_function(r_contour_levels)
            r_contour_colors = hsl_to_rgb(stack((zeros_like(r_contour_l), zeros_like(r_contour_l), r_contour_l), 
                                                axis=-1))
            
            axes.contour(self._x, self._y, self._r, levels=r_contour_levels, colors=r_contour_colors,
                         linewidths=linewidths, alpha=alpha)

        if self._arg_contours:

            arg_contour_levels = linspace(-PI, PI, num=self._arg_contours, endpoint=False)
            arg_contour_h = self._hue_function(arg_contour_levels)
            arg_contour_colors = hsl_to_rgb(stack((arg_contour_h, ones_like(arg_contour_h), 
                                                0.5 * ones_like(arg_contour_h)), axis=-1))

            # Constant arg(z) contour.
            for level, color in zip(arg_contour_levels, arg_contour_colors):
                rot = exp(-1.0j * level) * self._w
                axes.contour(self._x, self._y, where(rot.real > 0.0, rot.imag, nan), levels=[0.0], colors=color, 
                             linewidths=linewidths, alpha=alpha)
                
    def plot(self, axes: Axes, func: ComplexFunction, z_min: complex, z_max: complex, 
            no_bounds: bool = False, title: str = "Complex Plot", xlabel: str = "Re(z)", 
            ylabel: str = "Im(z)", alpha: float = 0.9, contour_linewidths: float = 0.7,
            contour_alpha: float = 1.0) -> None:
        
        """
        Creates a plot of the complex function.

        Parameters
        ----------
        axes : matplotlib Axes.
            The axes, on which the function should be plotted on.
        func : callable, complex -> complex.
            A complex function which you want to plot.
        z_min : complex.
            The lower-left corner of the plotting range.
        z_max : complex.
            The upper-right corner of the plotting range.
        no_bounds : bool.
            Setting this option to True prevents evaluating func directly on the bounds of the plotting range,
            defaults to False.
        title : string.
            The title of the complex plot, defualts to "Complex Plot".
        xlabel : string.
            The label of the real axis, defaults to "Re(z)".
        ylabel : string.
            The label of the imaginary axis, defaults to "Im(z)". 
        alpha : float, (0, 1).
            The transparency of the plot, 0 is transparent, 1 is opaque. Defaults to 0.9.
        contour_linewidths : float.
            The width of the contour lines, defaults to 0.7.
        contour_alpha : float.
            The transparency of the contour lines, 0 is transparent, 1 is opaque. Defaults to 1.0.
            
        """

        self._gamma = None

        self._x_min, self._y_min = z_min.real, z_min.imag
        self._x_max, self._y_max = z_max.real, z_max.imag

        self._get_function_values(func, no_bounds=no_bounds)
        self._plot_image(axes, title=title, xlabel=xlabel, ylabel=ylabel, alpha=alpha)

        self._contours(axes, linewidths=contour_linewidths, alpha=contour_alpha)
        
        axes.grid(which="major", linestyle="-", color="#3f3f3f", alpha=0.3)
        axes.grid(which="minor", linestyle="-", color="#3f3f3f", alpha=0.1)

        axes.xaxis.set_major_locator(MaxNLocator(5))
        axes.yaxis.set_major_locator(MaxNLocator(5))
        axes.xaxis.set_minor_locator(AutoMinorLocator(5))
        axes.yaxis.set_minor_locator(AutoMinorLocator(5))

    def argbar(self, axes: Axes, aspect_ratio: float = 10.0, label: str = "arg(w)", alpha: float = 0.9) -> None:

        """
        Creates a bar, that relates the complex angle to the color of the plot.

        Parameters
        ----------
        axes : matplotlib Axes.
            The axes, on which the bar should be placed on.
        aspect_ratio : float.
            The aspect ratio of the bar, height / width, defaults to 10.0.
        label : string.
            The label of the bar, defaults to "arg(w)". 
        alpha : float, (0, 1).
            The transparency of the bar, 0 is transparent, 1 is opaque. Defaults to 0.9.

        """

        axes.imshow(self._get_argbar_image(), extent=[-PI, PI, -PI, PI], origin="lower", aspect=aspect_ratio, 
                    alpha=alpha)
        axes.set_xlabel(label)
        axes.set_xticks([])
        axes.set_yticks([-PI, -0.75 * PI, -0.5 * PI, -0.25 * PI, 0.0,
                           0.25 * PI, 0.5 * PI, 0.75 * PI, PI])
        ticklabels = [r"-$\pi$", r"-3$\pi/4$", r"-$\pi/2$", r"-$\pi/4$", "0", 
                      r"$\pi/4$", r"$\pi/2$", r"3$\pi/4$", r"$\pi$"]
        axes.set_yticklabels(ticklabels)
    
    def rbar(self, axes: Axes, aspect_ratio: float = 10.0, label: str = "|w|", alpha: float = 0.9) -> None:

        """
        Creates a bar, that relates the magnitude of the function to the lightness of the plot.

        Parameters
        ----------
        axes : matplotlib Axes.
            The axes, on which the bar should be placed on.
        aspect_ratio : float.
            The aspect ratio of the bar, height / width, defaults to 10.0.
        label : string.
            The label of the bar, defaults to "|w|". 
        alpha : float, (0, 1).
            The transparency of the bar, 0 is transparent, 1 is opaque. Defaults to 0.9.

        """

        n = linspace(0.0, 1.0, num=9)
        il = self._inverse_lightness_function(n)

        axes.imshow(self._get_rbar_image(), extent=[0.0, 1.0, 0.0, 1.0], origin="lower", aspect=aspect_ratio, 
                    alpha=alpha)
        
        axes.set_xlabel(label)
        axes.set_xticks([])
        axes.set_yticks(n)
        axes.set_yticklabels(f"{iln:.2g}" for iln in il)
