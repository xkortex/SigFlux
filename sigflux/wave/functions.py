#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy import special


def hermite_function(n):
    """Generate a Hermite Function of order n
    https://en.wikipedia.org/wiki/Hermite_polynomials#Hermite_functions
    :param n: Order
    :return: function(t)
    """
    poly = special.hermite(n)

    def window(x):
        return (2 ** n * np.pi ** 0.5 * special.factorial(n)) ** (
            -0.5) * np.exp(-x * x / 2)

    def func(v):
        return window(v) * poly(v)

    return func


def morlet_function(z, na=1,):
    fc = 2*na / np.pi   # fc is critical frequency
    b = na / np.pi

    def func(t):
        return np.exp(-b * t ** 2) * np.exp(1j * z * np.pi * fc * t)

    return func
