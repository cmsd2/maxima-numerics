## Signal Processing

Functions for frequency-domain analysis and filtering. The FFT implementation uses Maxima's fftpack5 library (a Lisp translation of FFTPACK), which supports arbitrary input lengths (most efficient when the length has the form 2^r * 3^s * 5^t).

### Function: np_fft (a)

Compute the discrete Fourier transform of a 1D ndarray.

**Scaling convention (matches NumPy):**

    Y[k] = sum(x[n] * exp(-2 * %pi * %i * k * n / N),  n, 0, N-1)

The forward transform does **not** include a 1/N factor. Use `np_ifft` to invert.

The input `a` must be a 1D ndarray (real or complex). The result is always a complex 1D ndarray of the same length.

#### Examples

```maxima
(%i1) /* FFT of a real signal */
      A : ndarray([1.0, 2.0, 3.0, 4.0]);
(%o1)            ndarray([4], DOUBLE-FLOAT)
(%i2) F : np_fft(A);
(%o2)            ndarray([4], COMPLEX-DOUBLE-FLOAT)
(%i3) np_to_list(F);
(%o3)     [10.0, 2.0 %i - 2.0, - 2.0, - 2.0 %i - 2.0]
(%i4) /* DC component = sum of signal */
      np_ref(F, 0);
(%o4)                         10.0
(%i5) /* Round-trip: ifft(fft(x)) = x */
      np_to_list(np_real(np_ifft(F)));
(%o5)               [1.0, 2.0, 3.0, 4.0]
```

Frequency analysis of a sine wave:

```maxima
(%i1) N : 128$
(%i2) t : np_linspace(0, 1, N)$
(%i3) /* 10 Hz sine wave, sampled at 128 Hz */
      x : np_sin(np_scale(2 * %pi * 10, t))$
(%i4) F : np_fft(x)$
(%i5) /* Magnitude spectrum */
      mag : np_abs(F)$
```

See also: `np_ifft`, `np_abs`, `np_real`, `np_imag`, `np_angle`

### Function: np_ifft (a)

Compute the inverse discrete Fourier transform of a 1D ndarray.

**Scaling convention (matches NumPy):**

    x[n] = (1/N) * sum(Y[k] * exp(+2 * %pi * %i * k * n / N),  k, 0, N-1)

The inverse transform includes the 1/N normalization factor.

The input `a` must be a 1D ndarray (real or complex). The result is always a complex 1D ndarray of the same length. For real-valued signals, use `np_real` to extract the real part of the result.

#### Examples

```maxima
(%i1) /* Frequency-domain filtering */
      x : np_arange(0, 64)$
(%i2) F : np_fft(x)$
(%i3) /* Zero out high-frequency components (low-pass) */
(%i4) y : np_real(np_ifft(F))$
(%i5) /* Verify round-trip */
      closeto(np_ref(y, 0), 0.0);
(%o5)                         true
```

See also: `np_fft`, `np_real`

### Function: np_convolve (a, b)

Compute the 1D discrete linear convolution of two real signals.

Returns a 1D ndarray of length `len(a) + len(b) - 1` ("full" mode, matching NumPy's default). Both inputs must be 1D real ndarrays. Uses direct computation with O(len(a) * len(b)) complexity.

#### Examples

```maxima
(%i1) /* Convolve with impulse: identity operation */
      A : ndarray([1.0, 2.0, 3.0]);
(%o1)            ndarray([3], DOUBLE-FLOAT)
(%i2) np_to_list(np_convolve(A, ndarray([1.0])));
(%o2)                  [1.0, 2.0, 3.0]
(%i3) /* Moving average (box filter) */
      signal : ndarray([1.0, 3.0, 2.0, 5.0, 4.0]);
(%o3)            ndarray([5], DOUBLE-FLOAT)
(%i4) kernel : np_scale(1/3, np_ones(3));
(%o4)            ndarray([3], DOUBLE-FLOAT)
(%i5) np_to_list(np_convolve(signal, kernel));
(%o5) [0.333, 1.333, 2.0, 3.333, 3.666, 3.0, 1.333]
(%i6) /* Output length = len(signal) + len(kernel) - 1 */
      np_shape(np_convolve(signal, kernel));
(%o6)                          [7]
```

See also: `np_fft`, `np_ifft`
