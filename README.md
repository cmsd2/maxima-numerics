# numerics

[![Docs](https://img.shields.io/badge/docs-online-blue)](https://cmsd2.github.io/maxima-numerics/)

NumPy-like numerical computing for Maxima (SBCL only).

Built on [magicl](https://github.com/quil-lang/magicl) for BLAS/LAPACK-backed tensors.

## Requirements

- SBCL (>= 2.0)
- Quicklisp (for magicl and CL dependencies)
- BLAS/LAPACK (macOS: bundled via Accelerate; Linux: `liblapack-dev libblas-dev`)

## Install

```bash
mxpm install --path . --editable
```

## Usage

```maxima
load("numerics")$
A : ndarray(matrix([1,2],[3,4]));
B : np_inv(A);
np_to_matrix(B);
```

## License

MIT
