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

## Linux setup notes

Debian/Ubuntu's `sbcl` package ships a bare binary that doesn't export
`SBCL_HOME`, and Maxima launches sbcl with a custom core which prevents
runtime auto-detection — so `(require :sb-cltl2)` (which magicl pulls in)
can't find the contrib. Export it yourself:

```bash
export SBCL_HOME=/usr/lib/sbcl
```

Additional packages magicl + cffi-grovel pull in via C compilation:

```bash
sudo apt install libffi-dev
```

The first `load("numerics")` and `load(lapack)` compile a lot of Lisp from
source and exceed SBCL's default 1 GB heap. Bump it for the cold-cache
compile (subsequent runs use the cached fasls and fit in default heap):

```bash
maxima -X '--dynamic-space-size 8192' --batch-string 'load("numerics"); load(lapack);'
```

If you ever see `attempt to load an empty FASL file`, an earlier compile
was killed mid-write; clear the per-version cache and rerun with the
bigger heap:

```bash
rm -rf ~/.maxima/binary
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
