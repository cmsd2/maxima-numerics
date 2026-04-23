# Native C Library Dependencies

How Maxima packages can wrap native C libraries and how mxpm can support
discovery and loading.

## Context

The numerics package currently uses two strategies for calling external code:

1. **f2cl (Fortran→CL translation)** — ODEPACK, MINPACK, COBYLA. The original
   Fortran source is compiled to pure Common Lisp. Zero native dependencies,
   fully portable, ships as CL source. Limited to libraries that have been run
   through f2cl.

2. **CFFI via Quicklisp** — magicl calls BLAS/LAPACK at runtime via `dlopen`.
   Fast, but requires the user to have the shared library installed. magicl
   handles discovery internally with hardcoded platform-specific paths.

A third strategy is needed for wrapping C libraries like SUNDIALS that are
neither Fortran (so f2cl doesn't apply) nor already wrapped by a Quicklisp
package (so magicl's approach doesn't help). The goal is a general mechanism
that any Maxima package can use.

## Design

### pkg-config for library discovery

[pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/) is the
standard POSIX mechanism for discovering installed C libraries. When a user
installs a development package (e.g., `brew install sundials` or
`apt install libsundials-dev`), the package manager places `.pc` files that
describe the library's compiler flags, linker flags, and installation paths.

```
$ pkg-config --libs sundials_cvode
-L/opt/homebrew/lib -lsundials_cvode

$ pkg-config --cflags sundials_cvode
-I/opt/homebrew/include
```

Using pkg-config rather than hardcoded paths means:

- Works across package managers (Homebrew, apt, dnf, pacman, Nix, Spack, etc.)
- Works with custom install prefixes (`./configure --prefix=...`)
- Works with environment overrides (`PKG_CONFIG_PATH`)
- The user is responsible for installing the library correctly; we just query
  what's there

### Manifest declaration

Extend `manifest.toml` with an optional `[native]` section that declares
pkg-config dependencies:

```toml
[package]
name = "numerics-sundials"
version = "0.1.0"
entry = "numerics-sundials.mac"

[lisp]
quicklisp_systems = ["cffi"]

[native]
pkg_config = [
    { name = "sundials_ida", min_version = "6.0" },
    { name = "sundials_cvode", min_version = "6.0" },
    { name = "sundials_nvecserial" },
]

[native.install_hint]
macos = "brew install sundials"
debian = "sudo apt install libsundials-dev"
fedora = "sudo dnf install sundials-devel"
```

The `pkg_config` array lists the `.pc` modules the package needs. Each entry
has a `name` (the pkg-config module name) and an optional `min_version`.

The `install_hint` table provides human-readable install commands for common
platforms. These are printed when a dependency is missing — mxpm does not run
them automatically.

### mxpm behaviour

#### `mxpm install`

After cloning the package and installing Quicklisp dependencies, mxpm checks
native dependencies:

```
$ mxpm install numerics-sundials
  Cloning numerics-sundials...
  CL dependencies needed: cffi
  Checking native dependencies...
  ✓ sundials_ida >= 6.0 (found 7.2.0)
  ✓ sundials_cvode >= 6.0 (found 7.2.0)
  ✓ sundials_nvecserial (found 7.2.0)
  Done.
```

If a dependency is missing:

```
$ mxpm install numerics-sundials
  Cloning numerics-sundials...
  CL dependencies needed: cffi
  Checking native dependencies...
  ✗ sundials_ida >= 6.0 (not found)
  ✗ sundials_cvode >= 6.0 (not found)
  ✗ sundials_nvecserial (not found)

  Native libraries missing. Install them with:
    macOS:  brew install sundials
    Debian: sudo apt install libsundials-dev
    Fedora: sudo dnf install sundials-devel

  Then re-run: mxpm install numerics-sundials
```

mxpm does NOT attempt to install native packages automatically. It reports
what's missing and tells the user how to fix it. This avoids needing `sudo`,
avoids cross-platform package manager detection, and keeps mxpm's scope narrow.

The install still succeeds (the package files are placed) — the native check
is a warning, not a blocker. The actual failure happens at load time in Maxima
when CFFI tries to open the shared library.

#### `mxpm doctor` (new command)

A diagnostic command that checks all installed packages for missing native
dependencies:

```
$ mxpm doctor
  numerics: ok (no native dependencies)
  numerics-sundials: missing sundials_ida, sundials_cvode, sundials_nvecserial
    macOS: brew install sundials
```

#### Implementation

mxpm shells out to `pkg-config`:

```rust
fn check_pkg_config(name: &str, min_version: Option<&str>) -> PkgConfigResult {
    let args = match min_version {
        Some(v) => vec![name, "--atleast-version", v],
        None => vec![name, "--exists"],
    };
    let status = Command::new("pkg-config").args(&args).status();
    match status {
        Ok(s) if s.success() => {
            // Get actual version for display
            let version = Command::new("pkg-config")
                .args([name, "--modversion"])
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok());
            PkgConfigResult::Found(version)
        }
        _ => PkgConfigResult::NotFound,
    }
}
```

To extract library paths for use in CFFI definitions, mxpm can also query:

```
pkg-config --variable=libdir sundials_ida
```

This returns the directory containing the shared library (e.g.,
`/opt/homebrew/lib`), which can be passed to the Maxima loader.

### CFFI loading in the Maxima package

The Maxima package's loader uses CFFI to open the shared library at runtime.
Rather than hardcoding platform paths, it queries pkg-config:

```lisp
(defun pkg-config-libdir (module)
  "Query pkg-config for the library directory of MODULE. Returns nil if
   pkg-config is not available or the module is not found."
  (let* ((process (uiop:run-program
                    (list "pkg-config" "--variable=libdir" module)
                    :output :string :ignore-error-status t))
         (output (string-trim '(#\Newline #\Space) process)))
    (when (and (plusp (length output))
               (probe-file output))
      output)))

(defun pkg-config-lib-flags (module)
  "Query pkg-config for the linker library name of MODULE."
  (let* ((output (uiop:run-program
                   (list "pkg-config" "--libs-only-l" module)
                   :output :string :ignore-error-status t))
         (flags (string-trim '(#\Newline #\Space) output)))
    (when (plusp (length flags))
      flags)))
```

The library definition uses this to locate libraries at load time:

```lisp
(defvar *sundials-libdir*
  (or (pkg-config-libdir "sundials_ida")
      ;; Fallback: let dlopen search system paths
      nil))

(cffi:define-foreign-library libsundials-ida
  (:darwin
   (:or
    ;; Use pkg-config result if available
    ,@(when *sundials-libdir*
        (list (merge-pathnames "libsundials_ida.dylib"
                               *sundials-libdir*)))
    ;; Fallback to standard search
    (:default "libsundials_ida")))
  (:unix
   (:or
    ,@(when *sundials-libdir*
        (list (merge-pathnames "libsundials_ida.so"
                               *sundials-libdir*)))
    "libsundials_ida.so.7"
    "libsundials_ida.so"
    (:default "libsundials_ida")))
  (t (:default "libsundials_ida")))
```

If pkg-config is not installed or the module is not found, this falls back to
CFFI's default `dlopen` search, which checks `LD_LIBRARY_PATH` /
`DYLD_LIBRARY_PATH` and standard system paths.

### Load-time error handling

When the shared library cannot be found, the loader catches the CFFI error and
produces a clear message:

```lisp
(handler-case
    (cffi:use-foreign-library libsundials-ida)
  (cffi:load-foreign-library-error (e)
    (merror "numerics-sundials: SUNDIALS IDA library not found.~%~
             Install it with:~%  ~
             macOS:  brew install sundials~%  ~
             Debian: sudo apt install libsundials-dev~%~
             Then restart Maxima.~%~
             (Original error: ~A)" e)))
```

This surfaces as a standard Maxima error, catchable with `errcatch()`.

### User override

If pkg-config doesn't work for a user's setup (e.g., a custom build in a
non-standard location), they can set `PKG_CONFIG_PATH` before starting Maxima:

```
PKG_CONFIG_PATH=/opt/sundials-7.2/lib/pkgconfig maxima
```

Or push a directory onto CFFI's search path in their `maxima-init.mac`:

```maxima
:lisp (push #P"/opt/sundials-7.2/lib/" cffi:*foreign-library-directories*)
```

No changes to the package or mxpm are needed to support either override.

## What this does NOT cover

- **Automatic native package installation** — mxpm prints install hints but
  does not run `brew install` or `apt install`. This would require privilege
  escalation and platform detection that is out of scope.

- **Compiling C code** — some CL packages use CFFI's groveller to compile
  small C programs that extract constants and struct layouts from headers.
  This requires a C compiler and headers at ASDF load time. If we need this,
  the `[native]` section could grow a `build_requires` field. For now, the
  SUNDIALS API is simple enough that we can define constants and struct
  layouts directly in Lisp.

- **Header-only or static linking** — we assume shared libraries. Static
  linking would require compiling C code into a shared library first, which
  is a different workflow.

- **Windows** — pkg-config is not standard on Windows. Windows support is
  out of scope (Maxima/SBCL on Windows is rare in practice).

## Applicability

This design applies to any future Maxima package that needs a native C
library, not just SUNDIALS. Examples:

- A package wrapping SuiteSparse for sparse linear algebra
- A package wrapping HDF5 for scientific data I/O
- A package wrapping FFTW for high-performance FFT (alternative to fftpack5)

The pattern is always the same: declare the pkg-config modules in
`manifest.toml`, query pkg-config at load time, fall back to dlopen search,
and print a helpful error if the library isn't found.
