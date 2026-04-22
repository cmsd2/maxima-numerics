;;; numerics.asd — ASDF system definition for maxima-numerics

;; Core-only system (no Arrow/DuckDB, fewer native deps)
(defsystem "numerics/core"
  :description "Core ndarray operations only"
  :version "0.1.0"
  :license "MIT"
  :depends-on ("magicl" "trivial-garbage" "alexandria")
  :serial t
  :components
  ((:file "packages")
   (:module "core"
    :serial t
    :components
    ((:file "util")
     (:file "handle")
     (:file "display")
     (:file "convert")
     (:file "constructors")
     (:file "linalg")
     (:file "elementwise")
     (:file "signal")
     (:file "slicing")
     (:file "aggregation")))))

;; Full system (core + Arrow bridge)
(defsystem "numerics"
  :description "NumPy-like numerical computing for Maxima"
  :version "0.1.0"
  :license "MIT"
  :depends-on ("magicl" "cffi" "trivial-garbage" "static-vectors" "alexandria")
  :serial t
  :components
  ((:file "packages")
   (:module "core"
    :serial t
    :components
    ((:file "util")
     (:file "handle")
     (:file "display")
     (:file "convert")
     (:file "constructors")
     (:file "linalg")
     (:file "elementwise")
     (:file "signal")
     (:file "slicing")
     (:file "aggregation")))
   (:module "arrow"
    :serial t
    :components
    ((:file "schema")
     (:file "array")
     (:file "bridge")
     (:file "io")))))

;; Optimization system (core + Maxima's lbfgs)
(defsystem "numerics/optimize"
  :description "L-BFGS optimization for numerics ndarrays"
  :version "0.1.0"
  :license "MIT"
  :depends-on ("numerics/core")
  :serial t
  :components
  ((:module "optimize"
    :serial t
    :components
    ((:file "optimize")))))

;; Image I/O system (core + opticl)
(defsystem "numerics/image"
  :description "Image I/O and opticl interop for numerics"
  :version "0.1.0"
  :license "MIT"
  :depends-on ("numerics/core" "opticl")
  :serial t
  :components
  ((:module "image"
    :serial t
    :components
    ((:file "image")))))
