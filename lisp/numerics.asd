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
     (:file "slicing")
     (:file "aggregation")))
   (:module "arrow"
    :serial t
    :components
    ((:file "schema")
     (:file "array")
     (:file "bridge")
     (:file "io")))))
