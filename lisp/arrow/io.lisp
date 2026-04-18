;;; io.lisp — I/O functions (stubs, delegated to DuckDB in Phase 3)

(in-package #:maxima)

(defun $np_read_csv (path)
  "Read CSV file into a Table. Requires DuckDB subsystem (Phase 3)."
  (declare (ignore path))
  (merror "np_read_csv: requires DuckDB subsystem. ~
           Load with: :lisp (asdf:load-system \"numerics\")"))

(defun $np_read_parquet (path)
  "Read Parquet file into a Table. Requires DuckDB subsystem (Phase 3)."
  (declare (ignore path))
  (merror "np_read_parquet: requires DuckDB subsystem. ~
           Load with: :lisp (asdf:load-system \"numerics\")"))

(defun $np_write_csv (tbl path)
  "Write Table to CSV. Requires DuckDB subsystem (Phase 3)."
  (declare (ignore tbl path))
  (merror "np_write_csv: requires DuckDB subsystem. ~
           Load with: :lisp (asdf:load-system \"numerics\")"))

(defun $np_write_parquet (tbl path)
  "Write Table to Parquet. Requires DuckDB subsystem (Phase 3)."
  (declare (ignore tbl path))
  (merror "np_write_parquet: requires DuckDB subsystem. ~
           Load with: :lisp (asdf:load-system \"numerics\")"))
