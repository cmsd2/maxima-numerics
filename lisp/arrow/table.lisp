;;; table.lisp — Columnar table type (named 1D ndarray columns)

(in-package #:numerics)

(defstruct (table (:constructor %make-table))
  "A columnar table: named 1D ndarray columns."
  (column-names nil :type list)
  (columns nil :type list)    ; list of ndarray handles
  (nrows 0 :type fixnum))

(defun make-table (names columns)
  "Create a table from column names and ndarray columns."
  (assert (= (length names) (length columns)))
  (let ((nrows (if columns
                   (first (magicl:shape (ndarray-tensor (first columns))))
                   0)))
    (%make-table :column-names names :columns columns :nrows nrows)))

;;; Maxima-level table API

(in-package #:maxima)

(defun $table_p (x)
  "Predicate: is X a table handle?"
  (and (listp x)
       (listp (car x))
       (eq (caar x) '$table)
       (typep (cadr x) 'numerics:table)))

(defun table-unwrap (x)
  "Extract the Lisp table struct from a Maxima table expression."
  (unless ($table_p x)
    (merror "Expected a table, got: ~M" x))
  (cadr x))

(defun numerics-wrap-table (tbl)
  "Wrap a Lisp table struct into a Maxima expression."
  `(($table simp) ,tbl))

(defun $np_table (names columns)
  "Create a table from column names and ndarray columns.
   np_table([\"x\", \"y\"], [x_array, y_array])"
  (unless (and ($listp names) ($listp columns))
    (merror "np_table: expected lists for names and columns"))
  (let ((name-list (mapcar #'$sconcat (cdr names)))
        (col-list (mapcar #'numerics-unwrap (cdr columns))))
    (numerics-wrap-table (numerics:make-table name-list col-list))))

(defun $np_table_column (tbl name)
  "Extract a column by name as an ndarray: np_table_column(T, \"price\")"
  (let* ((t-struct (table-unwrap tbl))
         (name-str ($sconcat name))
         (names (numerics:table-column-names t-struct))
         (cols (numerics:table-columns t-struct))
         (pos (position name-str names :test #'string=)))
    (unless pos
      (merror "np_table_column: column ~S not found. Available: ~{~A~^, ~}"
              name-str names))
    (numerics-wrap (nth pos cols))))

(defun $np_table_to_ndarray (tbl)
  "Stack all numeric columns into a 2D ndarray: np_table_to_ndarray(T)"
  (let* ((t-struct (table-unwrap tbl))
         (cols (numerics:table-columns t-struct))
         (nrows (numerics:table-nrows t-struct))
         (ncols (length cols))
         (result (magicl:empty (list nrows ncols)
                               :type 'double-float :layout :column-major)))
    (loop for col in cols
          for j from 0
          do (let ((tensor (numerics:ndarray-tensor col)))
               (dotimes (i nrows)
                 (setf (magicl:tref result i j) (magicl:tref tensor i)))))
    (numerics-wrap (numerics:make-ndarray result))))

(defun $np_ndarray_to_table (a names)
  "Split a 2D ndarray into named columns: np_ndarray_to_table(A, [\"x\", \"y\"])"
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle))
         (shape (magicl:shape tensor))
         (nrow (first shape))
         (ncol (second shape))
         (name-list (mapcar #'$sconcat (cdr names))))
    (unless (= ncol (length name-list))
      (merror "np_ndarray_to_table: ~D columns but ~D names"
              ncol (length name-list)))
    (let ((columns
            (loop for j below ncol
                  collect (let ((col (magicl:empty (list nrow) :type 'double-float)))
                            (dotimes (i nrow)
                              (setf (magicl:tref col i) (magicl:tref tensor i j)))
                            (numerics:make-ndarray col)))))
      (numerics-wrap-table (numerics:make-table name-list columns)))))

(defun $np_table_shape (tbl)
  "Shape of table: np_table_shape(T) => [nrows, ncols]"
  (let ((t-struct (table-unwrap tbl)))
    `((mlist simp)
      ,(numerics:table-nrows t-struct)
      ,(length (numerics:table-columns t-struct)))))

(defun $np_table_names (tbl)
  "Column names: np_table_names(T) => [\"x\", \"y\"]"
  (let ((t-struct (table-unwrap tbl)))
    `((mlist simp) ,@(numerics:table-column-names t-struct))))

(defun $np_table_head (tbl &optional (n 5))
  "First n rows as a new table: np_table_head(T, n)"
  (let* ((t-struct (table-unwrap tbl))
         (names (numerics:table-column-names t-struct))
         (cols (numerics:table-columns t-struct))
         (actual-n (min n (numerics:table-nrows t-struct)))
         (new-cols
           (loop for col in cols
                 collect (let* ((tensor (numerics:ndarray-tensor col))
                                (result (magicl:empty (list actual-n)
                                                      :type 'double-float)))
                           (dotimes (i actual-n)
                             (setf (magicl:tref result i) (magicl:tref tensor i)))
                           (numerics:make-ndarray result)))))
    (numerics-wrap-table (numerics:make-table names new-cols))))
