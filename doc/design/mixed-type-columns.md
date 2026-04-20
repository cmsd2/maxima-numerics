# Mixed-Type Columns: String Data, Tables, and DuckDB

## Problem

ndarray is numeric only (`double-float`, `complex-double-float`), backed by
magicl BLAS tensors. The table type (`np_table`) holds named columns, but every
column must be an ndarray. This means:

- No way to represent string, categorical, or date columns
- `np_read_csv` is stubbed out — no file I/O for tabular data
- Plotting categorical data requires manual construction of Maxima string lists

Real-world data is mixed: a CSV might have `name` (string), `date` (date),
`price` (float), `category` (categorical). We need to handle this without
compromising ndarray's numeric performance.

## Design Principles

1. **ndarray stays numeric.** Its strength is BLAS-backed contiguous memory.
   Adding string dtypes would break the magicl tensor abstraction for no benefit.

2. **Tables hold typed columns.** Each column knows its type. Functions that
   work across types (length, slice, display) are generic. Numeric-only
   functions (`np_add`, `np_dot`) keep requiring ndarray.

3. **Arrow is the interchange format.** The same `ArrowSchema`/`ArrowArray`
   structs work for all types. We extend the bridge, not replace it.

4. **DuckDB is the I/O and query engine.** It reads files (CSV, Parquet, JSON),
   executes SQL, and returns typed results. We don't write our own CSV parser.

## Column Types

### Current: ndarray only

```
table
  └── columns: list of ndarray handles
```

### Proposed: typed columns

```
table
  └── columns: list of (ndarray | string-column)
```

A `string-column` is a new Lisp struct:

```lisp
(defstruct string-column
  "A 1D column of strings."
  (data #() :type simple-vector)   ; CL vector of strings
  (length 0 :type fixnum))
```

At the Maxima level it appears as `(($string_column simp) <struct>)`,
following the same opaque handle pattern as ndarray.

### Why not a generic column wrapper?

A generic `column` type that wraps ndarray would add an indirection layer for
all numeric operations — every `np_add`, `np_matmul` etc. would need to unwrap
column → ndarray first. This overhead is unnecessary: ndarray works fine as-is.

Instead, the table is the polymorphic container. Columns are stored as either
ndarray or string-column directly. The table API dispatches on column type
where needed.

### Future column types

If needed later, additional column types (int-column, date-column) can be added
following the same pattern. Each is a separate struct with its own backing
storage. The table dispatches on type. This is simpler than a tagged-union
generic column because each type has different operations and storage needs.

## Table Changes

### Struct

```lisp
(defstruct (table (:constructor %make-table))
  "A columnar table: named typed columns."
  (column-names nil :type list)
  (columns nil :type list)    ; list of ndarray handles or string-column structs
  (nrows 0 :type fixnum))
```

### Column type detection

```lisp
(defun numeric-column-p (col) (typep col 'numerics:ndarray))
(defun string-column-p (col) (typep col 'numerics:string-column))
```

### API changes

| Function | Change |
|---|---|
| `np_table(names, columns)` | Accept mix of ndarrays and string-columns |
| `np_table_column(T, name)` | Return ndarray or string-column handle |
| `np_table_shape(T)` | No change (counts all columns) |
| `np_table_names(T)` | No change |
| `np_table_head(T, n)` | Dispatch slicing per column type |
| `np_table_to_ndarray(T)` | Only include numeric columns (or error if mixed) |

### New functions

| Function | Description |
|---|---|
| `np_string_column(list)` | Create string-column from Maxima list of strings |
| `np_to_string_list(col)` | Convert string-column to Maxima list of strings |
| `string_column_p(x)` | Predicate |

## String-Column Operations

String-columns support a subset of operations that make sense for strings:

| Operation | Description |
|---|---|
| length | Number of elements |
| slice | Sub-column by index range |
| ref | Single element access |
| unique | Unique values |
| sort / argsort | Lexicographic sort |
| display | Show first/last N values |

Numeric operations (`np_add`, `np_mul`, etc.) reject string-columns with a
clear error message.

### Conversion to Maxima

`ax__maybe_list` in ax-plots needs to handle string-columns:

```maxima
ax__maybe_list(x) :=
  if ax__ndarray_p(x) then ax__ndarray_to_list(x)
  else if ax__string_column_p(x) then ax__string_column_to_list(x)
  else x $
```

This makes plotting work transparently:

```maxima
T : np_read_csv("sales.csv")$
ax_draw2d(ax_bar(
  np_table_column(T, "region"),    /* string-column → string list */
  np_table_column(T, "revenue")    /* ndarray → numeric list */
))$
```

## Arrow Bridge Extension

The Arrow C Data Interface uses the same `ArrowSchema` and `ArrowArray` structs
for all types. The `format` field in `ArrowSchema` identifies the type:

| Format code | Type | Buffers |
|---|---|---|
| `g` | float64 | [validity, data] |
| `u` | utf8 string | [validity, offsets (int32), char data] |
| `l` | int64 | [validity, data] |
| `i` | int32 | [validity, data] |
| `tdD` | date32 (days since epoch) | [validity, data] |

### Import: Arrow → column

Extend `arrow-to-ndarray` (or add `arrow-to-column`) to dispatch on format:

```lisp
(defun arrow-to-column (arrow-array-ptr arrow-schema-ptr)
  (let ((fmt (cffi:foreign-slot-value
               arrow-schema-ptr '(:struct arrow-schema) 'format)))
    (cond
      ((string= fmt "g") (arrow-float64-to-ndarray ...))
      ((string= fmt "u") (arrow-utf8-to-string-column ...))
      (t (error "unsupported Arrow format: ~S" fmt)))))
```

For utf8 strings, the ArrowArray has 3 buffers:

```
buffers[0] = validity bitmap (or null)
buffers[1] = offsets: int32 array of length (n+1)
buffers[2] = char data: contiguous utf8 bytes
```

To read string i: `bytes[offsets[i] .. offsets[i+1]]`, decode as utf8.

```lisp
(defun arrow-utf8-to-string-column (array-ptr n)
  (let* ((buffers (foreign-slot-value array-ptr ... 'buffers))
         (offsets-ptr (mem-aref buffers :pointer 1))
         (data-ptr (mem-aref buffers :pointer 2))
         (strings (make-array n)))
    (dotimes (i n)
      (let* ((start (mem-aref offsets-ptr :int32 i))
             (end (mem-aref offsets-ptr :int32 (1+ i)))
             (len (- end start)))
        (setf (aref strings i)
              (foreign-string-to-lisp data-ptr :offset start :count len
                                               :encoding :utf-8))))
    (make-string-column :data strings :length n)))
```

This is always a copy (CL strings are heap objects), which is fine — string
data is typically small relative to numeric columns.

### Export: string-column → Arrow

Build an ArrowArray with offsets + packed utf8 data:

```lisp
(defun string-column-to-arrow (col)
  ;; 1. Compute total byte length and build offsets
  ;; 2. Allocate static-vector for offsets (int32, n+1 elements)
  ;; 3. Allocate static-vector for packed char data
  ;; 4. Copy strings into packed buffer, recording offsets
  ;; 5. Build ArrowSchema with format "u"
  ;; 6. Build ArrowArray with 3 buffers [null, offsets, data]
  ...)
```

### Dictionary encoding (categorical)

Arrow supports dictionary encoding for categorical data. The schema's
`dictionary` field points to another schema describing the dictionary values.
The main array contains integer indices into the dictionary.

This is a future optimization — for now, store categorical data as plain utf8
string-columns. Dictionary encoding can be added later if memory or performance
requires it.

## DuckDB Integration

### Why DuckDB

DuckDB is an in-process analytical query engine (like SQLite for analytics).
It:

- Reads CSV, Parquet, and JSON files directly via SQL
- Auto-detects schemas and types
- Returns typed columnar results
- Embeds as a C library — no server process
- Has an existing Common Lisp wrapper: [cl-duckdb](https://github.com/ak-coram/cl-duckdb)

### cl-duckdb result format

cl-duckdb returns results as association lists of CL specialized vectors — not
Arrow arrays. Each column is a CL vector with type-appropriate elements:

```lisp
;; cl-duckdb result for: SELECT name, price FROM products
;; => (("name" . #("Widget" "Gadget" "Doohickey"))
;;     ("price" . #(9.99d0 24.99d0 4.99d0)))
```

String columns are `simple-vector` of CL strings. Float columns are
`(simple-array double-float (*))`.

### Bridge: cl-duckdb → table

```lisp
(defun duckdb-result-to-table (result)
  "Convert a cl-duckdb result alist to a numerics table."
  (let ((names (mapcar #'car result))
        (columns
          (mapcar (lambda (pair)
                    (let ((data (cdr pair)))
                      (etypecase data
                        ;; Float vector → ndarray
                        ((simple-array double-float (*))
                         (make-ndarray-from-float-vector data))
                        ;; String vector → string-column
                        (simple-vector
                         (make-string-column :data data
                                             :length (length data)))
                        ;; Integer vector → ndarray (coerce to double)
                        ((simple-array fixnum (*))
                         (make-ndarray-from-int-vector data)))))
                  result)))
    (make-table names columns)))
```

### ASDF system structure

Add a third ASDF system for the DuckDB layer:

```
numerics/core   — magicl, trivial-garbage, alexandria
numerics        — core + Arrow (cffi, static-vectors)
numerics/duckdb — numerics + cl-duckdb
```

The DuckDB layer is optional. Users without DuckDB installed use
`numerics/core` or `numerics` and construct tables manually.

### Maxima API

```maxima
/* Read a CSV file into a table */
T : np_read_csv("data.csv")$

/* Read with SQL query */
T : np_query("SELECT name, price FROM read_csv('data.csv') WHERE price > 10")$

/* Query an existing table (registers it as a DuckDB view) */
T2 : np_query("SELECT * FROM t WHERE price > 10", t = T)$

/* Read Parquet */
T : np_read_parquet("data.parquet")$
```

### Native library requirement

cl-duckdb requires the DuckDB C library (`libduckdb`) installed on the system.
This is a new native dependency, but only for the `numerics/duckdb` system.
Users who don't need file I/O or SQL don't need it.

Installation:
- macOS: `brew install duckdb`
- Linux: download from duckdb.org or package manager
- The library is ~30MB

## Implementation Phases

### Phase 1: string-column type

- Add `string-column` struct to `lisp/arrow/` (or new `lisp/core/column.lisp`)
- Maxima handle type `$string_column` with wrap/unwrap
- `np_string_column(list)`, `np_to_string_list(col)`, `string_column_p(x)`
- Display support (console and TeX)
- Extend table to accept string-columns
- Extend `np_table_head` to handle mixed columns
- Update ax-plots `ax__maybe_list` for string-columns
- Tests

### Phase 2: Arrow bridge for strings

- `arrow-utf8-to-string-column` import
- `string-column-to-arrow` export
- Extend `arrow-to-column` dispatcher on format code
- Tests with manually constructed Arrow utf8 arrays

### Phase 3: DuckDB integration

- Add `numerics/duckdb` ASDF system depending on cl-duckdb
- `duckdb-result-to-table` bridge
- Implement `np_read_csv`, `np_read_parquet`, `np_query`
- Optional loading: `load("numerics-duckdb")` or similar
- Tests with sample CSV files

## Data Flow

```
                          ┌─────────────────────────┐
                          │      Maxima user         │
                          │                          │
                          │  np_read_csv("data.csv") │
                          └──────────┬──────────────┘
                                     │
                          ┌──────────▼──────────────┐
                          │   DuckDB (cl-duckdb)     │
                          │   Reads CSV, infers      │
                          │   types, returns CL      │
                          │   vectors                │
                          └──────────┬──────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                 │
           ┌───────▼──────┐  ┌──────▼───────┐  ┌─────▼──────┐
           │ float vector  │  │ string vector│  │ int vector  │
           │ → ndarray     │  │ → string-col │  │ → ndarray   │
           └───────┬──────┘  └──────┬───────┘  └─────┬──────┘
                    │                │                 │
                    └────────────────┼────────────────┘
                                     │
                          ┌──────────▼──────────────┐
                          │   table                  │
                          │   [ndarray, string-col,  │
                          │    ndarray, ...]         │
                          └──────────┬──────────────┘
                                     │
              ┌──────────────────────┼─────────────────────┐
              │                      │                      │
    ┌─────────▼─────────┐  ┌────────▼────────┐   ┌────────▼────────┐
    │  np_table_column   │  │  ax_draw2d /    │   │  Arrow export   │
    │  → ndarray for     │  │  ax_plot2d      │   │  (external      │
    │    numeric ops     │  │  (plots both    │   │   consumers)    │
    │  → string-col for  │  │   types via     │   │                 │
    │    categorical     │  │   ax__maybe_*)  │   │                 │
    └────────────────────┘  └─────────────────┘   └─────────────────┘
```

## Open Questions

1. **Integer columns.** DuckDB returns integers as CL fixnums. Coerce to
   double-float ndarray (lossy for large ints)? Or add an int-column type?
   For now, coerce to double — matches NumPy convention and covers most cases.

2. **Date/time columns.** DuckDB returns `local-time` objects. Store as
   integer epoch values in ndarray (for arithmetic), or as a dedicated
   date-column (for display)? Likely epoch ndarray with a formatting helper.

3. **NULL handling.** DuckDB uses `nil` for SQL NULLs. For numeric columns,
   map to NaN. For string columns, map to `""` or a sentinel? NaN is the
   standard approach for numeric missing data.

4. **`np_query` table binding syntax.** How to pass Maxima tables as DuckDB
   views? cl-duckdb supports registering static tables. The exact syntax
   (`np_query(sql, t = T)`) needs prototyping.
