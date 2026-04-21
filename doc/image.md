## Image Processing

Functions for reading, writing, and displaying images as ndarrays. Requires the `numerics-image` module (loaded separately from the core `numerics` package). Images are represented as ndarrays with pixel values in the range 0.0 to 255.0.

**Requirements:** The [opticl](https://github.com/slyrus/opticl) library (installed automatically via Quicklisp when `numerics-image` is loaded).

### Loading the module

```maxima
(%i1) load("numerics")$
(%i2) load("numerics-image")$
```

### Image representation

- **Grayscale images** have shape `[height, width]`
- **Colour images** have shape `[height, width, 3]` (RGB channels)
- Pixel values are `double-float` in the range 0.0 to 255.0
- RGBA images have the alpha channel dropped on load (only RGB is kept)

### Function: np_read_image (path)

Read an image file and return an ndarray.

Supports PNG, JPEG, TIFF, PBM, and other formats supported by opticl. The `path` argument is a string. Returns shape `[h, w]` for grayscale or `[h, w, 3]` for colour images.

#### Examples

```maxima
(%i1) img : np_read_image("/path/to/photo.png");
(%o1)            ndarray([480, 640, 3], DOUBLE-FLOAT)
(%i2) np_shape(img);
(%o2)                      [480, 640, 3]
```

See also: `np_write_image`, `np_mandrill`

### Function: np_write_image (a, path)

Write an ndarray to an image file.

The output format is inferred from the file extension (`.png`, `.jpg`, `.tiff`, etc.). Values are clamped to [0, 255] and rounded to the nearest integer before writing.

#### Examples

```maxima
(%i1) img : np_mandrill()$
(%i2) np_write_image(img, "/tmp/mandrill_copy.png");
(%o2)                         done
```

See also: `np_read_image`, `np_imshow`

### Function: np_imshow (a)

Display an ndarray as an image in the notebook.

Writes a temporary PNG file and prints its path for the notebook renderer to pick up. Supports 2D grayscale `[h, w]` and 3D colour `[h, w, 3]` arrays. Values are clamped to [0, 255].

#### Examples

```maxima
(%i1) img : np_mandrill()$
(%i2) np_imshow(img);
(%o2)                         done
(%i3) /* Display a single colour channel */
      red : np_slice(img, all, all, 0)$
(%i4) np_imshow(red);
(%o4)                         done
```

See also: `np_write_image`, `np_read_image`

### Function: np_mandrill ()

Return the bundled 512x512 mandrill test image as an ndarray.

Returns shape `[512, 512, 3]` (RGB colour). The image is cached after the first call for efficient reuse. This is the standard "mandrill" (baboon) test image commonly used in image processing.

#### Examples

```maxima
(%i1) img : np_mandrill();
(%o1)            ndarray([512, 512, 3], DOUBLE-FLOAT)
(%i2) np_shape(img);
(%o2)                     [512, 512, 3]
(%i3) /* Extract green channel */
      green : np_slice(img, all, all, 1);
(%o3)            ndarray([512, 512], DOUBLE-FLOAT)
```

See also: `np_read_image`, `np_slice`

### Function: np_to_image (a)

Convert an ndarray to a raw opticl image array (unsigned-byte 8).

Returns a wrapped CL array suitable for passing to opticl functions directly. Values are clamped to [0, 255] and rounded. Useful for interoperating with opticl's image manipulation functions from Lisp.

#### Examples

```maxima
(%i1) img : np_mandrill()$
(%i2) raw : np_to_image(img);
(%o2)                    opticl_image(...)
```

See also: `np_from_image`

### Function: np_from_image (img)

Convert an opticl image array back to an ndarray.

Accepts either a wrapped `opticl_image(...)` form (from `np_to_image`) or a raw CL array (from `:lisp` calls). Returns a double-float ndarray with values 0.0 to 255.0.

#### Examples

```maxima
(%i1) img : np_mandrill()$
(%i2) raw : np_to_image(img)$
(%i3) back : np_from_image(raw);
(%o3)            ndarray([512, 512, 3], DOUBLE-FLOAT)
```

See also: `np_to_image`
