To build the `Cython` extensions locally, do

```
python setup.py build_ext --inplace
```

Do not commit the compiler-generated files to `git`.

Once the `Cython`-generated `*.c` files have been compiled,
the content of the `*.pyx` files can be used as if they are regular Python modules.

To install the package, run

```
python setup.py install_lib
```

This package uses `Cython`, and its `setup.py` file
contains some examples of compiling `Cython`.
