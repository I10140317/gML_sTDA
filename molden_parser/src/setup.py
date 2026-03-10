from setuptools import setup, Extension

molden_module = Extension(
    name="molden_parser",
    sources=["molden_parser.c"],
    extra_compile_args=["-O3", "-march=native", "-funroll-loops"],
)

setup(
    name="molden_parser",
    version="1.0",
    description="Fast Molden file parser in C",
    ext_modules=[molden_module],
)

''' The AO int has been replaced by the modified stda code,no longer needed !
molden_module = Extension(
    name="ao_overlap",
    sources=["gaussian_intmat.c"],
    extra_compile_args=["-O3", "-march=native", "-funroll-loops"],
)

setup(
    name="ao_overlap",
    version="1.0",
    description="Fast Molden file parser in C",
    ext_modules=[molden_module],
)
'''
