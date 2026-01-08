from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Cela dit à Python comment transformer le code C++ en module Python
extensions = [
    Extension(
        "core",                # Nom du module final
        ["core.pyx"],          # Fichier source
        include_dirs=[numpy.get_include()] # Besoin de Numpy pour les calculs
    )
]

setup(
    name='monotonic_align',
    ext_modules=cythonize(extensions),
)