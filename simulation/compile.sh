#!/bin/bash

# This script compiles your Fortran code with Forpy and creates an executable.

# Name of your main Fortran source file without the extension
MAIN_SOURCE="forpy_test"
# Names of your module source files without the extension
MODULE_SOURCES=("forpy_mod")
# Desired name of your executable
EXECUTABLE_NAME="forpy_test"

# Your Fortran compiler
# For example, 'gfortran' for Fortran compiler or 'ifort' for Intel Fortran compiler.
FORTRAN_COMPILER="gfortran"

# Flags for optimization and warning settings (optional but recommended)
# You can customize these flags based on your needs.
FORTRAN_FLAGS=""
# Forpy argument based on your Python version.
FORPy_ARG="`python3-config --ldflags --embed`"

# Check if the Fortran compiler is installed
if ! command -v "$FORTRAN_COMPILER" &> /dev/null
then
    echo "Error: $FORTRAN_COMPILER not found. Please install the Fortran compiler."
    exit 1
fi

# Compile the modules first
for module_source in "${MODULE_SOURCES[@]}"; do
    $FORTRAN_COMPILER $FORTRAN_FLAGS -c "$module_source".F90
    if [ $? -ne 0 ]; then
        echo "Module compilation failed: $module_source.F90"
        exit 1
    fi
done

# Compile the main program and link it with the compiled modules
$FORTRAN_COMPILER $FORTRAN_FLAGS -o "$EXECUTABLE_NAME" "$MAIN_SOURCE".f90 *.o $FORPy_ARG

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Executable '$EXECUTABLE_NAME' created."
else
    echo "Compilation failed."
fi
