# Deblurring
Estimating an Image's Blur Kernel Using Natural Image Statistics, and Deblurring it: An Analysis of the Goldstein-Fattal Method

Jérémy Anger, Gabriele Facciolo, Mauricio Delbracio

This program is part of the IPOL publication:
    http://www.ipol.im/pub/pre/211/

Version 20180630

# Compilation:
    run "cmake" to produce an executable named "Deblurring"
    requires a C++11 compatible compiler and the following libraries: libfftw3

# Usage:
    ./Deblurring BLURRY_IMAGE KERNEL_SIZE KERNEL_OUTPUT DEBLURRED_OUTPUT [--alpha COMPENSATION_FACTOR=2.1]

    - BLURRY_IMAGE: should be a tiff, png or jpeg file.
    - KERNEL_SIZE: should be an odd integer large enough to contains the actual estimated kernel
    - KERNEL_OUTPUT: output file for the estimated kernel, should be a .tif in order to keep floating point values
    - DEBLURRED_OUTPUT: output file for the deconvolved image (tif or png), will have the same dynamic range as the input image.
    - COMPENSATION_FACTOR: parameter alpha of the compensation filter, set it to 0 to disable the filtering
    For more options, use "./Deblurring --help"

# Example:
    ./Deblurring hollywood.jpg 15 kernel.tif deblurred.png

# Credits:
    iio.c/h and conjugate_gradient.hpp: from https://github.com/mnhrdt/imscript
    tvdeconv_20120607/: from http://www.ipol.im/pub/art/2012/g-tvdc/
    args.hxx: from https://github.com/Taywee/args

