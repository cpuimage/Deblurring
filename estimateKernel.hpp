#ifndef RUN_BLIND_KERNEL_ESTIMATION_H
#define RUN_BLIND_KERNEL_ESTIMATION_H

#include "image.hpp"
#include "options.hpp"

template <typename T>
void estimateKernel(img_t<T>& kernel, const img_t<T>& img,
                    int kernelSize, const options& opts);

#include "estimateKernel.cpp"

#endif

