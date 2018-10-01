#ifndef PHASE_RETRIEVAL_H
#define PHASE_RETRIEVAL_H

#include "image.hpp"
#include "options.hpp"

template <typename T>
void phaseRetrieval(img_t<T>& kernel, const img_t<T>& blurredPatch,
                    const img_t<T>& powerSpectrum, int kernelSize,
                    const options& opts);

#include "phaseRetrieval.cpp"

#endif

