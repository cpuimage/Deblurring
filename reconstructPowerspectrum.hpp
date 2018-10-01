#ifndef CALC_PS_FROM_PROJECTIONS
#define CALC_PS_FROM_PROJECTIONS

#include "image.hpp"
#include "angleSet.hpp"

template <typename T>
void reconstructPowerspectrum(img_t<T>& powerSpectrum, const img_t<T> acProjections,
                              const std::vector<angle_t>& angleSet, int psSize);

#include "reconstructPowerspectrum.cpp"

#endif

