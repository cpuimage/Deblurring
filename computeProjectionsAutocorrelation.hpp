#ifndef CALC_AC_PROJECTIONS_H
#define CALC_AC_PROJECTIONS_H

#include "image.hpp"
#include "angleSet.hpp"

template <typename T>
void computeProjectionsAutocorrelation(img_t<T>& acProjections, const img_t<T>& imgBlur,
                                       const std::vector<angle_t>& angleSet,
                                       int psSize, T compensationFactor);

#include "computeProjectionsAutocorrelation.cpp"

#endif

