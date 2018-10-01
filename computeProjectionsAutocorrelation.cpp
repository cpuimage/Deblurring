#include <cmath>
#include <cassert>
#include <cstring>
#include <algorithm>
#include "image.hpp"

#include "angleSet.hpp"
#include "projectImage.hpp"
#include "conjugate_gradient.hpp"

/// computed the autocorrelation of a signal up to a given window size
/// values at a radius windowRadius of the border of data are not used
template <typename T>
void computeAutocorrelation(std::vector<T>& out, const std::vector<T>& data,
                            int windowRadius)
{
    out.resize(windowRadius*2+1);
    std::fill(out.begin(), out.end(), 0.);

    int len = data.size();
    for (int i = 0; i <= windowRadius; i++) {
        T suma = 0.;
        T sumb = 0.;
        for (int j = 0; j < len - (windowRadius*2 + 1); j++) {
            suma += data[windowRadius + j] * data[windowRadius - i + j];
            sumb += data[windowRadius + j] * data[windowRadius + i + j];
        }

        T res = (suma + sumb) / 2.;
        out[windowRadius-i] = res;
        out[windowRadius+i] = res;
    }

    for (int i = 0; i < windowRadius*2+1; i++) {
        out[i] /= len - (windowRadius*2+1);
    }
}

/// whiten an image by convolving it with a 9 points 1D differentiation filter
/// the filter is applied to rows and columns (returns two images)
template <typename T>
static void whitenImage(img_t<T>& imgBlurX, img_t<T>& imgBlurY,
                        const img_t<T>& imgBlur)
{
    const T filter[] = { 3/840., -32/840., 168/840., -672/840.,
                         0, 672/840., -168/840., 32/840., -3/840. };
    int filterSize = sizeof(filter) / sizeof(*filter);
    int w = imgBlur.w;
    int h = imgBlur.h;

    imgBlurX.ensure_size(w, h);
    imgBlurX.set_value(0);
    imgBlurY.ensure_size(w, h);
    imgBlurY.set_value(0);

    // apply the filter vertically and horizontally
    for (int y = filterSize/2; y < h - filterSize/2; y++) {
        for (int x = filterSize/2; x < w - filterSize/2; x++) {
            for (int i = 0; i < filterSize; i++) {
                imgBlurX(x, y) += filter[filterSize-1 - i] * imgBlur(x + i - filterSize/2, y);
                imgBlurY(x, y) += filter[filterSize-1 - i] * imgBlur(x, y + i - filterSize/2);
            }
        }
    }
}

template <typename T>
void conv(T *y, T *x, int n, void *e)
{
    std::vector<T>& filter = *(std::vector<T>*)e;
    assert((int)filter.size() == n);
    for (int i = 0; i < n; i++) {
        y[i] = 0;
        for (int j = -n/2; j < n/2; j++) {
            int jj = j + n/2;
            // repeating boundaries
            int idx = std::max(0, std::min(n-1, i - j));
            y[i] += filter[jj] * x[idx];
        }
    }
}

template <typename T>
static void deconvolveAutocorrelation(std::vector<T>& acCompensatedRow,
                                      const std::vector<T>& acRow,
                                      const std::vector<T>& compensationFilter)
{
    auto deconvRow = acRow;
    // solve `acRow = convolve(compensationFilter, deconvRow)` to find deconvRow
    conjugate_gradient<T>(&deconvRow[0], conv, &acRow[0], acRow.size(), (void*)&compensationFilter);

    // detect negatives values in the center of the row
    bool hasNegatives = false;
    for (unsigned x = deconvRow.size() / 2 - 2; x <= deconvRow.size() / 2 + 2; x++)
        hasNegatives |= deconvRow[x] < 0.;

    // if there are some negative values, revert the changes
    if (hasNegatives) {
        acCompensatedRow = acRow;
    } else {
        acCompensatedRow = deconvRow;
    }
}

/// computes the autocorrelation of the projections of the whitened image
template <typename T>
void computeProjectionsAutocorrelation(img_t<T>& acProjections, const img_t<T>& imgBlur,
                                       const std::vector<angle_t>& angleSet,
                                       int psSize, T compensationFactor)
{
    img_t<T> projections;

    // whiten the image (horizontal and vertical filtering with the filter 'd')
    img_t<T> imgBlurX;
    img_t<T> imgBlurY;
    whitenImage(imgBlurX, imgBlurY, imgBlur);

    // compute the projections of the derivative
    projectImage(projections, imgBlurX, imgBlurY, angleSet);

    acProjections.ensure_size(psSize*2 + 1, projections.h);

    // build the compensation filter
    // k(x) = 1 / x^compensationFactor
    std::vector<T> compensationFilter(acProjections.w);
    if (compensationFactor > 0.) {
        int center = compensationFilter.size() / 2;
        T sum = 0.;
        for (int i = 0; i < (int) compensationFilter.size(); i++) {
            compensationFilter[i] = 1. / std::pow(std::abs(i - center) + 1, compensationFactor);
            sum += compensationFilter[i];
        }
        for (unsigned int i = 0; i < compensationFilter.size(); i++) {
            compensationFilter[i] /= sum;
        }
    }

#pragma omp parallel
    {
        std::vector<T> proj1d(projections.h);
#pragma omp for
        for (int j = 0; j < projections.h; j++) {
            // extract meaningful values of the projections
            proj1d.resize(0);
            for (int i = 0; i < projections.w; i++) {
                if (!std::isnan(projections(i, j))) {
                    proj1d.push_back(projections(i, j));
                }
            }

            // compute the mean
            T mean = 0.;
            for (T v : proj1d)
                mean += v;
            mean /= proj1d.size();

            // center
            for (T& v : proj1d)
                v -= mean;

            // compute the norm
            T norm = 0.;
            for (T v : proj1d)
                norm += v * v;
            norm = std::sqrt(norm);

            // normalize
            for (T& v : proj1d)
                v /= norm;

            // compute autocorrelation of the projection
            std::vector<T> autocorrelation;
            computeAutocorrelation(autocorrelation, proj1d, psSize);

            // apply the compensation filter
            if (compensationFactor > 0.) {
                // deconvolve the autocorrelation with the compensation filter
                deconvolveAutocorrelation(autocorrelation, autocorrelation, compensationFilter);
            }

            // extract the central part of the autocorrelation
            auto start = &autocorrelation[0] + autocorrelation.size() / 2 - acProjections.w / 2;
            std::copy(start, start + acProjections.w, &acProjections(0, j));
        }
    }
}

