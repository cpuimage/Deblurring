#include <vector>
#include "image.hpp"

#include "computeProjectionsAutocorrelation.hpp"
#include "reconstructPowerspectrum.hpp"
#include "phaseRetrieval.hpp"
#include "angleSet.hpp"
#include "projectImage.hpp"

/// search a patch with high variance in the greyscale blurred image
template <typename T>
static void searchBlurredPatch(img_t<T>& window, const img_t<T>& blurredImage,
                               int windowSize, int searchSamples)
{
    assert(blurredImage.d == 1);
    T best = 0;
    int best_x = 0;
    int best_y = 0;
    for (int i = 0; i < searchSamples; i++) {
        // draw a random position
        int x = (blurredImage.w - windowSize) * (T)rand() / RAND_MAX;
        int y = (blurredImage.h - windowSize) * (T)rand() / RAND_MAX;

        // compute the mean
        T mean = 0.;
        for (int yy = 0; yy < windowSize; yy++) {
            for (int xx = 0; xx < windowSize; xx++) {
                mean += blurredImage(x + xx, y + yy);
            }
        }
        mean /= windowSize * windowSize;

        // compute the variance
        T var = 0.;
        for (int yy = 0; yy < windowSize; yy++) {
            for (int xx = 0; xx < windowSize; xx++) {
                var += std::pow(blurredImage(x + xx, y + yy) - mean, 2);
            }
        }
        var /= windowSize * windowSize;

        // if the variance is greater, we keep the position
        if (var > best) {
            best = var;
            best_x = x;
            best_y = y;
        }
    }

    // copy the chosen window to the output
    window.ensure_size(windowSize, windowSize);
    for (int yy = 0; yy < windowSize; yy++) {
        for (int xx = 0; xx < windowSize; xx++) {
            window(xx, yy) = blurredImage(best_x + xx, best_y + yy);
        }
    }
}

/// apply a circular median filter on each column
template <typename T>
static void circularMedianFilter(img_t<T>& img, int size)
{
    img_t<T> copy(img);
    std::vector<T> pixels(size);
    for (int y = 0; y < img.h; y++) {
        for (int x = 0; x < img.w; x++) {
            // extract 'size' pixels of the column
            // centered around (x,y)
            for (int s = 0; s < size; s++) {
                int yy = (y + s - size/2 + img.h) % img.h;
                pixels[s] = copy(x, yy);
            }

            // sort the value
            std::sort(pixels.begin(), pixels.end());

            // save the median
            if (size % 2) {
                img(x, y) = pixels[size/2];
            } else {
                img(x, y) = (pixels[size/2-1] + pixels[size/2]) / 2.;
            }
        }
    }
}

/// adjust the autocorrelations using the given support (substract mu_theta and clip to 0, normalize, and median filter)
/// solve eq. (39), see Algorithm 4
template <typename T>
static void adjustAutocorrelations(img_t<T>& acProjectionsCorrected,
                                   const img_t<T>& acProjections,
                                   const std::vector<int>& support,
                                   bool medianFilter)
{
    int h = acProjections.h;
    int w = acProjections.w;

    acProjectionsCorrected.ensure_size(w, h);

    // for each orientation, cut and normalize the autocorrelation
    // according to the estimated support
    for (int y = 0; y < h; y++) {
        int cut = w / 2 - support[y];
        T cutVal = acProjections(cut, y);
        for (int x = 0; x < w; x++) {
            if (x <= cut || x >= w-1 - cut) {
                // set to 0 outside the support
                acProjectionsCorrected(x, y) = 0.;
            } else {
                // subtract mu_theta and clip to 0
                acProjectionsCorrected(x, y) = std::max(T(0.), acProjections(x, y) - cutVal);
            }
        }

        // normalize (compensate c_theta)
        T sum = 0.;
        for (int x = 0; x < w; x++) {
            sum += acProjectionsCorrected(x, y);
        }
        for (int x = 0; x < w; x++) {
            acProjectionsCorrected(x, y) /= sum;
        }
    }

    // apply a median filter across angles to smooth the coefficients
    if (medianFilter) {
        circularMedianFilter(acProjectionsCorrected, std::round(2.*std::sqrt(acProjections.h)));
    }
}

/// computed the autocorrelation of a signal up to a given window size
/// uses valid boundary condition
/// (this function differs from the computeAutocorrelation in the file computeProjectionsAutocorrelation.cpp
///  by the fact that this one is adapted to small support of data)
template <typename T>
void computeAutocorrelationSmallSupport(std::vector<T>& out, const std::vector<T>& data,
                                        int windowRadius)
{
    out.resize(windowRadius*2+1);
    std::fill(out.begin(), out.end(), 0.);

    int len = data.size();
    for (int i = -windowRadius; i <= windowRadius; i++) {
        T sum = 0.;

        for (int j = 0; j < len; j++) {
            if (j - i >= 0 && j - i < len) {  // valid boundary condition
                sum += data[j] * data[j - i];
            }
        }

        out[windowRadius + i] = sum;
    }
}

/// Reevaluate the support of the projection of the kernel
template <typename T>
static void reestimateKernelSupport(std::vector<int>& support, const img_t<T>& kernel,
                                    const std::vector<angle_t>& angleSet,
                                    int acRadius, T threshold=5e-2)
{
    std::vector<T> angles(angleSet.size());
    for (unsigned j = 0; j < angleSet.size(); j++) {
        angles[j] = angleSet[j].angle;
    }

    // compute the shear projections of the kernel
    img_t<T> shearProjections;
    projectImage(shearProjections, kernel, angleSet);
    shearProjections.transpose();

    support.resize(angleSet.size());

    img_t<T> ac(acRadius*2+1, angles.size());

    std::vector<T> autocorrelation;
    std::vector<T> proj(shearProjections.h);
    // for each orientation, compute the autocorrelation of the estimated kernel,
    // and estimate its support
    for (unsigned j = 0; j < angles.size(); j++) {
        // extract the projection
        for (int i = 0; i < shearProjections.h; i++) {
            proj[i] = shearProjections(j, i);
            if (std::isnan(proj[i]))
                proj[i] = 0.;
        }

        // compute the autocorrelation of the projection
        computeAutocorrelationSmallSupport(autocorrelation, proj, acRadius);

        // find the max of the autocorrelation (for the adaptative threshold)
        T max = *std::max_element(autocorrelation.begin(), autocorrelation.end());

        // search for the first value exceeding the threshold on the right side of the autocorrelation
        unsigned cursor = autocorrelation.size() - 1;
        while (cursor > autocorrelation.size()/2 && autocorrelation[cursor] < max*threshold)
            cursor--;
        // go back to the value that was below the threshold
        cursor++;
        // and substract half of the autocorrelation size to be within [0, acRadius]
        cursor -= autocorrelation.size() / 2;

        // set the support
        support[j] = cursor;
    }
}


/// estimate the kernel support from the autocorrelation of whitened projections
/// Algorithm 3
template <typename T>
static void initialSupportEstimation(std::vector<int>& support,
                                     const img_t<T> acProjections,
                                     T maxSlope=20./700.)
{
    int h = acProjections.h;
    int w = acProjections.w;

    support.resize(h);

    // local minima of the autocorrelations
    for (int j = 0; j < h; j++) {
        const T* start = &acProjections(w / 2, j);
        support[j] = std::min_element(start, start + w / 2) - start;
    }

    // apply the Lipschitz continuity constraint on the support
    std::vector<T> currentMinimums(h);
    std::fill(currentMinimums.begin(), currentMinimums.end(), w / 2);

    for (int j = 0; j < h; j++) {
        // if the support is lower than the current maximum, we apply the constraint
        if (support[j] < currentMinimums[j]) {
            currentMinimums[j] = support[j];

            // propagate the constraint to other values (to both sides at the same time)
            for (int yy = 1; yy <= h / 2; yy++) {
                // minimum value that is allowed at a distance 'yy' from the current position
                T minimumValue = support[j] + yy * maxSlope;

                int prevRow = (j - yy + h) % h;
                currentMinimums[prevRow] = std::min(currentMinimums[prevRow], minimumValue);

                int nextRow = (j + yy) % h;
                currentMinimums[nextRow] = std::min(currentMinimums[nextRow], minimumValue);
            }
        }
    }

    for (int j = 0; j < h; j++) {
        support[j] = std::ceil(currentMinimums[j]);
    }
}

/// estimate the kernel from a blurred image and a kernel size
/// Algorithm 1 of the paper
template <typename T>
void estimateKernel(img_t<T>& kernel, const img_t<T>& img,
                    int kernelSize, const options& opts)
{
    kernel.ensure_size(kernelSize, kernelSize);

    // convert the image to greyscale
    img_t<T> grey(img.w, img.h);
    grey.greyfromcolor(img);

    // search a blurred patch which will be used for kernel evaluation
    img_t<T> blurredPatch;
    searchBlurredPatch(blurredPatch, grey, 150, 100);

    // compute the angle set
    std::vector<angle_t> angleSet;
    computeProjectionAngleSet(angleSet, kernelSize*2);

    // compute the autocorrelation of the projection of the whitened image
    img_t<T> acProjections;
    computeProjectionsAutocorrelation(acProjections, grey, angleSet, kernelSize*2, opts.compensationFactor);
    int acRadius = acProjections.w / 2;

    // initial support estimation
    std::vector<int> support;
    initialSupportEstimation(support, acProjections);

    // iterative estimation
    img_t<T> powerSpectrum;
    img_t<T> acProjectionsCorrected;
    for (int i = 0; i < opts.Nouter; i++) {
        // adjust the autocorrelation using the estimated support
        adjustAutocorrelations(acProjectionsCorrected, acProjections, support, opts.medianFilter);

        // compute the power spectrum from the autocorrelation
        reconstructPowerspectrum(powerSpectrum, acProjectionsCorrected, angleSet, acRadius);

        // retrieve a kernel in spatial domain using the power spectrum
        phaseRetrieval(kernel, blurredPatch, powerSpectrum, kernelSize, opts);

        // reestimate the kernel support
        reestimateKernelSupport(support, kernel, angleSet, acRadius);
    }
}

