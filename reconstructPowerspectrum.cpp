#include <cstdlib>
#include <cmath>
#include <complex>
#include <cassert>

#include "image.hpp"
#include "angleSet.hpp"

/// reconstruct the power spectrum from a set of autocorrelations of projections
/// each projection is used to reconstruct one or more coefficients
template <typename T>
void reconstructPowerspectrum(img_t<T>& powerSpectrum, const img_t<T> acProjections,
                              const std::vector<angle_t>& angleSet, int psSize)
{
    powerSpectrum.ensure_size(psSize*2+1, psSize*2+1);
    powerSpectrum.set_value(0.);

    img_t<std::complex<T>> ftAutocorrelation(acProjections.w, 1);
    img_t<T> powerSpectrumSlice(acProjections.w, 1);

    for (unsigned j = 0; j < angleSet.size(); j++) {
        // compute the discrete Fourier transform of the autocorrelation
        // (= power spectrum by the Wiener-Khinchin theorem)
        for (int x = 0; x < ftAutocorrelation.w; x++)
            ftAutocorrelation[x] = acProjections(x, j);
        ftAutocorrelation.fft(ftAutocorrelation);

        for (int x = 0; x < acProjections.w; x++) {
            powerSpectrumSlice[x] = std::abs(ftAutocorrelation[x]);
        }

        T normalize = powerSpectrumSlice[0];
        for (int x = 0; x < acProjections.w; x++) {
            powerSpectrumSlice[x] /= normalize;
        }

        powerSpectrumSlice.fftshift();

        // extract and place back the coefficient that intersect the grid
        for (int i = 1; i < psSize + 1; i++) {
            int xOffset = i * angleSet[j].x;
            int yOffset = i * angleSet[j].y;

            if (std::abs(xOffset) > psSize || std::abs(yOffset) > psSize)
                break;

            // place the sample in the 2D power spectrum
            int sliceOffset = std::max(std::abs(xOffset), std::abs(yOffset));
            powerSpectrum(psSize + xOffset, psSize + yOffset) = powerSpectrumSlice[psSize + sliceOffset];
            powerSpectrum(psSize - xOffset, psSize - yOffset) = powerSpectrumSlice[psSize + sliceOffset];
        }
    }

    // the DC value of the kernel is 1
    powerSpectrum(psSize, psSize) = 1.;
}

