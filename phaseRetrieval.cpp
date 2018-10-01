#include <cmath>
#include <complex>
#include <cassert>

#include "image.hpp"
#include "deconvBregman.hpp"

/// Algorithm 6
template <typename T>
static void singlePhaseRetrieval(img_t<T>& kernel, const img_t<T>& magnitude,
                                 int kernelSize, int nbIterations)
{
    using complex = std::complex<T>;
    static const complex I(0, 1);

    kernel.ensure_size(kernelSize, kernelSize);
    kernel.set_value(0);

    const T alpha = 0.95;
    const T beta0 = 0.75;

    img_t<complex> ftkernel(magnitude.w, magnitude.h);
    img_t<T> g(magnitude.w, magnitude.h);
    img_t<complex> gft(g.w, g.h, g.d);
    img_t<T> g2(g);
    img_t<T> R(g);
    img_t<char> omega(g.w, g.h); // can't use bool because of std::vector

    for (int i = 0; i < ftkernel.size; i++) {
        T phase = ((T)rand()/RAND_MAX) * M_PI * 2 - M_PI;
        ftkernel[i] = magnitude[i] * std::exp(I * phase);
    }
    ftkernel.ifft(ftkernel);
    for (int i = 0; i < g.size; i++) {
        g[i] = std::real(ftkernel[i]);
    }

    for (int m = 0; m < nbIterations; m++) {
        T beta = beta0 + (T(1.) - beta0) * (T(1.) - std::exp(- std::pow(m / T(7.), T(3.))));

        for (int i = 0; i < g.size; i++) {
            gft[i] = g[i];
        }
        gft.fft(gft);

        for (int i = 0; i < gft.size; i++) {
            gft[i] = (alpha * magnitude[i] + (T(1.) - alpha) * std::abs(gft[i]))
                     * std::exp(I * std::arg(gft[i]));
        }

        gft.ifft(gft);
        for (int i = 0; i < g.size; i++) {
            g2[i] = std::real(gft[i]);
        }

        for (int i = 0; i < R.size; i++) {
            R[i] = T(2.) * g2[i] - g[i];
        }
        for (int i = 0; i < omega.size; i++) {
            omega[i] = R[i] < T(0.);
        }

        for (int y = 0; y < magnitude.h; y++)
        for (int x = kernelSize; x < magnitude.w; x++) {
            omega(x, y) = true;
        }
        for (int y = kernelSize; y < magnitude.h; y++)
        for (int x = 0; x < magnitude.w; x++) {
            omega(x, y) = true;
        }

        for (int i = 0; i < g.size; i++) {
            g[i] = omega[i] ? beta * g[i] + (T(1.) - T(2.)*beta) * g2[i] : g2[i];
        }
    }

    for (int y = 0; y < kernelSize; y++)
    for (int x = 0; x < kernelSize; x++)
        kernel(x, y) = g2(x, y) >= T(0.) ? g2(x, y) : T(0.);
    kernel.normalize();

    // apply the thresholding of 1/255
    for (int i = 0; i < kernel.size; i++) {
        kernel[i] = kernel[i] < T(1./255.) ? T(0.) : kernel[i];
    }
    kernel.normalize();
}

/// center the kernel at the center of the image
template <typename T>
static void centerKernel(img_t<T>& kernel)
{
    // compute its barycenter
    float dx = 0.f;
    float dy = 0.f;
    for (int y = 0; y < kernel.h; y++) {
        for (int x = 0; x < kernel.w; x++) {
            dx += kernel(x, y) * x;
            dy += kernel(x, y) * y;
        }
    }
    dx = std::round(dx);
    dy = std::round(dy);

    // center the kernel
    img_t<T> copy(kernel);
    for (int y = 0; y < kernel.h; y++) {
        for (int x = 0; x < kernel.w; x++) {
            int nx = (x + (int)dx + (kernel.w/2+1)) % kernel.w;
            int ny = (y + (int)dy + (kernel.h/2+1)) % kernel.h;
            kernel(x, y) = copy(nx, ny);
        }
    }
}

/// evaluate a kernel on a given blurry subimage
template <typename T>
static T evaluateKernel(const img_t<T>& kernel, const img_t<T>& blurredPatch, T deconvLambda)
{
    assert(blurredPatch.d == 1);

    // pad and deconvolve the patch
    img_t<T> paddedBlurredPatch;
    pad_and_taper(paddedBlurredPatch, blurredPatch, kernel);
    img_t<T> deconvPadded;
    deconvBregman(deconvPadded, paddedBlurredPatch, kernel, 10, deconvLambda);
    img_t<T> deconv;
    unpad(deconv, deconvPadded, kernel);

    // compute the l1 and l2 norm of the gradient of the deconvolved patch
    T normL1 = 0.;
    T normL2p2 = 0.;
    for (int y = 1; y < deconv.h; y++) {
        for (int x = 1; x < deconv.w; x++) {
            T dx = deconv(x, y) - deconv(x - 1, y);
            T dy = deconv(x, y) - deconv(x, y - 1);
            T norm = std::sqrt(dx*dx + dy*dy);
            normL1 += norm;
            normL2p2 += norm*norm;
        }
    }

    // returns the score of the kernel
    return normL1 / std::sqrt(normL2p2);
}

/// Algorithm 5
template <typename T>
void phaseRetrieval(img_t<T>& outkernel, const img_t<T>& blurredPatch,
                    const img_t<T>& powerSpectrum, int kernelSize,
                    const options& opts)
{
    img_t<T> magnitude(powerSpectrum.w, powerSpectrum.h);
    for (int i = 0; i < powerSpectrum.size; i++)
        magnitude[i] = std::sqrt(powerSpectrum[i]);
    magnitude.ifftshift(); // unshift the magnitude

    T globalCurrentScore = std::numeric_limits<T>::max();
#pragma omp parallel
    {
        img_t<T> kernel;
        img_t<T> kernel_mirror;
        T currentScore = std::numeric_limits<T>::max();
        img_t<T> bestKernel;
#pragma omp for nowait
        for (int k = 0; k < opts.Ntries; k++) {
            // retrieve one possible kernel
            singlePhaseRetrieval(kernel, magnitude, kernelSize, opts.Ninner);
            centerKernel(kernel);

            // mirror the kernel (because the phase retrieval can't distinguish between the kernel and its mirror)
            kernel_mirror.ensure_size(kernel.w, kernel.h);
            for (int y = 0; y < kernel.h; y++) {
                for (int x = 0; x < kernel.w; x++) {
                    kernel_mirror(x, y) = kernel(kernel.w-1 - x, kernel.h-1 - y);
                }
            }

            // evaluate the two kernels
            img_t<T>* kernels[2] = {&kernel, &kernel_mirror};
            T scores[2];
            for (int i = 0; i < 2; i++) {
                scores[i] = evaluateKernel(*(kernels[i]), blurredPatch, opts.intermediateDeconvolutionWeight);
            }

            // keep the best one
            if (scores[1] < scores[0]) {
                scores[0] = scores[1];
                kernels[0] = kernels[1];
            }

            // if the best of two is better than the current best, keep it
            if (scores[0] < currentScore) {
                currentScore = scores[0];
                bestKernel = *(kernels[0]);
            }
        }

        // aggregate results by keeping the best kernel
#pragma omp critical
        {
            if (currentScore < globalCurrentScore) {
                globalCurrentScore = currentScore;
                outkernel = bestKernel;
            }
        }
    }
}

