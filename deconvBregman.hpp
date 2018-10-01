#ifndef DECONV_BREGMAN_H
#define DECONV_BREGMAN_H

#include "image.hpp"

template <typename T>
void pad_and_taper(img_t<T>& u, const img_t<T>& f, const img_t<T>& K);
template <typename T>
void unpad(img_t<T>& u, const img_t<T>& f, const img_t<T>& K);

template <typename T>
void deconvBregman(img_t<T>& u, const img_t<T>& f, const img_t<T>& K,
                   int numIter=30, T lambda=2000., T beta=400.);

#include "deconvBregman.cpp"

#endif

