// simple C++ image library
//   Anger Jérémy
#pragma once

#include <cstdlib>
#include <string>
#include <complex>
#include <cmath>
#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>
#include <cassert>

#include <fftw3.h>
#include "fftw_allocator.hpp"

template <typename T>
class img_t {
public:
    static void use_threading(int n) {
#ifdef FFTW_HAS_THREADS
        if (n <= 1) return;
        fftw_init_threads();
        fftw_plan_with_nthreads(n);
#endif
    }

    int w, h, d;
    long size;
    std::vector<T, fftw_alloc<T>> data;
    fftw_plan forwardplan;
    fftw_plan backwardplan;
    fftwf_plan forwardplanf;
    fftwf_plan backwardplanf;

    img_t() : w(0), h(0), d(0), size(0), forwardplan(0), backwardplan(0), forwardplanf(0), backwardplanf(0) {}

    img_t(int w, int h, int d=1)
        : w(w), h(h), d(d), size(w*h*d), data(w*d*h), forwardplan(0), backwardplan(0), forwardplanf(0), backwardplanf(0) {
    }
    img_t(int w, int h, int d, T* data)
        : w(w), h(h), d(d), size(w*h*d), forwardplan(0), backwardplan(0), forwardplanf(0), backwardplanf(0) {
        this->data.assign(data, data+w*h*d);
    }

    img_t(const img_t<T>& o)
        : w(o.w), h(o.h), d(o.d), size(w*h*d), data(w*d*h), forwardplan(0), backwardplan(0), forwardplanf(0), backwardplanf(0) {
        copy(o);
    }

    ~img_t() {
        if (forwardplan)
#pragma omp critical (fftw)
            fftw_destroy_plan(forwardplan);
        if (backwardplan)
#pragma omp critical (fftw)
            fftw_destroy_plan(backwardplan);
        if (forwardplanf)
#pragma omp critical (fftw)
            fftwf_destroy_plan(forwardplanf);
        if (backwardplanf)
#pragma omp critical (fftw)
            fftwf_destroy_plan(backwardplanf);
    }

    inline T& operator[](int i) {
        return data[i];
    }
    inline const T& operator[](int i) const {
        return data[i];
    }
    inline T& operator()(int x, int y, int dd=0) {
        return data[dd+d*(x+y*w)];
    }
    inline const T& operator()(int x, int y, int dd=0) const {
        return data[dd+d*(x+y*w)];
    }

    void ensure_size(int w, int h, int d=1) {
        assert(w > 0);
        assert(h > 0);
        assert(d > 0);
        if (this->w != w || this->h != h || this->d != d) {
            this->w = w;
            this->h = h;
            this->d = d;
            size = w * h * d;
            data.resize(size);

            if (forwardplan) {
#pragma omp critical (fftw)
                fftw_destroy_plan(forwardplan);
                forwardplan = nullptr;
            }
            if (backwardplan) {
#pragma omp critical (fftw)
                fftw_destroy_plan(backwardplan);
                backwardplan = nullptr;
            }
        }
    }

    bool inside(int x, int y, int dd=0) const {
        return x >= 0 && x < w && y >= 0 && y < h && dd >= 0 && dd < d;
    }

    void set_value(const T& v) {
        std::fill(data.begin(), data.end(), v);
    }

    template <typename T2>
    T2 sum() const {
        T2 sum(0);
        for (int i = 0; i < size; i++)
            sum += data[i];
        return sum;
    }
    T sum() const {
        T sum(0);
        for (int i = 0; i < size; i++)
            sum += data[i];
        return sum;
    }

    void normalize() {
        T sum = this->sum();
        if (sum != 0.) {
            for (T& v : data) {
                v /= sum;
            }
        }
    }

    void greyfromcolor(const img_t<T>& color) {
        assert(d == 1);
        assert(w == color.w);
        assert(h == color.h);
        for (int i = 0; i < size; i++) {
            T val(0);
            for (int dd = 0; dd < color.d; dd++) {
                val += color[i * color.d + dd];
            }
            (*this)[i] = val / color.d;
        }
    }

    void copy(const img_t<T>& o) {
        assert(o.size == this->size);
        std::copy(o.data.begin(), o.data.end(), data.begin());
    }

    template <typename T2>
    void copy(const img_t<T2>& o) {
        assert(o.size == this->size);
        std::copy(o.data.begin(), o.data.end(), data.begin());
    }

    void fft(const img_t<std::complex<double> >& o) {
        static_assert(std::is_same<T, std::complex<double>>::value, "T must be complex double");
        assert(w == o.w);
        assert(h == o.h);
        assert(d == o.d);
        fftw_complex* out = reinterpret_cast<fftw_complex*>(&data[0]);
        if (!forwardplan) {
            img_t<T> tmp(w, h, d);
            tmp.copy(*this);
            int n[] = {h, w};
#pragma omp critical (fftw)
            forwardplan = fftw_plan_many_dft(2, n, d, out, n, d, 1, out,
                                              n, d, 1, FFTW_FORWARD, FFTW_ESTIMATE);
            copy(tmp);
        }
        this->copy(o);
        fftw_execute(forwardplan);
    }

    void ifft(const img_t<std::complex<double> >& o) {
        static_assert(std::is_same<T, std::complex<double>>::value, "T must be complex double");
        assert(w == o.w);
        assert(h == o.h);
        assert(d == o.d);
        fftw_complex* out = reinterpret_cast<fftw_complex*>(&data[0]);
        if (!backwardplan) {
            img_t<T> tmp(w, h, d);
            tmp.copy(*this);
            int n[] = {h, w};
#pragma omp critical (fftw)
            backwardplan = fftw_plan_many_dft(2, n, d, out, n, d, 1, out,
                                              n, d, 1, FFTW_BACKWARD, FFTW_ESTIMATE);
            copy(tmp);
        }
        double norm = w * h;
        for (int i = 0; i < size; i++)
            (*this)[i] = o[i] / norm;
        fftw_execute(backwardplan);
    }

    void fft(const img_t<std::complex<float> >& o) {
        static_assert(std::is_same<T, std::complex<float>>::value, "T must be complex float");
        assert(w == o.w);
        assert(h == o.h);
        assert(d == o.d);
        fftwf_complex* out = reinterpret_cast<fftwf_complex*>(&data[0]);
        if (!forwardplanf) {
            img_t<T> tmp(w, h, d);
            tmp.copy(*this);
            int n[] = {h, w};
#pragma omp critical (fftw)
            forwardplanf = fftwf_plan_many_dft(2, n, d, out, n, d, 1, out,
                                              n, d, 1, FFTW_FORWARD, FFTW_ESTIMATE);
            copy(tmp);
        }
        this->copy(o);
        fftwf_execute(forwardplanf);
    }

    void ifft(const img_t<std::complex<float> >& o) {
        static_assert(std::is_same<T, std::complex<float>>::value, "T must be complex float");
        assert(w == o.w);
        assert(h == o.h);
        assert(d == o.d);
        fftwf_complex* out = reinterpret_cast<fftwf_complex*>(&data[0]);
        if (!backwardplanf) {
            img_t<T> tmp(w, h, d);
            tmp.copy(*this);
            int n[] = {h, w};
#pragma omp critical (fftw)
            backwardplanf = fftwf_plan_many_dft(2, n, d, out, n, d, 1, out,
                                                n, d, 1, FFTW_BACKWARD, FFTW_ESTIMATE);
            copy(tmp);
        }
        float norm = w * h;
        for (int i = 0; i < size; i++)
            (*this)[i] = o[i] / norm;
        fftwf_execute(backwardplanf);
    }

    void fftshift() {
        img_t<T> copy(*this);

        int halfw = (this->w + 1) / 2.;
        int halfh = (this->h + 1) / 2.;
        int ohalfw = this->w - halfw;
        int ohalfh = this->h - halfh;
        for (int l = 0; l < this->d; l++) {
            for (int y = 0; y < halfh; y++) {
                for (int x = 0; x < ohalfw; x++) {
                    (*this)(x, y + ohalfh, l) = copy(x + halfw, y, l);
                }
            }
            for (int y = 0; y < halfh; y++) {
                for (int x = 0; x < halfw; x++) {
                    (*this)(x + ohalfw, y + ohalfh, l) = copy(x, y, l);
                }
            }
            for (int y = 0; y < ohalfh; y++) {
                for (int x = 0; x < ohalfw; x++) {
                    (*this)(x, y, l) = copy(x + halfw, y + halfh, l);
                }
            }
            for (int y = 0; y < ohalfh; y++) {
                for (int x = 0; x < halfw; x++) {
                    (*this)(x + ohalfw, y, l) = copy(x, y + halfh, l);
                }
            }
        }
    }

    void ifftshift() {
        img_t<T> copy(*this);

        int halfw = (this->w + 1) / 2.;
        int halfh = (this->h + 1) / 2.;
        int ohalfw = this->w - halfw;
        int ohalfh = this->h - halfh;
        for (int l = 0; l < this->d; l++) {
            for (int y = 0; y < ohalfh; y++) {
                for (int x = 0; x < halfw; x++) {
                    (*this)(x, y + halfh, l) = copy(x + ohalfw, y, l);
                }
            }
            for (int y = 0; y < ohalfh; y++) {
                for (int x = 0; x < ohalfw; x++) {
                    (*this)(x + halfw, y + halfh, l) = copy(x, y, l);
                }
            }
            for (int y = 0; y < halfh; y++) {
                for (int x = 0; x < halfw; x++) {
                    (*this)(x, y, l) = copy(x + ohalfw, y + ohalfh, l);
                }
            }
            for (int y = 0; y < halfh; y++) {
                for (int x = 0; x < ohalfw; x++) {
                    (*this)(x + halfw, y, l) = copy(x, y + ohalfh, l);
                }
            }
        }
    }

    template <typename T2>
    void padcirc(const img_t<T2>& o) {
        set_value(0);
        int ww = o.w / 2;
        int hh = o.h / 2;
        for (int dd = 0; dd < d; dd++) {
            int od;
            if (d == o.d)
                od = dd;
            else if (o.d == 1)
                od = 0;
            else
                assert(false);
            for (int y = 0; y < hh; y++) {
                for (int x = 0; x < ww; x++) {
                    (*this)(w  - ww + x, h  - hh + y, dd) = o(x, y, od);
                }
                for (int x = ww; x < o.w; x++) {
                    (*this)(- ww + x, h  - hh + y, dd) = o(x, y, od);
                }
            }
            for (int y = hh; y < o.h; y++) {
                for (int x = 0; x < ww; x++) {
                    (*this)(w  - ww + x, - hh + y, dd) = o(x, y, od);
                }
                for (int x = ww; x < o.w; x++) {
                    (*this)(- ww + x, - hh + y, dd) = o(x, y, od);
                }
            }
        }
    }

    void transpose() {
        img_t<T> o(*this);
        this->w = o.h;
        this->h = o.w;
        for (int y = 0; y < o.h; y++) {
            for (int x = 0; x < o.w; x++) {
                for (int dd = 0; dd < d; dd++) {
                    (*this)(y, x, dd) = o(x, y, dd);
                }
            }
        }
    }

    void transposeToMatlab() {
        img_t<T> o(*this);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                for (int dd = 0; dd < d; dd++) {
                    (*this)[y + h * (x + w * dd)] = o(x, y, dd);
                }
            }
        }
    }

    void transposeFromMatlab() {
        img_t<T> o(*this);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                for (int dd = 0; dd < d; dd++) {
                    (*this)(x, y, dd) = o[y + h * (x + w * dd)];
                }
            }
        }
    }
};
