// from imscript (https://github.com/mnhrdt/imscript)
// a matrix-free linear solver

#pragma once

#include <cmath>

template <typename T>
using linear_map_t = void(*)(T *y, T *x, int n, void *e);

template <typename T>
static T scalar_product(T *x, T *y, int n)
{
	T r = 0;
	for (int i = 0; i < n; i++)
		r += x[i] * y[i];
	return r;
}

#define FOR(i,n) for(int i = 0; i < n; i++)

template <typename T>
static void fancy_conjugate_gradient(T *x,
		linear_map_t<T> A, const T *b, int n, void *e,
		T *x0, int max_iter, T min_residual)
{
	T *r  = (T*) malloc(n * sizeof(T));
	T *p  = (T*) malloc(n * sizeof(T));
	T *Ap = (T*) malloc(n * sizeof(T));

	A(Ap, x0, n, e);

	FOR(i,n) x[i] = x0[i];
	FOR(i,n) r[i] = b[i] - Ap[i];
	FOR(i,n) p[i] = r[i];

	for (int iter = 0; iter < max_iter; iter++) {
		A(Ap, p, n, e);
		T   App    = scalar_product(Ap, p, n);
		T   rr_old = scalar_product(r, r, n);
		T   alpha  = rr_old / App;
		FOR(i,n) x[i]   = x[i] + alpha * p[i];
		FOR(i,n) r[i]   = r[i] - alpha * Ap[i];
		T   rr_new = scalar_product(r, r, n);
		/*fprintf(stderr, "iter=%d, rr_new=%g\n", iter, rr_new);*/
		if (sqrt(rr_new) < min_residual)
			break;
		T   beta   = rr_new / rr_old;
		FOR(i,n) p[i]   = r[i] + beta * p[i];
	}

	free(r);
	free(p);
	free(Ap);
}

template <typename T>
void conjugate_gradient(T *x, linear_map_t<T> A, const T *b, int n, void *e)
{
	for (int i = 0; i < n; i++)
		x[i] = 0;
	int max_iter = n;
	T min_residual = 1e-10;

	fancy_conjugate_gradient(x, A, b, n, e, x, max_iter, min_residual);
}

