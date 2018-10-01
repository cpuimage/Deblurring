/**
 * @file tvreg.h
 * @brief TV-regularized image restoration
 * @author Pascal Getreuer <getreuer@gmail.com>
 * 
 * Copyright (c) 2010-2012, Pascal Getreuer
 * All rights reserved.
 * 
 * This program is free software: you can use, modify and/or 
 * redistribute it under the terms of the simplified BSD License. You 
 * should have received a copy of this license along this program. If 
 * not, see <http://www.opensource.org/licenses/bsd-license.html>.
 */
#ifndef _TVREG_H_
#define _TVREG_H_

#include "basic.h"
#include "num.h"

/** @brief Default fidelity weight */
#define TVREGOPT_DEFAULT_LAMBDA         25
/** @brief Default convegence tolerance */
#define TVREGOPT_DEFAULT_TOL            1e-3
/** @brief Default penalty weight on the d = grad u constraint */
#define TVREGOPT_DEFAULT_GAMMA1         5
/** @brief Default penalty weight on the z = u constraint */
#define TVREGOPT_DEFAULT_GAMMA2         8
/** @brief Default maximum number of Bregman iterations */
#define TVREGOPT_DEFAULT_MAXITER        100

/* tvregopt is encapsulated by forward declaration */
typedef struct tag_tvregopt tvregopt;

int TvRestore(num *u, const num *f, int Width, int Height, int NumChannels,
    tvregopt *Opt);

tvregopt *TvRegNewOpt();
void TvRegFreeOpt(tvregopt *Opt);
void TvRegSetLambda(tvregopt *Opt, num Lambda);
void TvRegSetVaryingLambda(tvregopt *Opt,
    const num *VaryingLambda, int LambdaWidth, int LambdaHeight);
void TvRegSetKernel(tvregopt *Opt, 
    const num *Kernel, int KernelWidth, int KernelHeight);
void TvRegSetTol(tvregopt *Opt, num Tol);
void TvRegSetGamma1(tvregopt *Opt, num Gamma1);
void TvRegSetGamma2(tvregopt *Opt, num Gamma2);
void TvRegSetMaxIter(tvregopt *Opt, int MaxIter);
int TvRegSetNoiseModel(tvregopt *Opt, const char *NoiseModel);
void TvRegSetPlotFun(tvregopt *Opt, 
    int (*PlotFun)(int, int, num, const num*, int, int, int, void*),
    void *PlotParam);
void TvRegPrintOpt(const tvregopt *Opt);
const char *TvRegGetAlgorithm(const tvregopt *Opt);

int TvRestoreSimplePlot(int State, int Iter, num Delta,
    ATTRIBUTE_UNUSED const num *u, 
    ATTRIBUTE_UNUSED int Width, 
    ATTRIBUTE_UNUSED int Height, 
    ATTRIBUTE_UNUSED int NumChannels,
    ATTRIBUTE_UNUSED void *Param);

#endif /* _TVREG_H_ */
