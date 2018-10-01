/**
 * @file zsolve_inc.c
 * @brief z-subproblem solvers for restoration with non-Gaussian noise models
 * @author Pascal Getreuer <getreuer@gmail.com>
 * 
 * This file has routines for solving the z subproblem.  These routines are
 * used only when either
 * 
 *   - the noise model is non-Gaussian,
 * 
 *   - the restoration problem has both deconvolution and spatially-varying
 *     lambda (e.g., simultaneous deconvolution-inpainting).
 * 
 * Otherwise, the restoration problem is solved with a simpler d,u splitting.
 * The variable tvregsolver::UseZ indicates whether the restoration includes
 * a z subproblem.
 * 
 * The general form of the z subproblem is
 * \f[ \operatorname*{arg\,min}_{z}\,\lambda\sum_{i,j}F(z_{i,j},f_{i,j})
 * +\frac{\gamma_2}{2}\sum_{i,j}(z_{i,j}-u_{i,j}-b^2_{i,j})^2, \f]
 * where F depends on the noise model and the second term is a penalty to 
 * encourage the constraint z = u.  The optimal z is the solution of
 * \f[ \lambda\partial_z F(z,f) + \gamma_2(z - u - b^2) = 0. \f]
 * 
 * 
 * Copyright (c) 2010-2012, Pascal Getreuer
 * All rights reserved.
 * 
 * This program is free software: you can use, modify and/or 
 * redistribute it under the terms of the simplified BSD License. You 
 * should have received a copy of this license along this program. If 
 * not, see <http://www.opensource.org/licenses/bsd-license.html>.
 */

/** 
 * @brief Solve the z-subproblem with a Laplace (L1) noise model 
 * @param S tvregsolver state
 * 
 * This routine solves the z-subproblem with Laplace noise model to update z,
 * \f[ \operatorname*{arg\,min}_{z}\,\lambda\sum_{i,j}\lvert z_{i,j}-f_{i,j}
 * \rvert+\frac{\gamma_2}{2}\sum_{i,j}(z_{i,j}-u_{i,j}-b^2_{i,j})^2. \f]
 * The auxiliary variable \f$ b^2 \f$ is also updated according to
 * \f[ b^2 = b^2 + u - z. \f]
 * Instead of representing \f$ b^2 \f$ directly, we use 
 * \f$ \tilde{z}=z-b^2 \f$, which is algebraically equivalent but requires
 * less arithmetic.
 */
static void ZSolveL1(tvregsolver *S);
/** 
 * @brief Solve the z-subproblem with a Gaussian (L2) noise model 
 * @param S tvregsolver state
 *
 * Solves the z-subproblem with Gaussian noise model to update z,
 * \f[ \operatorname*{arg\,min}_{z}\,\frac{\lambda}{2}\sum_{i,j}(z_{i,j}-
 * f_{i,j})^2+\frac{\gamma_2}{2}\sum_{i,j}(z_{i,j}-u_{i,j}-b^2_{i,j})^2. \f]
 * The auxiliary variable \f$ b^2 \f$ is also updated according to
 * \f$ b^2 = b^2 + u - z \f$ through ztilde.
 * 
 * \note In most Gaussian noise restoration problems, the simpler d,u 
 * splitting algorithm is used (UseZ = 0).  This routine is only needed in
 * the special case of deconvolution with spatially-varying lambda.
 */
static void ZSolveL2(tvregsolver *S);
/** 
 * @brief Solve the z-subproblem with a Poisson noise model 
 * @param S tvregsolver state
 * 
 * Solves the z-subproblem with Poisson noise model to update z,
 * \f[ \operatorname*{arg\,min}_{z}\,\lambda\sum_{i,j}(z_{i,j} - f_{i,j}\log
 * z_{i,j})+\frac{\gamma_2}{2}\sum_{i,j}(z_{i,j}-u_{i,j}-b^2_{i,j})^2, \f]
 * The auxiliary variable \f$ b^2 \f$ is also updated according to
 * \f$ b^2 = b^2 + u - z \f$ through ztilde.
 */
static void ZSolvePoisson(tvregsolver *S);

#ifndef DOXYGEN
#ifndef _FIDELITY
/* Recursively include file 3 times to define different z solvers */
#define _FIDELITY  1
#include __FILE__           /* Define ZSolveL1 */
#define _FIDELITY  2
#include __FILE__           /* Define ZSolveL2 */
#define _FIDELITY  3
#include __FILE__           /* Define ZSolvePoisson */
#else /* if _FIDELITY is defined */

#include "tvregopt.h"

#if _FIDELITY == 1
static void ZSolveL1(tvregsolver *S)
#elif _FIDELITY == 2
static void ZSolveL2(tvregsolver *S)
#elif _FIDELITY == 3
static void ZSolvePoisson(tvregsolver *S)
#endif
{
    num *z = S->z;
    num *ztilde = S->ztilde;
    const num *Ku = S->Ku;
    const num *f = S->f;
    const num *VaryingLambda = S->Opt.VaryingLambda;
    const num Gamma2 = S->Opt.Gamma2;
    const int Width = S->Width;
    const int Height = S->Height;
    const int NumChannels = S->NumChannels;
    const int PadWidth = S->PadWidth;
    const int PadHeight = S->PadHeight;
    
    const long PadJump = ((long)PadWidth)*(PadHeight - Height);
    num znew;
    long k;
    int x, y;

    if(!VaryingLambda) /* Constant fidelity weight */
    {
        const num Beta = S->Opt.Lambda / Gamma2;
        
        for(k = 0; k < NumChannels; k++, Ku += PadJump)
            for(y = 0; y < Height; y++,
                z += Width, ztilde += Width, f += Width, Ku += PadWidth)
            {
                for(x = 0; x < Width; x++)
                {
#                   if _FIDELITY == 1      /* L1 fidelity      */
                        znew = Ku[x] - f[x] + z[x] - ztilde[x];
                
                        if(znew > Beta)
                            znew += f[x] - Beta;
                        else if(znew < -Beta)
                            znew += f[x] + Beta;
                        else
                            znew = f[x];
#                   elif _FIDELITY == 2    /* L2 fidelity      */
                        znew = (Ku[x] + z[x] - ztilde[x] + Beta*f[x]) 
                            / (1 + Beta);
#                   elif _FIDELITY == 3    /* Poisson fidelity */
                        znew = (Ku[x] + z[x] - ztilde[x] - Beta)/2;
                        znew = znew + (num)sqrt(znew*znew + Beta*f[x]);
#                   endif
                    
                    ztilde[x] += 2*znew - z[x] - Ku[x];
                    z[x] = znew;
                }
            }
    }
    else    /* Spatially varying fidelity weight */
    {
        const num *LambdaPtr;
        num Beta;
        
        for(k = 0; k < NumChannels; k++, Ku += PadJump)
            for(y = 0, LambdaPtr = VaryingLambda; y < Height; y++,
                z += Width, ztilde += Width, f += Width, Ku += PadWidth)
            {
                for(x = 0; x < Width; x++)
                {
                    Beta = LambdaPtr[x] / Gamma2;
                    
#                   if _FIDELITY == 1      /* L1 fidelity      */
                        znew = Ku[x] - f[x] + z[x] - ztilde[x];
                
                        if(znew > Beta)
                            znew += f[x] - Beta;
                        else if(znew < -Beta)
                            znew += f[x] + Beta;
                        else
                            znew = f[x];
#                   elif _FIDELITY == 2    /* L2 fidelity      */
                        znew = (Ku[x] + z[x] - ztilde[x] + Beta*f[x])
                            / (1 + Beta);
#                   elif _FIDELITY == 3    /* Poisson fidelity */
                        znew = (Ku[x] + z[x] - ztilde[x] - Beta)/2;
                        znew = znew + (num)sqrt(znew*znew + Beta*f[x]);
#                   endif
                    
                    ztilde[x] += 2*znew - z[x] - Ku[x];
                    z[x] = znew;
                }
                
                LambdaPtr += Width;
            }
    }
}
#undef _FIDELITY
#endif /* _FIDELITY */
#endif /* DOXYGEN */
