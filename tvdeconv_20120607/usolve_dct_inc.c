/**
 * @file usolve_dct_inc.c
 * @brief u-subproblem DCT solver for TV-regularized deconvolution
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
#include <string.h>
#include "util_deconv.h"


/** 
 * @brief Compute \f$ \mathcal{C}_\mathrm{2e}(\alpha \varphi * A) \f$
 * @param ATrans the destination
 * @param TransformA FFTW plan, transforming A to ATrans
 * @param KernelTrans the transform of the convolution kernel
 * @param Width, Height, NumChannels image dimensions
 * @param Alpha positive scalar
 * 
 * As an intermediate computation for the u subproblem, this routine computes
 * \f$ \mathrm{ATrans}= \mathcal{C}_\mathrm{2e}(\alpha \varphi * A) \f$.
 */
static void AdjBlurDct(num *ATrans, FFT(plan) TransformA, 
    const num *KernelTrans, 
    int Width, int Height, int NumChannels, num Alpha)
{
    const long NumPixels = ((long)Width) * ((long)Height);
    long i;
    int k;
    
    /* Compute ATrans = DCT[A] */
    FFT(execute)(TransformA);
    
    /* Compute ATrans = Alpha . KernelTrans . ATrans */
    for(k = 0; k < NumChannels; k++, ATrans += NumPixels)
        for(i = 0; i < NumPixels; i++)
            ATrans[i] = Alpha * KernelTrans[i] * ATrans[i];
}


/** 
 * @brief Intializations to prepare TvRestore for DCT-based deconvolution 
 * @param S tvreg solver state
 * @return 1 on success, 0 on failure
 *
 * This routine sets up FFTW transform plans and precomputes the
 * transform \f$ \mathcal{C}_\mathrm{1e}(\frac{\lambda}{\gamma}\varphi *
 * \varphi-\Delta) \f$ in S->DenomTrans.  If UseZ = 0, the transform
 * \f$ \mathcal{C}_\mathrm{2e}(\frac{\lambda}{\gamma}\varphi *f) \f$ is 
 * precomputed in S->ATrans.
 */
static int InitDeconvDct(tvregsolver *S)
{    
    num *KernelTrans = S->KernelTrans;
    num *DenomTrans = S->DenomTrans;
    num *B = S->B;
    const num *Kernel = S->Opt.Kernel;
    const int KernelWidth = S->Opt.KernelWidth;
    const int KernelHeight = S->Opt.KernelHeight;
    const int Width = S->Width;
    const int Height = S->Height;
    const num Alpha = S->Alpha;
    const long NumPixels = ((long)Width) * ((long)Height);
    const long PadNumPixels = ((long)Width + 1) * ((long)Height + 1);
    FFT(plan) Plan = NULL;
    FFT(r2r_kind) Kind[2];
    long i;
    int x0, y0, x, y, xi, yi, Size[2];
    int exit;
    
    for(i = 0; i < PadNumPixels; i++)
        B[i] = 0;
    
    x0 = -KernelWidth/2;
    y0 = -KernelHeight/2;
    
    /* Pad Kernel to size Width by Height.  If Kernel
       happens to be larger, it is folded. */
    for(y = 0; y < y0 + KernelHeight; y++)
    {
        yi = WSymExtension(Height + 1, y);
    
        for(x = 0; x < x0 + KernelWidth; x++)
        {
            xi = WSymExtension(Width + 1, x);
            B[xi + (Width + 1)*yi] += Kernel[(x - x0) + KernelWidth*(y - y0)];
        }
    }
    
    /* Compute the DCT-I transform of the padded Kernel */
#ifdef _OPENMP
#pragma omp critical (fftw)
#endif
    exit = !(Plan = FFT(plan_r2r_2d)(Height + 1, Width + 1, B, KernelTrans,
            FFTW_REDFT00, FFTW_REDFT00, FFTW_ESTIMATE | FFTW_DESTROY_INPUT));
    if (exit)
        return 0;
    
    FFT(execute)(Plan);
#ifdef _OPENMP
#pragma omp critical (fftw)
#endif
    FFT(destroy_plan)(Plan);
    
    /* Cut last row and column from KernelTrans */
    for(y = 1, i = Width; y < Height; y++, i += Width)
        memmove(KernelTrans + i, KernelTrans + i + y, sizeof(num)*Width);    
    
    /* Precompute the denominator that will be used in the u-subproblem. */    
    for(y = 0, i = 0; y < Height; y++)
        for(x = 0; x < Width; x++, i++)
            DenomTrans[i] = 
                (num)(4*NumPixels*(Alpha*KernelTrans[i]*KernelTrans[i]
                + 2*(2 - cos(x*M_PI/Width) - cos(y*M_PI/Height))));

    /* Plan DCT-II transforms */
    Size[1] = Width;
    Size[0] = Height;
    Kind[0] = Kind[1] = FFTW_REDFT10;
    
#ifdef _OPENMP
#pragma omp critical (fftw)
#endif
    exit = !(S->TransformA = FFT(plan_many_r2r)(2, Size, S->NumChannels, 
        S->A, NULL, 1, NumPixels, S->ATrans, NULL, 1, NumPixels, Kind,
        FFTW_ESTIMATE | FFTW_DESTROY_INPUT))
        || !(S->TransformB = FFT(plan_many_r2r)(2, Size, S->NumChannels, 
        S->B, NULL, 1, NumPixels, S->BTrans, NULL, 1, NumPixels, Kind,
        FFTW_ESTIMATE | FFTW_DESTROY_INPUT));
    if (exit)
        return 0;
    
    /* Plan inverse DCT-II transforms (DCT-III) */
    Kind[0] = Kind[1] = FFTW_REDFT01;
    
#ifdef _OPENMP
#pragma omp critical (fftw)
#endif
    exit = !(S->InvTransformA = FFT(plan_many_r2r)(2, Size, S->NumChannels, 
        S->ATrans, NULL, 1, NumPixels, S->A, NULL, 1, NumPixels, Kind,
        FFTW_ESTIMATE | FFTW_DESTROY_INPUT))
        || !(S->InvTransformB = FFT(plan_many_r2r)(2, Size, S->NumChannels, 
        S->BTrans, NULL, 1, NumPixels, S->B, NULL, 1, NumPixels, Kind,
        FFTW_ESTIMATE | FFTW_DESTROY_INPUT));
    if (exit)
        return 0;
    
    /* Compute ATrans = Alpha . KernelTrans . DCT[f] */
    if(!S->UseZ)
    {
        memcpy(S->A, S->f, sizeof(num)*NumPixels*S->NumChannels);
        AdjBlurDct(S->ATrans, S->TransformA, 
            KernelTrans, Width, Height, S->NumChannels, Alpha);
    }
    
    S->Ku = S->A;
    return 1;
}


/** 
 * @brief Compute BTrans = ( ATrans - DCT[div(dtilde)] ) / DenomTrans
 * 
 * This subroutine is a part of the DCT u-subproblem solution that is common 
 * to both the d,u splitting (UseZ = 0) and d,u,z splitting (UseZ = 1).
 */
static void UTransSolveDct(num *BTrans, num *B, FFT(plan) TransformB,
    num *ATrans, const numvec2 *dtilde, 
    const num *DenomTrans, int Width, int Height, int NumChannels)
{
    const long NumPixels = ((long)Width) * ((long)Height);
    long i;
    int k;
    
    /* Compute B = div(dtilde) */
    Divergence(B, Width, Height, dtilde, Width, Height, NumChannels);
    
    /* Compute BTrans = DCT[B] */
    FFT(execute)(TransformB);

    /* Compute BTrans = ( ATrans - BTrans ) / DenomTrans */
    for(k = 0; k < NumChannels; k++, ATrans += NumPixels, BTrans += NumPixels)
        for(i = 0; i < NumPixels; i++)
            BTrans[i] = (ATrans[i] - BTrans[i]) / DenomTrans[i];
}


/** 
 * @brief Solve the u subproblem using DCT transforms (UseZ = 0)
 * @param S tvreg solver state
 *
 * This routine solves the u-subproblem 
 * \f[ \tfrac{\lambda}{\gamma}\varphi *\varphi *u -\Delta u = \tfrac{\lambda}{
 * \gamma}\varphi *f -\operatorname{div}\tilde{d}. \f]
 * The solution is obtained using discrete cosine transforms (DCTs) as
 * \f[ u=\mathcal{C}_\mathrm{2e}^{-1}\left[\frac{\mathcal{C}_\mathrm{2e}
 * \bigl(\frac{\lambda}{\gamma}\varphi *f-\operatorname{div}\tilde{d}\bigr)}{
 * \mathcal{C}_\mathrm{1e}(\frac{\lambda}{\gamma}\varphi *\varphi-\Delta)}
 * \right], \f]
 * where \f$ \mathcal{C}_\mathrm{1e} \f$ and \f$ \mathcal{C}_\mathrm{2e} \f$
 * denote the DCT-I and DCT-II transforms of the same period lengths.  Two of 
 * the above quantities are precomputed by InitDeconvDct(): the transform of 
 * \f$ \frac{\lambda}{\gamma}\varphi *f \f$ is stored in S->ATrans and the 
 * transformed denominator is stored in S->DenomTrans.
 */
static num UDeconvDct(tvregsolver *S)
{
    /* BTrans = ( ATrans - DCT[div(dtilde)] ) / DenomTrans */
    UTransSolveDct(S->BTrans, S->B, S->TransformB, S->ATrans, S->dtilde,
        S->DenomTrans, S->Width, S->Height, S->NumChannels);
    /* B = IDCT[BTrans] */
    FFT(execute)(S->InvTransformB);
    /* Compute ||B - u||, and assign u = B */
    return UUpdate(S);
}


#if defined(TVREG_USEZ) || defined(DOXYGEN)
/** 
 * @brief Solve the u subproblem using DCT transforms (UseZ = 1)
 * @param S tvreg solver state
 *
 * This extended version of UDeconvDct() is used when performing DCT-based
 * deconvolution with the three-auxiliary variable algorithm (UseZ = 1).
 * The u subproblem in this case is
 * \f[ \tfrac{\gamma_2}{\gamma_1}\varphi *\varphi *u -\Delta u = \tfrac{
 * \gamma_2}{\gamma_1}\varphi *\tilde{z} -\operatorname{div}\tilde{d}. \f]
 * Compared to UDeconvDct(), the main differences are that the DCT of ztilde
 * is computed and \f$ \mathrm{Ku} = \varphi * u \f$  is updated.
 */
static num UDeconvDctZ(tvregsolver *S)
{
    num *ATrans = S->ATrans;
    num *BTrans = S->BTrans;
    const num *KernelTrans = S->KernelTrans;
    const int NumChannels = S->NumChannels;
    const long NumPixels = ((long)S->Width) * ((long)S->Height);
    long i;
    int k;
    
    /* Compute ATrans = Alpha . KernelTrans . DCT[ztilde] */
    memcpy(S->A, S->ztilde, sizeof(num)*NumPixels*NumChannels);
    AdjBlurDct(ATrans, S->TransformA, KernelTrans,
        S->Width, S->Height, NumChannels, S->Alpha);
    /* BTrans = ( ATrans - DCT[div(dtilde)] ) / DenomTrans */
    UTransSolveDct(BTrans, S->B, S->TransformB, ATrans, S->dtilde,
        S->DenomTrans, S->Width, S->Height, NumChannels);
    
    /* Compute ATrans = KernelTrans . BTrans */
    for(k = 0; k < NumChannels; k++, ATrans += NumPixels, BTrans += NumPixels)
        for(i = 0; i < NumPixels; i++)
            ATrans[i] = KernelTrans[i] * BTrans[i];
    
    /* A = IDCT[ATrans] = new Ku */
    FFT(execute)(S->InvTransformA);
    /* B = IDCT[BTrans] = new u */
    FFT(execute)(S->InvTransformB);
    
    /* Compute ||B - u||, and assign u = B */
    return UUpdate(S);
}
#endif
