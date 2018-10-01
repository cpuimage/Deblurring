/**
 * @file usolve_dft_inc.c
 * @brief u-subproblem DFT solver for TV-regularized deconvolution
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
 * @brief Symmetrically pad an image to twice its size
 * @param Dest the destination
 * @param Src the source image
 * @param Width, Height, NumChannels the dimensions of Src
 * 
 * The Src image of size Width by Height is reflected over each axis to 
 * create an image that is 2*Width by 2*Height.
 */
static void SymmetricPadding(num *Dest, const num *Src, 
    int Width, int Height, int NumChannels)
{
    const int InPlace = (Dest == Src);
    const int PadWidth = 2*Width;
    const long ChannelJump = ((long)PadWidth) * ((long)Height);
    const int SrcStride = (InPlace) ? PadWidth : Width;
    int x, y, k;
    
    for(k = 0; k < NumChannels; k++)
    {
        for(y = 0; y < Height; y++, Dest += PadWidth, Src += SrcStride)
        {
            if(!InPlace)
                memcpy(Dest, Src, sizeof(num) * Width);
            
            for(x = 0; x < Width; x++)
                Dest[Width + x] = Dest[Width - 1 - x];
            
            memcpy(Dest + ((long)(2*(Height - y) - 1)) * PadWidth,
                Dest, sizeof(num) * PadWidth);
        }
        
        Dest += ChannelJump;
        
        if(InPlace)
            Src = Dest;
    }
}


/** @brief Compute ATrans = Alpha . conj(KernelTrans) . DFT[ztilde] */
static void AdjBlurFourier(numcomplex *ATrans, num *A, FFT(plan) TransformA,
    const numcomplex *KernelTrans, const num *ztilde,
    int Width, int Height, int NumChannels, num Alpha)
{
    const int PadWidth = 2*Width;
    const int PadHeight = 2*Height;    
    const int TransWidth = PadWidth/2 + 1;
    const long TransNumPixels = ((long)TransWidth) * ((long)PadHeight);
    long i;
    int k;
    
    /* Compute A as a symmetric padded version of ztilde */
    SymmetricPadding(A, ztilde, Width, Height, NumChannels);
    
    /* Compute ATrans = DFT[A] */
    FFT(execute)(TransformA);
    
    /* Compute ATrans = Alpha . conj(KernelTrans) . ATrans */
    for(k = 0; k < NumChannels; k++, ATrans += TransNumPixels)
        for(i = 0; i < TransNumPixels; i++)
        {
            num Temp = Alpha*(KernelTrans[i][0] * ATrans[i][1] 
                - KernelTrans[i][1] * ATrans[i][0]);
            ATrans[i][0] = Alpha*(KernelTrans[i][0] * ATrans[i][0] 
                + KernelTrans[i][1] * ATrans[i][1]);
            ATrans[i][1] = Temp;
        }
}


/** 
 * @brief Intializations to prepare TvRestore for Fourier deconvolution 
 * @param S tvreg solver state
 * @return 1 on success, 0 on failure
 */
static int InitDeconvFourier(tvregsolver *S)
{    
    num *B = S->B;
    numcomplex *ATrans = (numcomplex *)S->ATrans;
    numcomplex *BTrans = (numcomplex *)S->BTrans;
    numcomplex *KernelTrans = (numcomplex *)S->KernelTrans;
    num *DenomTrans = S->DenomTrans;        
    const num *Kernel = S->Opt.Kernel;
    const int KernelWidth = S->Opt.KernelWidth;
    const int KernelHeight = S->Opt.KernelHeight;
    const int PadWidth = S->PadWidth;
    const int PadHeight = S->PadHeight;
    const num Alpha = S->Alpha;
    const long PadNumPixels = ((long)PadWidth) * ((long)PadHeight);
    const int TransWidth = PadWidth/2 + 1;    
    FFT(plan) Plan = NULL;
    long i;
    int PadSize[2], x0, y0, x, y, xi, yi;
    int exit;
    
    for(i = 0; i < PadNumPixels; i++)
        B[i] = 0;
    
    x0 = -KernelWidth/2;
    y0 = -KernelHeight/2;
    
    /* Pad Kernel to size PadWidth by PadHeight.  If Kernel
       happens to be larger, it is wrapped. */
    for(y = y0, i = 0; y < y0 + KernelHeight; y++)
    {
        yi = PeriodicExtension(PadHeight, y);
    
        for(x = x0; x < x0 + KernelWidth; x++, i++)
        {
            xi = PeriodicExtension(PadWidth, x);
            B[xi + PadWidth*yi] += Kernel[i];
        }
    }
    
    /* Compute the Fourier transform of the padded Kernel */
#ifdef _OPENMP
#pragma omp critical (fftw)
#endif
    exit = !(Plan = FFT(plan_dft_r2c_2d)(PadHeight, PadWidth, B, 
             KernelTrans, FFTW_ESTIMATE | FFTW_DESTROY_INPUT));
    if (exit)
        return 0;

    FFT(execute)(Plan);
#ifdef _OPENMP
#pragma omp critical (fftw)
#endif
    FFT(destroy_plan)(Plan);
    
    /* Precompute the denominator that will be used in the u-subproblem. */    
    for(y = 0, i = 0; y < PadHeight; y++)
        for(x = 0; x < TransWidth; x++, i++)
            DenomTrans[i] = 
                (num)(PadNumPixels*(Alpha*(KernelTrans[i][0]*KernelTrans[i][0]
                + KernelTrans[i][1]*KernelTrans[i][1])
                + 2*(2 - cos(x*M_2PI/PadWidth) - cos(y*M_2PI/PadHeight))));
    
    /* Plan Fourier transforms */
    PadSize[1] = PadWidth;
    PadSize[0] = PadHeight;

#ifdef _OPENMP
#pragma omp critical (fftw)
#endif
    exit = !(S->TransformA = FFT(plan_many_dft_r2c)(2, PadSize, S->NumChannels, 
        S->A, NULL, 1, PadNumPixels, ATrans, NULL, 1, 
        TransWidth*PadHeight, FFTW_ESTIMATE | FFTW_DESTROY_INPUT))
        || !(S->InvTransformA = FFT(plan_many_dft_c2r)(2, PadSize, 
        S->NumChannels, ATrans, NULL, 1, TransWidth*PadHeight, S->A, 
        NULL, 1, PadNumPixels, FFTW_ESTIMATE | FFTW_DESTROY_INPUT))
        || !(S->TransformB = FFT(plan_many_dft_r2c)(2, PadSize, 
        S->NumChannels, S->B, NULL, 1, PadNumPixels, BTrans, NULL, 1,
        TransWidth*PadHeight, FFTW_ESTIMATE | FFTW_DESTROY_INPUT))
        || !(S->InvTransformB = FFT(plan_many_dft_c2r)(2, PadSize, 
        S->NumChannels, BTrans, NULL, 1, TransWidth*PadHeight, S->B, 
        NULL, 1, PadNumPixels, FFTW_ESTIMATE | FFTW_DESTROY_INPUT));
    if (exit)
        return 0;
    
    /* Compute ATrans = Alpha . conj(KernelTrans) . DFT[f] */
    if(!S->UseZ)
        AdjBlurFourier(ATrans, S->A, S->TransformA, 
            (const numcomplex *)KernelTrans, S->f, 
            S->Width, S->Height, S->NumChannels, Alpha);
    
    S->Ku = S->A;
    return 1;
}


/** 
 * @brief Compute BTrans = ( ATrans - DFT[div(dtilde)] ) / DenomTrans 
 * 
 * This subroutine is a part of the DFT u-subproblem solution that is common 
 * to both the d,u splitting (UseZ=0) and d,u,z splitting (UseZ=1).
 */
static void UTransSolveFourier(numcomplex *BTrans, num *B, FFT(plan) TransformB,
    numcomplex *ATrans, const numvec2 *dtilde, 
    const num *DenomTrans, int Width, int Height, int NumChannels)
{
    const long PadWidth = 2*Width;
    const long PadHeight = 2*Height;    
    const long TransWidth = PadWidth/2 + 1;
    const long TransNumPixels = TransWidth * PadHeight;
    long i;
    int k;
    
    /* Compute B = div(dtilde) and pad with even half-sample symmetry */
    Divergence(B, PadWidth, PadHeight, dtilde, Width, Height, NumChannels);
    SymmetricPadding(B, B, Width, Height, NumChannels);
    
    /* Compute BTrans = DFT[B] */
    FFT(execute)(TransformB);

    /* Compute BTrans = ( ATrans - BTrans ) / DenomTrans */
    for(k = 0; k < NumChannels; k++, 
        ATrans += TransNumPixels, BTrans += TransNumPixels)
        for(i = 0; i < TransNumPixels; i++)
        {
            BTrans[i][0] = (ATrans[i][0] - BTrans[i][0]) / DenomTrans[i];
            BTrans[i][1] = (ATrans[i][1] - BTrans[i][1]) / DenomTrans[i];
        }
}


/** 
 * @brief Solve the u-subproblem using DFT transforms (UseZ = 0)
 *
 * This routine solves the u-subproblem 
 * \f[ \tfrac{\lambda}{\gamma}K^* Ku -\Delta u = \tfrac{\lambda}{
 * \gamma}K^* f -\operatorname{div}\tilde{d}, \f]
 * where K denotes the blur operator \f$ Ku := \varphi * u \f$.  The solution
 * is obtained using the discrete Fourier transform (DFT) as
 * \f[ u=\mathcal{F}^{-1}\left[\frac{\frac{\lambda}{\gamma}\overline{
 * \mathcal{F}(\varphi)}\cdot\mathcal{F}(Ef)- \mathcal{F}\bigl(E
 * \operatorname{div}(d-b)\bigr)}{\frac{\lambda}{\gamma}\lvert\mathcal{F}(
 * \varphi)\rvert^2 - \mathcal{F}(\Delta)}\right], \f]
 * where E denotes symmetric extension and \f$ \mathcal{F} \f$ denotes the 
 * DFT.
 */
static num UDeconvFourier(tvregsolver *S)
{
    /* BTrans = ( ATrans - DFT[div(dtilde)] ) / DenomTrans */
    UTransSolveFourier((numcomplex *)S->BTrans, S->B, S->TransformB, 
        (numcomplex *)S->ATrans, S->dtilde, S->DenomTrans, 
        S->Width, S->Height, S->NumChannels);
    /* B = IDFT[BTrans] */
    FFT(execute)(S->InvTransformB);
    /* Trim padding, compute ||B - u||, and assign u = B */
    return UUpdate(S);
}


#if defined(TVREG_USEZ) || defined(DOXYGEN)
/** 
 * @brief Solve the u-subproblem using DFT transforms (UseZ = 1)
 *
 * This extended version of UDeconvFourier is used when performing Fourier-
 * based deconvolution with the three-auxiliary variable algorithm (UseZ = 1),
 * that is, in a deconvolution problem with a non-symmetric kernel and non-
 * Gaussian noise model.
 */
static num UDeconvFourierZ(tvregsolver *S)
{
    numcomplex *ATrans = (numcomplex *)S->ATrans;
    numcomplex *BTrans = (numcomplex *)S->BTrans;
    const numcomplex *KernelTrans = (const numcomplex *)S->KernelTrans;
    const int TransWidth = S->PadWidth/2 + 1;
    const long TransNumPixels = ((long)TransWidth) * ((long)S->PadHeight);
    long i;
    int k;
    
    /* Compute ATrans = Alpha . conj(KernelTrans) . DFT[ztilde] */
    AdjBlurFourier(ATrans, S->A, S->TransformA, KernelTrans, S->ztilde,
        S->Width, S->Height, S->NumChannels, S->Alpha);
    /* BTrans = ( ATrans - DFT[div(dtilde)] ) / DenomTrans */
    UTransSolveFourier((numcomplex *)S->BTrans, S->B, S->TransformB, 
        ATrans, S->dtilde, S->DenomTrans, 
        S->Width, S->Height, S->NumChannels);
    
    /* Compute ATrans = KernelTrans . BTrans */
    for(k = 0; k < S->NumChannels; k++, 
        ATrans += TransNumPixels, BTrans += TransNumPixels)
        for(i = 0; i < TransNumPixels; i++)
        {
            ATrans[i][0] = KernelTrans[i][0] * BTrans[i][0]
                - KernelTrans[i][1] * BTrans[i][1];
            ATrans[i][1] = KernelTrans[i][0] * BTrans[i][1]
                + KernelTrans[i][1] * BTrans[i][0];
        }
    
    /* A = IDFT[ATrans] = new Ku */
    FFT(execute)(S->InvTransformA);
    /* B = IDFT[BTrans] = new u */
    FFT(execute)(S->InvTransformB);
    
    /* Trim padding, compute ||B - u||, and assign u = B */
    return UUpdate(S);
}
#endif
