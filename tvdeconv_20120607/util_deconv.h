/**
 * @file util_deconv.h
 * @brief Utility routines used in both DCT and DFT based deconvolution
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
#ifndef _UTIL_DECONV_H_
#define _UTIL_DECONV_H_

#include "tvregopt.h"


/**
 * @brief Compute discrete 2D divergence
 * @param DivV the divergence of V
 * @param DivWidth, DivHeight the dimensions of DivV
 * @param V input vector field 
 * @param Width, Height, NumChannels the dimensions of V
 * 
 * The discrete divergence is defined as the negative adjoint of the discrete
 * gradient operator, \f$ \operatorname{div}:=-\nabla^*
 * =-\partial^*_x-\partial^*_y, \f$ where \f$ -\partial^* \f$ is
 * \f[ \begin{pmatrix}-\partial^* g_0 \\-\partial^* g_1 \\ \vdots \\
 * -\partial^* g_{N-2} \\ -\partial^* g_{N-1}\end{pmatrix} = \begin{pmatrix}
 * \hphantom{-}1 & & & & \\ -1 & 1 & & & \\ & \ddots & \ddots & & \\ & & -1 & 
 * \hphantom{-}1 & \\ & & & -1 & 0 \end{pmatrix}\begin{pmatrix} g_0 \\ g_1 \\ 
 * \vdots \\ g_{N-2} \\ g_{N-1}\end{pmatrix}. \f]
 * In the interior of the domain, the discrete divergence reduces to backward
 * differences, 
 * \f[ \operatorname{div}V_{i,j}=V^x_{i,j}-V^x_{i-1,j}
 * +V^y_{i,j}-V^y_{i,j-1}, \f]
 * for i = 1, ..., Width-2, j = 1, ..., Height-2.
 * 
 * The input vector field V is represented as an array of numvec2 elements,
@code
    V[i + Width*(j + Height*k)].x = x-component at pixel (i,j) channel k,
    V[i + Width*(j + Height*k)].y = y-component at pixel (i,j) channel k,
@endcode
 * where i = 0, ..., Width-1, j = 0, ..., Height-1, and k = 0, ..., 
 * NumChannels-1.  
 */
static void Divergence(num *DivV, int DivWidth, int DivHeight,
    const numvec2 *V, int Width, int Height, int NumChannels)
{
    int x, y, k;
    
    for(k = 0; k < NumChannels; k++)
    {
        /* Top-left corner */
        DivV[0] = V[0].x + V[0].y;
        
        /* Top row, x = 1, ..., Width - 2 */
        for(x = 1; x < Width - 1; x++)
            DivV[x] = V[x].x - V[x - 1].x + V[x].y;
        
        /* Top-right corner */
        DivV[x] = V[x].y;
        DivV += DivWidth;
        V += Width;
        
        for(y = 1; y < Height - 1; y++, DivV += DivWidth, V += Width)
        {
            /* Left edge */
            DivV[0] = V[0].x + V[0].y - V[-Width].y;
            
            /* Interior */
            for(x = 1; x < Width - 1; x++)
                DivV[x] = V[x].x - V[x - 1].x 
                    + V[x].y - V[x - Width].y;
            
            /* Top-right corner */
            DivV[x] = V[x].y - V[x - Width].y;
        }
        
        /* Bottom-reft corner */
        DivV[0] = V[0].x;
        
        /* Bottom row, x = 1, ..., Width - 2 */
        for(x = 1; x < Width - 1; x++)
            DivV[x] = V[x].x - V[x - 1].x;
        
        /* Bottom-right corner */
        DivV[x] = 0;
        DivV += ((long)DivWidth)*((long)DivHeight - Height + 1);
        V += Width;
    }
}


/** 
 * @brief Trims padding, computes ||B - u||, and assigns u = B 
 * @param S tvreg solver state
 * @return the norm ||B - u||
 */
static num UUpdate(tvregsolver *S)
{
    num *u = S->u;
    const num *B = S->B;
    const int Width = S->Width;
    const int Height = S->Height;
    const int PadWidth = S->PadWidth;
    const int PadHeight = S->PadHeight;
    const long PadJump = ((long)PadWidth) * (PadHeight - Height);
    num Norm = 0;
    int x, y, k;
    
    for(k = 0; k < S->NumChannels; k++, B += PadJump)
        for(y = 0; y < Height; y++, u += Width, B += PadWidth)
            for(x = 0; x < Width; x++)
            {
                num unew = B[x];
                num Diff = unew - u[x];
                Norm += Diff * Diff;
                u[x] = unew;
            }
    
    return (num)sqrt(Norm) / S->fNorm;
}


/**
 * @brief Boundary handling function for whole-sample symmetric extension
 * @param N is the data length
 * @param i is an index into the data
 * @return an index that is always between 0 and N - 1
 * 
 * Extends data "abcde" to "...cbabcdedcbabcde..."
 */
static ATTRIBUTE_ALWAYSINLINE int WSymExtension(int N, int i)
{
    while(1)
    {
        if(i < 0)
            i = -i;
        else if(i >= N)        
            i = (2*N - 2) - i;
        else
            return i;
    }
}


/**
 * @brief Boundary handling function for periodic extension
 * @param N is the data length
 * @param i is an index into the data
 * @return an index that is always between 0 and N - 1
 * 
 * Extends data "abcde" to "...deabcdeabcdeabc..."
 */
static ATTRIBUTE_ALWAYSINLINE int PeriodicExtension(int N, int i)
{
    while(1)
    {
        if(i < 0)
            i += N;
        else if(i >= N)
            i -= N;
        else
            return i;
    }
}

#endif /* _UTIL_DECONV_H_ */
