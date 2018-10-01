/**
 * @file dsolve_inc.c
 * @brief Solve the d subproblem
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

#include "tvregopt.h"


/** 
 * @brief Solve the d subproblem with vectorial shrinkage
 * @param S tvreg solver state
 * 
 * This routine solves the d subproblem to update d,
 * \f[ \operatorname*{arg\,min}_{d}\,\sum_{i,j}\lvert d_{i,j}\rvert+\frac{
 * \gamma}{2}\sum_{i,j}\lvert d_{i,j}-b_{i,j}-\nabla u_{i,j}\rvert^2, \f]
 * where \f$ \nabla \f$ is the discrete gradient and the second term is a 
 * penalty to encourage the constraint \f$ d = \nabla u \f$.  The solution is
 * the vectorial shrinkage with shrinkage parameter \f$ 1/\gamma \f$,
 * \f[ d_{i,j}=\frac{\nabla u_{i,j}+b_{i,j}}{\lvert\nabla u_{i,j}+b_{i,j}
 * \rvert}\max\bigl\{\lvert\nabla u_{i,j}+b_{i,j}\rvert-1/\gamma,0\bigr\}. \f]
 * The discrete gradient of u is computed with forward differences.  At the 
 * right and bottom boundaries, the difference is set to zero.
 * 
 * The routine also updates the auxiliary variable b according to
 * \f[ b = b + \nabla u - d. \f]
 * Rather than representing b directly, we use  \f$ \tilde d = d - b \f$, 
 * which is algebraically equivalent but requires less arithmetic.
 * 
 * To represent the vector field d, we implement d as a numvec2 array of 
 * size Width x Height x NumChannels such that
@code
    d[i + Width*(j + Height*k)].x = x-component at pixel (i,j) channel k,
    d[i + Width*(j + Height*k)].y = y-component at pixel (i,j) channel k,
@endcode
 * where i = 0, ..., Width-1, j = 0, ..., Height-1, and k = 0, ..., 
 * NumChannels-1.  This structure is also used for \f$ \tilde d \f$.
 */
static void DSolve(tvregsolver *S)
{
    numvec2 *d = S->d;
    numvec2 *dtilde = S->dtilde;
    const num *u = S->u;    
    const int Width = S->Width;
    const int Height = S->Height;
    const int NumChannels = S->NumChannels;
    const num Thresh = 1/S->Opt.Gamma1;
    const num ThreshSquared = Thresh * Thresh;
    const long ChannelStride = ((long)Width) * ((long)Height);
    const long NumEl = NumChannels * ChannelStride;
    numvec2 dnew;
    num Magnitude;
    long i;
    int x, y;
    
    for(y = 0; y < Height - 1; y++, d++, dtilde++, u++)
    {
        /* Perform vectorial shrinkage for interior points */
        for(x = 0; x < Width - 1; x++, d++, dtilde++, u++)
        {
            for(i = 0, Magnitude = 0; i < NumEl; i += ChannelStride)
            {
                d[i].x += (u[i + 1]     - u[i]) - dtilde[i].x;
                d[i].y += (u[i + Width] - u[i]) - dtilde[i].y;
                Magnitude += d[i].x*d[i].x + d[i].y*d[i].y;
            }
            
            if(Magnitude > ThreshSquared)
            {
                Magnitude = 1 - Thresh/(num)sqrt(Magnitude);
                
                for(i = 0; i < NumEl; i += ChannelStride)
                {  
                    dnew.x = Magnitude*d[i].x;
                    dnew.y = Magnitude*d[i].y;
                    dtilde[i].x = 2*dnew.x - d[i].x;
                    dtilde[i].y = 2*dnew.y - d[i].y;
                    d[i] = dnew;
                }
            }
            else
                for(i = 0; i < NumEl; i += ChannelStride)
                {
                    dtilde[i].x = -d[i].x;
                    dtilde[i].y = -d[i].y;
                    d[i].x = 0;
                    d[i].y = 0;
                }
        }
        
        /* Right edge */
        for(i = 0, Magnitude = 0; i < NumEl; i += ChannelStride)
        {
            d[i].y += (u[i + Width] - u[i]) - dtilde[i].y;
            Magnitude += d[i].y*d[i].y;
            d[i].x = dtilde[i].x = 0;
        }
        
        if(Magnitude > ThreshSquared)
        {
            Magnitude = 1 - Thresh/(num)sqrt(Magnitude);
            
            for(i = 0; i < NumEl; i += ChannelStride)
            {
                dnew.y = Magnitude*d[i].y;
                dtilde[i].y = 2*dnew.y - d[i].y;
                d[i].y = dnew.y;
            }
        }
        else
            for(i = 0; i < NumEl; i += ChannelStride)
            {
                dtilde[i].y = -d[i].y;
                d[i].y = 0;
            }
    }
    
    /* Bottom edge */
    for(x = 0; x < Width - 1; x++, d++, dtilde++, u++)
    {
        for(i = 0, Magnitude = 0; i < NumEl; i += ChannelStride)
        {
            d[i].x += (u[i + 1] - u[i]) - dtilde[i].x;
            Magnitude += d[i].x*d[i].x;
            d[i].y = dtilde[i].y = 0;
        }
        
        if(Magnitude > ThreshSquared)
        {
            Magnitude = 1 - Thresh/(num)sqrt(Magnitude);
            
            for(i = 0; i < NumEl; i += ChannelStride)
            {
                dnew.x = Magnitude*d[i].x;
                dtilde[i].x = 2*dnew.x - d[i].x;
                d[i].x = dnew.x;
            }
        }
        else
            for(i = 0; i < NumEl; i += ChannelStride)
            {
                dtilde[i].x = -d[i].x;
                d[i].x = 0;
            }
    }
    
    /* Bottom-right corner */
    for(i = 0; i < NumEl; i += ChannelStride)
        d[i].x = d[i].y = dtilde[i].x = dtilde[i].y = 0;
}
