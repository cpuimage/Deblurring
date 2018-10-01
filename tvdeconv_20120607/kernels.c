/**
 * @file kernels.c
 * @brief Convolution kernels 
 * @author Pascal Getreuer <getreuer@gmail.com>
 * 
 * 
 * Copyright (c) 2011, Pascal Getreuer
 * All rights reserved.
 * 
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD License. You
 * should have received a copy of this license along with this program. 
 * If not, see <http://www.opensource.org/licenses/bsd-license.html>.
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "kernels.h"

#ifndef M_SQRT1_2
/** @brief The constant sqrt(1/2) */
#define M_SQRT1_2   0.70710678118654752440084436210484904
#endif

#ifndef M_PI
/** @brief The constant pi */
#define M_PI        3.14159265358979323846264338327950288
#endif


static int KernelRescale(image *Kernel)
{    
    const int NumEl = Kernel->Width * Kernel->Height;
    num *Data = Kernel->Data;
    num Sum = 0;
    int n;

    for(n = 0; n < NumEl; n++)
        Sum += Data[n];

    if(Sum == 0)
    {
        fprintf(stderr, "Kernel must have nonzero sum.\n");
        return 0;
    }

    for(n = 0; n < NumEl; n++)
        Data[n] /= Sum;

    return 1;
}


static int GaussianKernel(image *Kernel, num Sigma)
{
    const int R = (int)ceil(4*Sigma);
    const int W = 2*R + 1;
    const double ExpDenom = 2.0*(double)Sigma*(double)Sigma;
    num *Data;
    double Sum;
    int x, y;
    

    if(!Kernel || Sigma < 0.0 || !(AllocImageObj(Kernel, W, W, 1)))
        return 0;
    
    Data = Kernel->Data;
    
    if(Sigma == 0.0)
        Data[0] = 1;
    else
    {
        for(y = -R, Sum = 0; y <= R; y++)
            for(x = -R; x <= R; x++)
            {
                Data[(R + x) + W*(R + y)] = (num)exp(-(x*x + y*y)/ExpDenom);
                Sum += Data[(R + x) + W*(R + y)];
            }

        for(y = -R; y <= R; y++)
            for(x = -R; x <= R; x++)
                Data[(R + x) + W*(R + y)] = 
                    (num)(Data[(R + x) + W*(R + y)]/Sum);
    }
    
    return 1;
}


static int DiskKernel(image *Kernel, num Radius)
{
    const int R = (int)ceil(Radius - 0.5);
    const int W = 2*R + 1;
    const int Res = 8;
    const double RadiusSqr = (double)Radius*(double)Radius;
    const double RadiusInnerSqr = 
        ((double)Radius - M_SQRT1_2)*((double)Radius - M_SQRT1_2);
    const double RadiusOuterSqr = 
        ((double)Radius + M_SQRT1_2)*((double)Radius + M_SQRT1_2);
    num *Data;
    double Sum, xl, yl, Start = -0.5 + 0.5/Res, Step = 1.0/Res;
    int c, x, y, m, n;
    

    if(!Kernel || Radius < 0.0 || !(AllocImageObj(Kernel, W, W, 1)))
        return 0;
    
    Data = Kernel->Data;
    
    if(Radius <= 0.5)
        Data[0] = 1;
    else
    {
        for(y = -R, Sum = 0; y <= R; y++)
            for(x = -R; x <= R; x++)
            {
                if(x*x + y*y <= RadiusInnerSqr)
                    c = Res*Res;
                else if(x*x + y*y > RadiusOuterSqr)
                    c = 0;
                else
                    for(n = 0, yl = y + Start, c = 0; n < Res; n++, yl += Step)
                        for(m = 0, xl = x + Start; m < Res; m++, xl += Step)
                            if(xl*xl + yl*yl <= RadiusSqr)
                                c++;
                            
                Data[(R + x) + W*(R + y)] = (num)c;
                Sum += c;
            }

        for(y = -R; y <= R; y++)
            for(x = -R; x <= R; x++)
                Data[(R + x) + W*(R + y)] =
                    (num)(Data[(R + x) + W*(R + y)]/Sum);
    }

    return 1;
}

    
static int MakeNamedKernel(image *Kernel, const char *String)
{
    const char *ColonPtr;    
    num KernelParam;
    int Length;
    char KernelName[32];
    
    if(!Kernel || !(ColonPtr = strchr(String, ':')) 
        || (Length = (int)(ColonPtr - String)) > 9)
        return 0;
                
    strncpy(KernelName, String, Length);
    KernelName[Length] = '\0';
    KernelParam = (num)atof(ColonPtr + 1);
    
    if(!strcmp(KernelName, "Gaussian") || !strcmp(KernelName, "gaussian"))
        return GaussianKernel(Kernel, KernelParam);
    else if(!strcmp(KernelName, "Disk") || !strcmp(KernelName, "disk"))
        return DiskKernel(Kernel, KernelParam);
    else
        return 0;
}


/**
 * @brief Read a kernel from the command line option stirng
 * @param Kernel pointer to kernel image
 * @param String command line optionstring
 * @return 1 on success, 0 on failure
 * 
 * The syntax of string can be "disk:<radius>" or "gaussian:<sigma>".
 * Otherwise, the string is assumed to be a file name.
 */ 
int ReadKernel(image *Kernel, const char *String)
{
    if(Kernel->Data)
    {
        FreeImageObj(*Kernel);
        *Kernel = NullImage;
    }
    
    if(MakeNamedKernel(Kernel, String)
        || ReadMatrixFromFile(Kernel, String, KernelRescale))
        return 1;
    else
        return 0;
}
