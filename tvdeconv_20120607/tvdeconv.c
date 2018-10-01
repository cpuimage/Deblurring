/**
 * @file tvdeconv.c
 * @brief Total variation regularized deconvolution IPOL demo
 * @author Pascal Getreuer <getreuer@gmail.com>
 * 
 * 
 * Copyright (c) 2010-2012, Pascal Getreuer
 * All rights reserved.
 * 
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD License. You
 * should have received a copy of this license along with this program. 
 * If not, see <http://www.opensource.org/licenses/bsd-license.html>.
 */

/**
 * @mainpage
 * @verbinclude readme.txt
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include "tvreg.h"
#include "cliio.h"
#include "kernels.h"


/** @brief Program parameters struct */
typedef struct
{
    /** @brief Input file name */
    const char *InputFile;
    /** @brief Output file name */
    const char *OutputFile;
    /** @brief Quality for saving JPEG images (0 to 100) */
    int JpegQuality;
    
    /** @brief Noise standard deviation */
    num Lambda;
    /** @brief Blur kernel */
    image Kernel;
    /** @brief Noise model */
    const char *Noise;
} programparams;


/** @brief Print program information and usage message */
static void PrintHelpMessage()
{    
    puts("Total variation deconvolution demo, P. Getreuer 2011-2012\n\n"
    "Usage: tvdeconv [param:value ...] input output\n\n"
    "where \"input\" and \"output\" are " 
    READIMAGE_FORMATS_SUPPORTED " files.\n");
    puts("Parameters");
    puts("  K:<kernel>             blur kernel for deconvolution");
    puts("      K:disk:<radius>         filled disk kernel");
    puts("      K:gaussian:<sigma>      Gaussian kernel");
    puts("      K:<file>                read kernel from text or image file");
    puts("  lambda:<value>         fidelity weight");
    puts("  noise:<model>          noisy model");
    puts("      noise:gaussian          additive Gaussian noise (default)");
    puts("      noise:laplace           Laplace noise");
    puts("      noise:poisson           Poisson noise");
    puts("  f:<file>               input file (alternative syntax)");
    puts("  u:<file>               output file (alternative syntax)");
#ifdef USE_LIBJPEG
    puts("  jpegquality:<number>   quality for saving JPEG images (0 to 100)");
#endif
    puts("\nExample: \n"
    "   imblur noise:gaussian:5 K:disk:2 input.bmp blurry.bmp\n");
}

int TvDeconv(image u, image f, image Kernel, num Lambda, const char *Noise);
int ParseParams(programparams *Params, int argc, const char *argv[]);

int main(int argc, char **argv)
{
    programparams Params;
    image f = NullImage, u = NullImage;
    int Status = 1;
    
    if(!ParseParams(&Params, argc, (const char **)argv))
        goto Catch;    
    
    /* Read the input image */
    if(!ReadImageObj(&f, Params.InputFile))
        goto Catch;
    else if(!AllocImageObj(&u, f.Width, f.Height, f.NumChannels))
    {
        fputs("Out of memory.\n", stderr);
        goto Catch;
    }
    
    if(!TvDeconv(u, f, Params.Kernel, Params.Lambda, Params.Noise))
        goto Catch;
    
    /* Write the deconvolved image */
    if(!WriteImageObj(u, Params.OutputFile, Params.JpegQuality))    
        fprintf(stderr, "Error writing to \"%s\".\n", Params.OutputFile);
    
    Status = 0;
Catch:
    FreeImageObj(u);
    FreeImageObj(f); 
    FreeImageObj(Params.Kernel);
    return Status;
}


int TvDeconv(image u, image f, image Kernel, num Lambda, const char *Noise)
{
    tvregopt *Opt = NULL;
    int Success;
    
    if(!(Opt = TvRegNewOpt()))
    {
        fputs("Out of memory.\n", stderr);
        return 0;
    }
    else if(!(TvRegSetNoiseModel(Opt, Noise)))
    {
        fprintf(stderr, "Unknown noise model, \"%s\".\n", Noise);
        TvRegFreeOpt(Opt);
        return 0;
    }
    
    memcpy(u.Data, f.Data, sizeof(num)*((size_t)f.Width)
        *((size_t)f.Height)*f.NumChannels);
    TvRegSetKernel(Opt, Kernel.Data, Kernel.Width, Kernel.Height);
    TvRegSetLambda(Opt, Lambda);
    TvRegSetMaxIter(Opt, 140);
    
    if(!(Success = TvRestore(u.Data, f.Data, 
        f.Width, f.Height, f.NumChannels, Opt)))
        fputs("Error in computation.\n", stderr);
    
    TvRegFreeOpt(Opt);
    return Success;
}


/** @brief Parse command line arguments */
int ParseParams(programparams *Params, int argc, const char *argv[])
{
    static const char *DefaultOutputFile = (char *)"out.bmp";
    const char *Param, *Value;
    num NumValue;
    char TokenBuf[256];
    int k, kread, Skip;
    
        
    /* Set parameter defaults */
    Params->InputFile = NULL;
    Params->OutputFile = DefaultOutputFile;
    Params->JpegQuality = 85;
    
    Params->Lambda = 20;    
    Params->Kernel = NullImage;
    Params->Noise = "gaussian";
        
    if(argc < 2)
    {
        PrintHelpMessage();
        return 0;
    }    
    
    k = 1;
    
    while(k < argc)
    {
        Skip = (argv[k][0] == '-') ? 1 : 0;        
        kread = CliParseArglist(&Param, &Value, TokenBuf, sizeof(TokenBuf),
            k, &argv[k][Skip], argc, argv, ":");        
       
        if(!Param)
        {
            if(!Params->InputFile)
                Param = (char *)"f";
            else
                Param = (char *)"u";
        }
        
        if(Param[0] == '-')     /* Argument begins with two dashes "--" */
        {
            PrintHelpMessage();
            return 0;
        }

        if(!strcmp(Param, "f") || !strcmp(Param, "input"))
        {
            if(!Value)
            {
                fprintf(stderr, "Expected a value for option %s.\n", Param);
                return 0;
            }
            Params->InputFile = Value;
        }
        else if(!strcmp(Param, "u") || !strcmp(Param, "output"))
        {
            if(!Value)
            {
                fprintf(stderr, "Expected a value for option %s.\n", Param);
                return 0;
            }
            Params->OutputFile = Value;
        }
        else if(!strcmp(Param, "K"))
        {
            if(!Value)
            {
                fprintf(stderr, "Expected a value for option %s.\n", Param);
                return 0;
            }
            else if(!ReadKernel(&Params->Kernel, Value))
                return 0;
        }
        else if(!strcmp(Param, "lambda"))
        {
            if(!CliGetNum(&NumValue, Value, Param))
                return 0;
            else if(NumValue <= 0)
            {
                fputs("Parameter lambda must be positive.\n", stderr);
                return 0;
            } 
            else
                Params->Lambda = (int)NumValue;
        }
        else if(!strcmp(Param, "noise"))
        {
            if(!Value)
            {
                fprintf(stderr, "Expected a value for option %s.\n", Param);
                return 0;
            }
            else
                Params->Noise = Value;
        }
        else if(!strcmp(Param, "jpegquality"))
        {
            if(!CliGetNum(&NumValue, Value, Param))
                return 0;
            else if(NumValue < 0 || 100 < NumValue)
            {
                fputs("JPEG quality must be between 0 and 100.\n", stderr);
                return 0;
            } 
            else
                Params->JpegQuality = (int)NumValue;
        }
        else if(Skip)
        {
            fprintf(stderr, "Unknown option \"%s\".\n", Param);
            return 0;
        }
        else
        {
            if(!Params->InputFile)
                Params->InputFile = argv[k];
            else
                Params->OutputFile = argv[k];
            
            kread = k;
        }
        
        k = kread + 1;
    }
    
    if(!Params->Kernel.Data && !ReadKernel(&Params->Kernel, "disk:0"))
        return 0;
    
    if(!Params->InputFile)
    {
        PrintHelpMessage();
        return 0;
    }

    return 1;
}
