/** 
 * @file basic.c
 * @brief Memory management, portable types, and math constants
 * @author Pascal Getreuer <getreuer@gmail.com>
 * 
 * Copyright (c) 2010-2011, Pascal Getreuer
 * All rights reserved.
 * 
 * This program is free software: you can use, modify and/or 
 * redistribute it under the terms of the simplified BSD License. You 
 * should have received a copy of this license along this program. If 
 * not, see <http://www.opensource.org/licenses/bsd-license.html>.
 */

#include <stdlib.h>
#include <stdarg.h>
#include "basic.h"


/** @brief malloc with an error message on failure. */
void *MallocWithErrorMessage(size_t Size)
{
    void *Ptr;
    
    if(!(Ptr = malloc(Size)))
        ErrorMessage("Memory allocation of %u bytes failed.\n", Size);
        
    return Ptr;
}


/** @brief realloc with an error message and free on failure. */
void *ReallocWithErrorMessage(void *Ptr, size_t Size)
{
    void *NewPtr;
    
    if(!(NewPtr = realloc(Ptr, Size)))
    {
        ErrorMessage("Memory reallocation of %u bytes failed.\n", Size);
        Free(Ptr);  /* Free the previous block on failure */
    }
        
    return NewPtr;
}


/** @brief Redefine this function to customize error messages. */
void ErrorMessage(const char *Format, ...)
{
    va_list Args;
    
    va_start(Args, Format);
    /* Write a formatted error message to stderr */
    vfprintf(stderr, Format, Args);
    va_end(Args);
}
