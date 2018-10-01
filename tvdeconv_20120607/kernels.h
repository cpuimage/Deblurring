/**
 * @file kernels.h
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

#ifndef _KERNELS_H_
#define _KERNELS_H_

#include "cliio.h"

int ReadKernel(image *Kernel, const char *String);

#endif /* _KERNELS_H_ */
