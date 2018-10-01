/**
 * @file num.h
 * @brief num typedef
 * @author Pascal Getreuer <getreuer@gmail.com>
 * 
 * This file defines type "num", which by default is a typedef for double.
 * If NUM_SINGLE is defined, then num is a typedef for float.
 * 
 * Copyright (c) 2010-2012, Pascal Getreuer
 * All rights reserved.
 * 
 * This program is free software: you can use, modify and/or 
 * redistribute it under the terms of the simplified BSD License. You 
 * should have received a copy of this license along this program. If 
 * not, see <http://www.opensource.org/licenses/bsd-license.html>.
 */
#ifndef _NUM_H_
#define _NUM_H_

#ifdef NUM_SINGLE
/* Use single-precision datatype */
typedef float num;
#else
/* Use double-precision datatype */
typedef double num;
#endif

#endif /* _NUM_H_ */
