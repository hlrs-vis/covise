/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * source/config.h.  Formerly generated automatically by configure.
 */

#ifndef _CONFIG_H_
#define _CONFIG_H_
#include "vrmlexport.h"
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
/*
 * Vrml 97 library
 * Copyright (C) 1998 Chris Morley
 *
 * config.h
 */

#define LIBVRML_MAJOR_VERSION 0
#define LIBVRML_MINOR_VERSION 7
#define LIBVRML_MICRO_VERSION 9

#ifdef __cplusplus
#ifdef _WIN32
static const char SLASH = '\\';
#else
static const char SLASH = '/';
#endif
#endif /* __cplusplus */

/*
 * Make sure that PI and friends are defined.
 * This is needed under platforms that are not directly BSD derived
 * (even under GNU libc this is not included by default).
 */
#ifndef M_PI
#define M_PI 3.14159265358979323846 /* pi */
#endif

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923 /* pi/2 */
#endif

#ifndef M_PI_4
#define M_PI_4 0.78539816339744830962 /* pi/4 */
#endif

#ifndef M_1_PI
#define M_1_PI 0.31830988618379067154 /* 1/pi */
#endif

/* General feature configuration */
#define HAVE_LIBJPEG 1
#define HAVE_LIBPNG 1
#define HAVE_JAVASCRIPT 1
#define HAVE_JDK 0

#include <stdint.h>

#endif /* _CONFIG_H_ */
