/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
	FILE: vrml.h

	DESCRIPTION:  Basic includes for modules in vrmlexp

	CREATED BY: greg finch

	HISTORY: created 9, May 1997

 *>	Copyright (c) 1997, All Rights Reserved.
 **********************************************************************/

#ifndef __VRML__H__
#define __VRML__H__

#include <stdio.h>
#include "max.h"
#include "resource.h"
#include "iparamm.h"
#include "bmmlib.h"
#include "utilapi.h"
#include "decomp.h"

//#define DDECOMP
#ifdef DDECOMP
#include "cdecomp.h"
#endif

TCHAR *GetString(int id);

#endif