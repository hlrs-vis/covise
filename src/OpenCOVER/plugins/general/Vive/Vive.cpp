/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

/****************************************************************************\
**                                                            (C)2016 HLRS  **
**                                                                          **
** Description: Vive Plugin				                                 **
**                                                                          **
**                                                                          **
** Author: Uwe Woessner		                                             **
**                                                                          **
** History:  								                                 **
** Sep-16  v1	    				       		                             **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include "Vive.h"

#include <iostream>

#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>

using namespace opencover;

Vive::Vive()
{
}

bool Vive::init()
{
	fprintf(stderr, "Vive::init\n");
	return true;
}

// this is called if the plugin is removed at runtime
Vive::~Vive()
{
	fprintf(stderr, "Vive::~Vive\n");
}

void Vive::preFrame()
{
	
}

COVERPLUGIN(Vive)

