/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Template Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "Reallabor.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>

using namespace opencover;

Reallabor::Reallabor()
{
    fprintf(stderr, "Reallabor::Reallabor\n");
}

// this is called if the plugin is removed at runtime
Reallabor::~Reallabor()
{
    fprintf(stderr, "Reallabor::~Reallabor\n");
}

void
Reallabor::preFrame()
{
}

COVERPLUGIN(Reallabor)
