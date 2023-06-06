/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: Hello OpenCOVER Plugin (is polite)                          **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** June 2008  v1	    				       		                                **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "Hello.h"
#include <cover/coVRPluginSupport.h>
using namespace opencover;
Hello::Hello()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "Hello World\n");
}

// this is called if the plugin is removed at runtime
Hello::~Hello()
{
    fprintf(stderr, "Goodbye\n");
}

COVERPLUGIN(Hello)
