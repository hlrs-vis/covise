/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2016 HLRS  **
 **                                                                          **
 ** Description: Streetview Plugin XML Parser	                             **
 **                                                                          **
 **                                                                          **
 ** Author: M.Guedey		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** Sep-16  v1	    				       		                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "SVXMLParser.h"
#include <cover/RenderObject.h>

#include <xercesc/dom/DOM.hpp>

using namespace opencover;

SVXMLParser::SVXMLParser()
{
}

bool SVXMLParser::init()
{
    fprintf(stderr, "SVXMLParser::SVXMLParser\n");

    return true;
}

// this is called if the plugin is removed at runtime
SVXMLParser::~SVXMLParser()
{
    fprintf(stderr, "SVXMLParser::~SVXMLParser\n");
}

void
SVXMLParser::preFrame()
{
}
