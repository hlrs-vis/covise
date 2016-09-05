/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SVXMLParser_H
#define SVXMLParser_H

/****************************************************************************\ 
 **                                                            (C)2016 HLRS  **
 **                                                                          **
 ** Description: Streetview Plugin - XML Parser	                             **
 **                                                                          **
 **                                                                          **
 ** Author: M.Guedey		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** Sep-16  v1	    				       		                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>

using namespace opencover;
using namespace vrml;

class SVXMLParser 
{
public:
    SVXMLParser();
    ~SVXMLParser();
    void preFrame();
	bool init();

private:
};
#endif
