/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VIVE_H
#define VIVE_H

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

#include <cover/coVRPlugin.h>


class Vive : public opencover::coVRPlugin
{
public:
    Vive();
    ~Vive();
    void preFrame();
	bool init();

private:
};
#endif
