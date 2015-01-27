/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _HYDRA_PLUGIN_H
#define _HYDRA_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Tracker Plugin for Razer Hydra                              **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** dec-11  v1	    				       		                                   **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;
#define SIXENSE_STATIC_LIB
#include <sixense.h>

class Hydra : public coVRPlugin
{
public:
    Hydra();
    ~Hydra();
    void preFrame();

    // this will be called to get the button status
    unsigned int button(int station);

private:
    sixenseAllControllerData acd;
    int frame;
};
#endif
