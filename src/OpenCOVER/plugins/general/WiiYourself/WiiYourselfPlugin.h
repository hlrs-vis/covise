/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _WIIYOURSELF_PLUGIN_H
#define _WIIYOURSELF_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: WiiYourself Plugin support for wiimote as buttonsystem      **
 ** based on a bsd licensed lib called wiiyourself                           **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** dec-09  v1	    				       		                                   **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;
#include <wiimote.h>

class WiiMote : public coVRPlugin
{
public:
    WiiMote();
    ~WiiMote();

    // this will be called to get the button status
    unsigned int button(int station);

private:
    wiimote *wii;
};
#endif
