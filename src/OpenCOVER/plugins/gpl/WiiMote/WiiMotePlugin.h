#ifndef _TEMPLATE_PLUGIN_H
#define _TEMPLATE_PLUGIN_H
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
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include "Wiimote.h"

class WiiMote : public coVRPlugin
{
public:
    WiiMote();
    ~WiiMote();

    // this will be called to get the button status
    unsigned int button(int station);

private:
    Wiimote *wii;
};
#endif
