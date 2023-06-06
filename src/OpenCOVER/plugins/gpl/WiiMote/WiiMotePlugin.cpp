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

#include "WiiMotePlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>

WiiMote::WiiMote()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "WiiMote::WiiMote\n");
    wii = new Wiimote();
}

// this is called if the plugin is removed at runtime
WiiMote::~WiiMote()
{
    fprintf(stderr, "WiiMote::~WiiMote\n");
}

unsigned int WiiMote::button(int station)
{
    unsigned int b = 0;
    wii->getButtons(station, &b);
    return b;
}

COVERPLUGIN(WiiMote)
