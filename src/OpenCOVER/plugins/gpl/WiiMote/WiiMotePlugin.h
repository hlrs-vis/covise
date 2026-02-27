#ifndef _TEMPLATE_PLUGIN_H
#define _TEMPLATE_PLUGIN_H

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
