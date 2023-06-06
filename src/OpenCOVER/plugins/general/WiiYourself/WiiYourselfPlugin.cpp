/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


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

#include "WiiYourselfPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>

WiiMote::WiiMote()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "WiiMote::WiiMote\n");
    wii = new wiimote();
    int count = 0;
    while (!wii->Connect(wiimote::FIRST_AVAILABLE))
    {
        _tprintf(_T("Try to connect WiiMote\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b%d "), count % 3);
        count++;
#ifdef USE_BEEPS_AND_DELAYS
        Beep(500, 30);
        Sleep(1000);
#endif
    }
}

// this is called if the plugin is removed at runtime
WiiMote::~WiiMote()
{
    fprintf(stderr, "WiiMote::~WiiMote\n");
    delete wii;
}

unsigned int WiiMote::button(int station)
{
    if (!wii->IsConnected())
    {
        int count = 0;
        //reconnect
        while (!wii->Connect(wiimote::FIRST_AVAILABLE))
        {
            _tprintf(_T("Try to reconnect WiiMote\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b%d "), count % 3);
            count++;
#ifdef USE_BEEPS_AND_DELAYS
            Beep(500, 30);
            Sleep(1000);
#endif
        }
    }

    wii->RefreshState();
    unsigned int b = 0;
    if (wii->Button.B())
        b = 1;
    if (wii->Button.One())
        b = 2;
    if (wii->Button.A())
        b = 4;
    if (wii->Button.Two())
        b = 8;
    if (wii->Button.Plus())
        b = 0x10;
    if (wii->Button.Minus())
        b = 0x20;
    if (wii->Button.Home())
        b = 0x30;
    if (wii->Button.Left())
        b = 0x40000000;
    if (wii->Button.Right())
        b = 0x10000000;
    if (wii->Button.Up())
        b = 0x80000000;
    if (wii->Button.Down())
        b = 0x20000000;
    return b;
}

COVERPLUGIN(WiiMote)
