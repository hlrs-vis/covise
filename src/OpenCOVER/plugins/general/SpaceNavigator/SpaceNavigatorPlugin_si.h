/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SPACENAVIGATOR_PLUGIN_H
#define _SPACENAVIGATOR_PLUGIN_H

#include <stdio.h>
#include <cover/coVRPluginSupport.h>
#include "si.h"
#include "siapp.h"

typedef struct SMD
{
    float tx;
    float ty;
    float tz;
    float h;
    float p;
    float r;
    unsigned int buttonStatus;
} SpaceMouseData;

class SpaceNavigator : public opencover::coVRPlugin
{
public:
    SpaceNavigator();
    ~SpaceNavigator();

    // this will be called to get the button status
    unsigned int button(int station);
    virtual void preFrame();

private:
    void doNavigation();
    void pollDevice();

    SpaceMouseData m_smd;
    float m_trans_sensitivity;
    float m_rot_sensitivity;
    SiHdl m_dev_handle;
    SiDevInfo m_dev_info;

#ifdef _WIN32
    HWND m_hwnd;
#endif
};

#endif
