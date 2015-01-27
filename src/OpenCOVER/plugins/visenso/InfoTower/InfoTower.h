/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _InfoTower_NODE_PLUGIN_H
#define _InfoTower_NODE_PLUGIN_H

#include <util/common.h>

// #include <OpenVRUI/sginterface/vruiActionUserData.h>

// #include <OpenVRUI/coMenuItem.h>
// #include <OpenVRUI/coMenu.h>
// #include <cover/coTabletUI.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>

// #include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>

// #include <cover/coVRMSController.h>
// #include <cover/coVRPluginSupport.h>

#include <config/CoviseConfig.h>

// #include <util/byteswap.h>
// #include <util/coFileUtil.h>

#ifndef M_PI
#define M_PI 3.1415926535897931
#endif

using namespace covise;
using namespace opencover;

class InfoTowerPlugin : public coVRPlugin
{
public:
    InfoTowerPlugin();
    ~InfoTowerPlugin();
    bool init();
    bool destroy();

    void preSwapBuffers(int windowNumber);

private:
    static void allocatePixelBuffer(uint8_t **pixel_buffer, GLenum format, uint width, uint height);
    int oldWidth, oldHeight;
    uint8_t *pixels_left_eye;
    uint8_t *pixels_right_eye;
    class SharedMemoryImageBuffer *sharedMemoryImageBuffer;

    GLenum GL_fmt;

    std::string externalApplication;
    void startExternalApplication();
    void closeExternalApplication();
    HANDLE externalApplicationHandle;
};
#endif
