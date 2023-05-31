/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#define __STDC_CONSTANT_MACROS

// #include <config/CoviseConfig.h>

// #include <cover/coVRTui.h>
#include <cover/coVRConfig.h>
// #include <cover/coVRAnimationManager.h>
// #include <grmsg/coGRMsg.h>
// #include <grmsg/coGRSnapshotMsg.h>

#include <stdio.h>
#include <boost/filesystem.hpp>

#include <config/coConfigConstants.h>
#include <config/CoviseConfig.h>

#include "SharedMemoryImageBuffer.h"
#include "InfoTower.h"
#ifdef WIN32
#include <Shellapi.h>
#endif

InfoTowerPlugin::InfoTowerPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    GL_fmt = GL_BGR_EXT;
    oldWidth = 0;
    oldHeight = 0;
    pixels_left_eye = NULL;
    pixels_right_eye = NULL;
    externalApplicationHandle = NULL;
    sharedMemoryImageBuffer = new SharedMemoryImageBuffer(SharedMemoryImageBuffer::Producer);
}

bool InfoTowerPlugin::init()
{
    cerr << "In InfoTowerPlugin::init()" << endl;
    GL_fmt = GL_BGR_EXT;
    oldWidth = 0;
    oldHeight = 0;
    pixels_left_eye = NULL;
    pixels_right_eye = NULL;

    externalApplication = coCoviseConfig::getEntry("value", "COVER.Plugin.InfoTower.ExternalApplication", "");
    startExternalApplication();

    return true;
}

bool InfoTowerPlugin::destroy()
{
    closeExternalApplication();
    return true;
}

// this is called if the plugin is removed at runtime
InfoTowerPlugin::~InfoTowerPlugin()
{
    closeExternalApplication();
    if (pixels_left_eye != NULL)
        delete[] pixels_left_eye;
    if (pixels_right_eye != NULL)
        delete[] pixels_right_eye;
    delete sharedMemoryImageBuffer;
}

void
InfoTowerPlugin::preSwapBuffers(int windowNumber)
{

    // only capture the first window and only on the master
    if (windowNumber == 0)
    {
        //fprintf(stderr,"glRead...\n");
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        int x, y, width = 0, height = 0;
        coVRConfig::instance()->windows[0].window->getWindowRectangle(x, y, width, height);

        if (width != oldWidth || height != oldHeight)
        {
            try
            {
                sharedMemoryImageBuffer->SetImageSize(width, height, GL_fmt);
            }
            catch (std::exception &e)
            {
                cerr << "Caught exception: " << e.what() << endl;
            }

            cerr << "Setting buffer width = " << width << ", height = " << height << endl;
            oldWidth = width;
            oldHeight = height;
        }
        glReadBuffer(GL_BACK);
        glReadPixels(0, height / 2, width, height / 2, GL_fmt, GL_UNSIGNED_BYTE, sharedMemoryImageBuffer->GetBackImage());

        glReadPixels(0, 0, width, height / 2, GL_fmt, GL_UNSIGNED_BYTE, static_cast<unsigned char *>(sharedMemoryImageBuffer->GetBackImage()) + sharedMemoryImageBuffer->GetBufferSize() / 2);
        sharedMemoryImageBuffer->SwapBuffers();
    }
}

void
InfoTowerPlugin::allocatePixelBuffer(uint8_t **pixel_buffer, GLenum format, uint8_t width, uint8_t height)
{
    if (format != GL_BGR_EXT)
        return;

    if (*pixel_buffer != NULL)
    {
        delete[] * pixel_buffer;
        *pixel_buffer = NULL;
    }

    cerr << "Allocating pixel buffer width = " << width << ", height = " << height << endl;

    *pixel_buffer = new uint8_t[width * height * 24 / 8];
}

void
InfoTowerPlugin::startExternalApplication()
{
    if (externalApplication == "")
        return;
    if (externalApplicationHandle != NULL)
        return;

    boost::filesystem::path externalAppPath(externalApplication);
    string workingDir = externalAppPath.parent_path().string();
#ifdef WIN32
    SHELLEXECUTEINFO shellExInfo;
    shellExInfo.cbSize = sizeof(SHELLEXECUTEINFO);
    shellExInfo.fMask = SEE_MASK_NOCLOSEPROCESS;
    shellExInfo.hwnd = NULL;
    shellExInfo.lpVerb = "open";
    shellExInfo.lpFile = externalApplication.c_str();
    shellExInfo.lpParameters = "";
    shellExInfo.lpDirectory = workingDir.c_str();
    shellExInfo.nShow = SW_SHOW;
    shellExInfo.hInstApp = NULL;

    std::cerr << "Starting " << shellExInfo.lpFile << " at " << shellExInfo.lpDirectory << std::endl;
    ShellExecuteEx(&shellExInfo);
    externalApplicationHandle = shellExInfo.hProcess;
#else
    chgdir(workingDir.c_str());
    execvp(externalApplication.c_str(), externalApplication.c_str());
#endif
}

void
InfoTowerPlugin::closeExternalApplication()
{
    if (externalApplicationHandle != NULL)
    {
        TerminateProcess(externalApplicationHandle, 1);
        externalApplicationHandle = NULL;
    }
}

COVERPLUGIN(InfoTowerPlugin)
