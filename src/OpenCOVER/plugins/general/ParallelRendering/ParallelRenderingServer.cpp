/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <ParallelRenderingServer.h>

#include "ParallelRenderingCompositor.h"

#include <iostream>
using std::cerr;
using std::endl;

#ifdef __linux__
#define HAVE_POSIX_MEMALIGN
#endif

ParallelRenderingServer::ParallelRenderingServer(int numClients, bool compositorRenders)
{

    this->image = NULL;
    this->numClients = numClients;
    this->pixels = 0;
    this->compositorRenders = compositorRenders;
    startClient = compositorRenders ? 1 : 0;

    frameTime = 0.0;
    connected = false;

    compositors = new ParallelRenderingCompositor *[numClients];

    this->externalPixelFormat = GL_BGRA;

    osg::Camera *camera = cover->screens[0].camera.get();

    const osg::Viewport *vp = camera->getViewport();
    width = (unsigned)vp->width();
    height = (unsigned)vp->height();

#ifdef HAVE_POSIX_MEMALIGN
    void *buf = NULL;
    if (posix_memalign(&buf, sysconf(_SC_PAGESIZE), width * height * 4) != 0)
        std::cerr << "allocation of page-aligned image buffer failed" << std::endl;
    else
        image = (unsigned char *)buf;
#endif
    if (!image)
        image = (unsigned char *)malloc(width * height * 4);

    keepRunning = true;
}

ParallelRenderingServer::~ParallelRenderingServer()
{

    delete[] compositors;
}

void ParallelRenderingServer::readBackImage()
{
    //cerr << "ParallelRenderingClient::readBackImage info: reading" << endl;

    osg::Camera *camera = cover->screens[0].camera.get();
    const osg::Viewport *vp = camera->getViewport();

    if (width != (unsigned)vp->width() || height != (unsigned)vp->height() || image == NULL)
    {

        if (image)
        {
            free(image);
            image = NULL;
        }

        width = (unsigned)vp->width();
        height = (unsigned)vp->height();
#ifdef HAVE_POSIX_MEMALIGN
        void *buf = NULL;
        if (posix_memalign(&buf, sysconf(_SC_PAGESIZE), width * height * 4) != 0)
            std::cerr << "allocation of page-aligned image buffer failed" << std::endl;
        else
            image = (unsigned char *)buf;
#endif
        if (!image)
            image = (unsigned char *)malloc(width * height * 4);
    }

    //FIXME Doesn't work for quad stereo
    glReadBuffer(GL_BACK);
    glReadPixels((int)(cover->screens[0].viewportXMin * cover->windows[0].sx),
                 (int)(cover->screens[0].viewportYMin * cover->windows[0].sy),
                 width, height, externalPixelFormat, GL_UNSIGNED_BYTE, image);
}

bool ParallelRenderingServer::isConnected()
{

    return connected;
}

bool ParallelRenderingServer::addCompositor(int channel, ParallelRenderingCompositor *compositor)
{

    if (channel < 0 || channel > numClients - 1)
        return false;

    compositors[channel] = compositor;
    return true;
}

void ParallelRenderingServer::exit()
{

    keepRunning = false;
}
