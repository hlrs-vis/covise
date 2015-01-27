/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <ParallelRenderingClient.h>
#include <util/unixcompat.h>
#include <iostream>

#define _XOPEN_SOURCE 600
#include <stdlib.h>

#ifdef __linux__
#define HAVE_POSIX_MEMALIGN
#endif

ParallelRenderingClient::ParallelRenderingClient(int number, const std::string &compositor)
{
    this->image = NULL;
    this->compositor = compositor;
    this->number = number;
    this->connected = false;
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

ParallelRenderingClient::~ParallelRenderingClient()
{

    if (image)
        free(image);
}

void ParallelRenderingClient::readBackImage()
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
    glReadPixels(0, 0, width, height, externalPixelFormat, GL_UNSIGNED_BYTE, image);
}

bool ParallelRenderingClient::isConnected()
{

    return connected;
}

void ParallelRenderingClient::exit()
{

    keepRunning = false;
}
