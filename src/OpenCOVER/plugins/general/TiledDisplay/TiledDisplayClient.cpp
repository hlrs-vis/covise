/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//#include <winsock2.h>
#include "TiledDisplayClient.h"
#include <util/unixcompat.h>
#include <iostream>

#include <stdlib.h>

#ifdef __linux__
#define HAVE_POSIX_MEMALIGN
#endif

TiledDisplayClient::TiledDisplayClient(int number, const std::string &compositor)
{
    this->image = NULL;
    this->compositor = compositor;
    this->number = number;
    this->dataAvailable = false;
    this->bufferAvailable = true;
    this->keepRunning = true;
    this->externalPixelFormat = GL_BGRA;
}

TiledDisplayClient::~TiledDisplayClient()
{
}

void TiledDisplayClient::readBackImage(const osg::Camera &cam)
{

    fillLock.lock();

    //cerr << "TiledDisplayClient::readBackImage info: reading" << endl;

    const osg::Viewport *vp = cam.getViewport();

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

    dataAvailable = true;
    bufferAvailable = false;

    fillLock.unlock();
}

bool TiledDisplayClient::isImageAvailable()
{
    int locked = fillLock.trylock();
    if (locked == 0 && bufferAvailable)
    {
        fillLock.unlock();
        return true;
    }
    else
    {
        if (locked == 0)
            fillLock.unlock();
        return false;
    }
}

void TiledDisplayClient::exit()
{
    keepRunning = false;
}
